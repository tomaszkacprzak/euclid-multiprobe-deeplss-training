"""Deeper nested hierarchical local-window transformer variants.

This module intentionally mirrors :mod:`nested_transfomer` while adding two
configuration-friendly depth stabilizers:

* pre-normalized transformer blocks, enabled by default; and
* DropPath / stochastic depth on residual branches.
"""

import torch
import torch.nn as nn

from .nested_transfomer import MLP, NestedPatchMerge4, make_channel_dims


class DropPath(nn.Module):
    """Drop residual paths per sample during training.

    The implementation follows stochastic depth: entire residual branches are
    randomly zeroed per batch item and surviving branches are rescaled by the
    keep probability so expected activations are preserved.
    """

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        if drop_prob < 0.0 or drop_prob >= 1.0:
            raise ValueError("drop_prob must satisfy 0 <= drop_prob < 1")
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        return x.div(keep_prob) * random_tensor


class DeepTransformerBlock(nn.Module):
    """Transformer block with configurable pre-norm and stochastic depth."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: int = 4,
        drop_path: float = 0.0,
        pre_norm: bool = True,
        residual_dropout: float = 0.0,
    ):
        super().__init__()
        self.pre_norm = pre_norm
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio)
        self.residual_dropout = nn.Dropout(residual_dropout) if residual_dropout > 0.0 else nn.Identity()
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pre_norm:
            attn_input = self.norm1(x)
            attn_out, _ = self.attn(attn_input, attn_input, attn_input, need_weights=False)
            x = x + self.drop_path(self.residual_dropout(attn_out))
            x = x + self.drop_path(self.residual_dropout(self.mlp(self.norm2(x))))
            return x

        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm1(x + self.drop_path(self.residual_dropout(attn_out)))
        x = self.norm2(x + self.drop_path(self.residual_dropout(self.mlp(x))))
        return x


class DeepNestedLocalWindowBlock(nn.Module):
    """Local nested-window attention using :class:`DeepTransformerBlock`."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_levels: int = 3,
        mlp_ratio: int = 4,
        drop_path: float = 0.0,
        pre_norm: bool = True,
        residual_dropout: float = 0.0,
    ):
        super().__init__()
        if window_levels < 1:
            raise ValueError("window_levels must be >= 1")
        self.window_levels = window_levels
        self.block = DeepTransformerBlock(
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop_path=drop_path,
            pre_norm=pre_norm,
            residual_dropout=residual_dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        num_nested_levels = x.ndim - 3
        if num_nested_levels <= 0:
            raise ValueError("DeepNestedLocalWindowBlock needs at least one nested resolution dimension.")

        levels_used = min(self.window_levels, num_nested_levels)
        original_shape = x.shape
        dim = x.shape[-1]
        window_shape = x.shape[-levels_used - 1 : -1]

        for size in window_shape:
            if size != 4:
                raise ValueError("Every nested resolution dimension must have size 4.")

        sequence_length = 1
        for size in window_shape:
            sequence_length *= size

        x = x.contiguous().reshape(-1, sequence_length, dim)
        x = self.block(x)
        return x.reshape(original_shape)


class DeepNestedHierarchicalLocalWindowTransformer(nn.Module):
    """Nested hierarchical local-window transformer with depth controls.

    The tensor interface matches ``NestedHierarchicalLocalWindowTransformer``.
    Depth is controlled by ``local_blocks_per_level`` and ``global_blocks``;
    stochastic depth is controlled by ``drop_path_rate`` and
    ``drop_path_schedule``.
    """

    def __init__(
        self,
        in_channels,
        embed_dim,
        num_nested_levels,
        base_embed_dim=128,
        growth="constant",
        num_heads=4,
        window_levels=3,
        local_blocks_per_level=2,
        global_blocks=2,
        mlp_ratio=4,
        drop_path_rate=0.1,
        drop_path_schedule="linear",
        pre_norm=True,
        residual_dropout=0.0,
    ):
        super().__init__()
        if num_nested_levels < 0:
            raise ValueError("num_nested_levels must be >= 0")
        if local_blocks_per_level < 0:
            raise ValueError("local_blocks_per_level must be >= 0")
        if global_blocks < 1:
            raise ValueError("global_blocks must be >= 1")
        if drop_path_rate < 0.0 or drop_path_rate >= 1.0:
            raise ValueError("drop_path_rate must satisfy 0 <= drop_path_rate < 1")
        if drop_path_schedule not in {"linear", "constant"}:
            raise ValueError("drop_path_schedule must be 'linear' or 'constant'")

        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_nested_levels = num_nested_levels
        self.base_embed_dim = base_embed_dim
        self.growth = growth
        self.num_heads = num_heads
        self.window_levels = window_levels
        self.drop_path_rate = drop_path_rate
        self.drop_path_schedule = drop_path_schedule
        self.pre_norm = pre_norm
        self.residual_dropout = residual_dropout

        self.channel_dims = make_channel_dims(base_embed_dim, num_nested_levels, growth)
        for dim in self.channel_dims:
            if dim % num_heads != 0:
                raise ValueError(f"Channel dimension {dim} must be divisible by num_heads={num_heads}.")

        total_blocks = num_nested_levels * local_blocks_per_level + global_blocks
        if drop_path_schedule == "linear" and total_blocks > 1:
            drop_rates = torch.linspace(0.0, drop_path_rate, total_blocks).tolist()
        else:
            drop_rates = [drop_path_rate] * total_blocks
        drop_iter = iter(drop_rates)

        self.input_proj = nn.Linear(in_channels, self.channel_dims[0])
        self.local_stages = nn.ModuleList()
        for level in range(num_nested_levels):
            dim = self.channel_dims[level]
            self.local_stages.append(
                nn.ModuleList(
                    [
                        DeepNestedLocalWindowBlock(
                            dim=dim,
                            num_heads=num_heads,
                            window_levels=window_levels,
                            mlp_ratio=mlp_ratio,
                            drop_path=next(drop_iter),
                            pre_norm=pre_norm,
                            residual_dropout=residual_dropout,
                        )
                        for _ in range(local_blocks_per_level)
                    ]
                )
            )

        self.patch_merges = nn.ModuleList(
            [NestedPatchMerge4(self.channel_dims[level], self.channel_dims[level + 1]) for level in range(num_nested_levels)]
        )

        final_dim = self.channel_dims[-1]
        self.global_blocks = nn.ModuleList(
            [
                DeepTransformerBlock(
                    dim=final_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    drop_path=next(drop_iter),
                    pre_norm=pre_norm,
                    residual_dropout=residual_dropout,
                )
                for _ in range(global_blocks)
            ]
        )
        self.norm = nn.LayerNorm(final_dim)
        self.head = nn.Linear(final_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        expected_ndim = 3 + self.num_nested_levels
        if x.ndim != expected_ndim:
            raise ValueError(f"Expected input with {expected_ndim} dims: (B, C, N, 4, ..., 4), got shape {tuple(x.shape)}.")

        _, channels, _ = x.shape[:3]
        if channels != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, got {channels}.")
        for size in x.shape[3:]:
            if size != 4:
                raise ValueError("Every nested resolution dimension must have size 4.")

        x = x.movedim(1, -1).contiguous()
        x = self.input_proj(x)
        for level in range(self.num_nested_levels):
            for block in self.local_stages[level]:
                x = block(x)
            x = self.patch_merges[level](x)

        for block in self.global_blocks:
            x = block(x)

        x = self.norm(x)
        x = x.mean(dim=1)
        return self.head(x)
