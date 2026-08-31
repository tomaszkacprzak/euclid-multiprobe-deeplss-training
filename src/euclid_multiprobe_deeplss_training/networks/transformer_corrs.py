from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from pyracorr import PyracorrFastFootprint


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ShiftedWindowBlock1D(nn.Module):
    """
    Pre-norm 1D transformer block with local window attention.

    Input/output:
        x: [batch_size, sequence_length, embedding_dim]

    For shift_size == 0:
        [0 ........ W-1] [W ........ 2W-1] ...

    For shift_size == W // 2:
        [edge] [W/2 ........ W/2+W-1] ... [edge]
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int,
        shift_size: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if dim % num_heads != 0:
            raise ValueError(
                f"dim={dim} must be divisible by num_heads={num_heads}"
            )

        if window_size < 1:
            raise ValueError("window_size must be positive")

        if not 0 <= shift_size < window_size:
            raise ValueError(
                "shift_size must satisfy 0 <= shift_size < window_size"
            )

        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)

        self.attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attention_dropout,
            batch_first=True,
        )

        self.attention_output_dropout = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(dim)

        self.feed_forward = FeedForward(
            dim=dim,
            hidden_dim=int(dim * mlp_ratio),
            dropout=dropout,
        )

    def _window_attention(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform self-attention independently within each local window.

        Instead of cyclically rolling the sequence and constructing a
        pairwise wraparound mask, we offset the window grid with left
        padding. For valid tokens, this is equivalent to cyclic shifting
        followed by masking the wrapped edge regions.
        """
        batch_size, length, dim = x.shape
        window_size = self.window_size

        # Regular windows use no left padding.
        #
        # For a shift S, adding W-S tokens on the left produces:
        #
        #   [padding, x[0:S]]
        #   [x[S:S+W]]
        #   [x[S+W:S+2W]]
        #   ...
        #
        left_pad = (
            0
            if self.shift_size == 0
            else window_size - self.shift_size
        )

        # Make the padded length divisible by window_size.
        right_pad = (-(left_pad + length)) % window_size

        # Pad sequence dimension, not embedding dimension.
        x = F.pad(
            x,
            pad=(0, 0, left_pad, right_pad),
            mode="constant",
            value=0.0,
        )

        padded_length = x.shape[1]
        num_windows = padded_length // window_size

        # True for real tokens; False for padding.
        valid = torch.ones(
            batch_size,
            length,
            dtype=torch.bool,
            device=x.device,
        )
        valid = F.pad(
            valid,
            pad=(left_pad, right_pad),
            mode="constant",
            value=False,
        )

        # Convert windows into additional batch elements:
        #
        # [B, num_windows, W, D]
        #     -> [B * num_windows, W, D]
        #
        windows = x.reshape(
            batch_size,
            num_windows,
            window_size,
            dim,
        ).reshape(
            batch_size * num_windows,
            window_size,
            dim,
        )

        valid_windows = valid.reshape(
            batch_size,
            num_windows,
            window_size,
        ).reshape(
            batch_size * num_windows,
            window_size,
        )

        attended_windows, _ = self.attention(
            query=windows,
            key=windows,
            value=windows,
            # For MultiheadAttention, True means this key is ignored.
            key_padding_mask=~valid_windows,
            need_weights=False,
        )

        # Reverse the window partition.
        x = attended_windows.reshape(
            batch_size,
            num_windows,
            window_size,
            dim,
        ).reshape(
            batch_size,
            padded_length,
            dim,
        )

        # Remove left and right padding.
        return x[:, left_pad : left_pad + length, :]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm attention residual.
        x = x + self.attention_output_dropout(
            self._window_attention(self.norm1(x))
        )

        # Pre-norm feed-forward residual.
        x = x + self.feed_forward(self.norm2(x))

        return x


class ShiftedWindowTransformerRegressor(nn.Module):
    """
    1D shifted-window transformer for multi-output regression.

    Input:
        x: [batch_size, num_dimensions, num_channels]

    Output:
        y: [batch_size, embed_dim]
    """

    def __init__(
        self,
        input_channels: int,
        embed_dim: int,
        max_length: int,
        inner_embed_dim: int = 32,
        depth: int = 6,
        num_heads: int = 8,
        window_size: int = 64,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        attention_dropout: float = 0.0,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()

        if input_channels < 1:
            raise ValueError("input_channels must be positive")

        if embed_dim < 1:
            raise ValueError("embed_dim must be positive")

        if max_length < 1:
            raise ValueError("max_length must be positive")

        if depth < 1:
            raise ValueError("depth must be positive")

        if inner_embed_dim % num_heads != 0:
            raise ValueError(
                "inner_embed_dim must be divisible by num_heads"
            )

        self.input_channels = input_channels
        self.max_length = max_length

        # Project each position's input-channel vector into the model
        # embedding dimension:
        #
        # [B, L, input_channels] -> [B, L, inner_embed_dim]
        #
        self.input_projection = nn.Linear(
            input_channels,
            inner_embed_dim,
        )

        # Absolute position information. This is useful because window
        # attention alone does not identify the absolute sequence position.
        self.position_embedding = nn.Parameter(
            torch.empty(1, max_length, inner_embed_dim)
        )
        nn.init.trunc_normal_(
            self.position_embedding,
            std=0.02,
        )

        self.input_dropout = nn.Dropout(dropout)

        # Even blocks: regular windows.
        # Odd blocks: windows shifted by half their size.
        self.blocks = nn.ModuleList(
            [
                ShiftedWindowBlock1D(
                    dim=inner_embed_dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=(
                        0
                        if block_index % 2 == 0
                        else window_size // 2
                    ),
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                )
                for block_index in range(depth)
            ]
        )

        self.final_norm = nn.LayerNorm(inner_embed_dim)

        # Map the globally pooled representation to M regression values.
        self.regression_head = nn.Sequential(
            nn.Linear(inner_embed_dim, inner_embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_embed_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                "Expected x with shape "
                "[batch_size, num_dimensions, num_channels]"
            )

        _, length, channels = x.shape

        if length < 1:
            raise ValueError("num_dimensions must be positive")

        if channels != self.input_channels:
            raise ValueError(
                f"Expected {self.input_channels} input channels, "
                f"got {channels}"
            )

        if length > self.max_length:
            raise ValueError(
                f"Input length {length} exceeds "
                f"max_length={self.max_length}"
            )

        x = self.input_projection(x)

        x = (
            x
            + self.position_embedding[:, :length, :]
        )

        x = self.input_dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)

        # Global aggregation across num_dimensions.
        pooled = x.mean(dim=1)

        # No output activation: appropriate for unconstrained regression.
        return self.regression_head(pooled)



class InputBatchNorm(nn.Module):
    def __init__(self, num_dimensions, num_channels):
        super().__init__()

        self.num_dimensions = num_dimensions
        self.num_channels = num_channels

        self.bn = nn.BatchNorm1d(num_dimensions * num_channels)

    def forward(self, x):
        # x: (B, D, C)
        original_shape = x.shape

        # (B, D, C) -> (B, D*C)
        x = x.reshape(x.shape[0], -1)

        # Normalize each of the D*C features across the batch
        x = self.bn(x)

        # (B, D*C) -> (B, D, C)
        x = x.reshape(original_shape)

        return x

class ShiftedWindowTransformerCorrNetwork(nn.Module):
    """Encode part-sky maps by applying a transformer to their correlations.

    As in :class:`~.transformer_cls.ShiftedWindowTransformerClsNetwork`, inputs have shape
    ``(batch, pixels, channels)``, and pyracorr computes every correlation.  
    The resulting correlation sequence is passed to a 
    :class:`ShiftedWindowTransformerRegressor`, with one input feature per
    correlation.
    """

    tag = "corr_transformer"

    def __init__(
        self,
        *,
        indices: list[int] | torch.Tensor,
        nside: int,
        num_channels: int,
        embed_dim: int,
        lmax: int | None = None,
        sub_batch_size: int = 16,
        device: torch.device | str | None = None,
        unstack_function: callable | None = None,
        inner_embed_dim: int = 128,
        depth: int = 6,
        num_heads: int = 8,
        window_size: int = 64,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.unstack_function = unstack_function
        self.embed_dim = int(embed_dim)
        self.num_channels = int(num_channels)
        if self.num_channels <= 0:
            raise ValueError("num_channels must be positive.")

        # TODO: Implement this
        
        raise NotImplementedError("PyracorrFastFootprint is not implemented yet")

        self.num_correlations = self.num_channels * (self.num_channels + 1) // 2
        self.transformer = ShiftedWindowTransformerRegressor(
            input_channels=self.num_correlations,
            embed_dim=self.embed_dim,
            max_length=self.lmax,
            inner_embed_dim=inner_embed_dim,
            depth=depth,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            attention_dropout=attention_dropout,
        )

        self.correlation_batch_norm = InputBatchNorm(self.lmax, self.num_correlations)


    def forward(self, maps: torch.Tensor) -> torch.Tensor:
        """Return one transformer embedding for every set of input maps."""
        if maps.ndim != 3:
            raise ValueError(
                "maps must have shape (batch_size, num_pixels, num_channels)."
            )
        if maps.shape[-1] != self.num_channels:
            raise ValueError(
                f"Expected {self.num_channels} input channels, "
                f"got {maps.shape[-1]}."
            )

        # Pyracorr accepts shape (batch, channels, pixels)
        maps  = x.movedim(-1, 1)
        maps.contiguous()
        correlations = self.corr(maps)

        # TODO: Selected upper triangular part of the correlation matrix, and flatten it
        

        return self.transformer(correlations)
