import math
import torch
import torch.nn as nn


def make_channel_dims(base_embed_dim, num_nested_levels, growth):
    """
    Returns the channel dimension at each resolution level.

    There are M nested levels, so there are M merges.

    Example with M = 4 and base_embed_dim = 64:

        constant:
            [64, 64, 64, 64, 64]

        double:
            [64, 128, 256, 512, 1024]

        full:
            [64, 256, 1024, 4096, 16384]
    """
    if growth == "constant":
        factor = 1
    elif growth == "double":
        factor = 2
    elif growth == "full":
        factor = 4
    else:
        raise ValueError(
            "growth must be one of: 'constant', 'double', 'full'"
        )

    dims = [base_embed_dim]

    for _ in range(num_nested_levels):
        dims.append(dims[-1] * factor)

    return dims


class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4):
        super().__init__()

        hidden_dim = dim * mlp_ratio

        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    Standard transformer block over a sequence.

    Input:
        x: (B_like, S, D)

    where:
        B_like = any batch-like dimension
        S      = sequence length
        D      = feature/channel dimension
    """

    def __init__(self, dim, num_heads, mlp_ratio=4):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True,
        )

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio)

    def forward(self, x):
        shortcut = x

        x_norm = self.norm1(x)

        attn_out, _ = self.attn(
            x_norm,
            x_norm,
            x_norm,
            need_weights=False,
        )

        x = shortcut + attn_out
        x = x + self.mlp(self.norm2(x))

        return x


class NestedLocalWindowBlock(nn.Module):
    """
    Local attention over the last few nested resolution dimensions.

    Input:
        x: (B, N, 4, 4, ..., 4, D)

    Example:

        x: (B, N, 4, 4, 4, 4, D)

    If window_levels = 3, attention is applied over:

        4 × 4 × 4 = 64 tokens

    Internally:

        (B, N, 4, 4, 4, 4, D)
            ↓
        (B * N * 4, 64, D)
            ↓ attention
        (B, N, 4, 4, 4, 4, D)

    This does not reshape the data into a 2D image.
    It only flattens local nested windows into sequences.
    """

    def __init__(
        self,
        dim,
        num_heads,
        window_levels=3,
        mlp_ratio=4,
    ):
        super().__init__()

        if window_levels < 1:
            raise ValueError("window_levels must be >= 1")

        self.window_levels = window_levels

        self.block = TransformerBlock(
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
        )

    def forward(self, x):
        """
        x: (B, N, 4, 4, ..., 4, D)
        """
        num_nested_levels = x.ndim - 3

        if num_nested_levels <= 0:
            raise ValueError(
                "NestedLocalWindowBlock needs at least one nested resolution dimension."
            )

        levels_used = min(self.window_levels, num_nested_levels)

        original_shape = x.shape
        D = x.shape[-1]

        # Shape of the local nested window.
        #
        # Example:
        #   x: (B, N, 4, 4, 4, 4, D)
        #   levels_used = 3
        #   window_shape = (4, 4, 4)
        #   sequence_length = 64
        window_shape = x.shape[-levels_used - 1 : -1]

        for size in window_shape:
            if size != 4:
                raise ValueError(
                    "Every nested resolution dimension must have size 4."
                )

        sequence_length = math.prod(window_shape)

        # Flatten local nested window into a sequence:
        #
        #   (..., 4, 4, 4, D) -> (..., 64, D)
        x = x.contiguous().reshape(-1, sequence_length, D)

        x = self.block(x)

        # Restore nested tensor shape.
        x = x.reshape(original_shape)

        return x


class NestedPatchMerge4(nn.Module):
    """
    Merge the last nested resolution dimension.

    Input:
        x: (B, N, 4, 4, ..., 4, in_dim)

    Output:
        x: (B, N, 4, 4, ..., out_dim)

    The final nested dimension has size 4.

    For each parent token:

        4 child tokens × in_dim features = 4 * in_dim features

    Then:

        4 * in_dim -> out_dim

    The value of out_dim depends on the channel growth strategy.
    """

    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.norm = nn.LayerNorm(4 * in_dim)
        self.reduction = nn.Linear(4 * in_dim, out_dim)

    def forward(self, x):
        """
        x: (B, N, 4, 4, ..., 4, in_dim)
        """
        if x.ndim < 4:
            raise ValueError(
                "NestedPatchMerge4 needs at least one nested resolution dimension."
            )

        if x.shape[-2] != 4:
            raise ValueError("The last nested dimension must have size 4.")

        if x.shape[-1] != self.in_dim:
            raise ValueError(
                f"Expected last dimension {self.in_dim}, got {x.shape[-1]}."
            )

        # Everything except the final nested dimension and channel dimension.
        #
        # Example:
        #   x:            (B, N, 4, 4, 4, D)
        #   prefix_shape: (B, N, 4, 4)
        prefix_shape = x.shape[:-2]

        # Concatenate the 4 children into the channel dimension:
        #
        #   (B, N, 4, 4, 4, D)
        #       ->
        #   (B, N, 4, 4, 4D)
        x = x.contiguous().reshape(*prefix_shape, 4 * self.in_dim)

        x = self.norm(x)
        x = self.reduction(x)

        return x

class NestedHierarchicalLocalWindowTransformer(nn.Module):
    """
    Hierarchical Local Window Transformer for nested tensors.

    Input:
        x: (B, C, N, 4, 4, ..., 4)

    where:
        B = batch size
        C = input channels
        N = number of top-level/basic patches
        M = num_nested_levels
        each nested resolution dimension has size 4

    Internal representation:
        x: (B, N, 4, 4, ..., 4, D)

    Processing:
        input projection
        -> local nested attention
        -> patch merge
        -> local nested attention
        -> patch merge
        -> ...
        -> final tensor of shape (B, N, D_final)
        -> global attention over N tokens
        -> pooling over N
        -> prediction head

    The final global attention operates over N tokens, so internally it has
    an N × N attention matrix per head.
    """

    def __init__(
        self,
        in_channels,
        num_outputs,
        num_nested_levels,
        base_embed_dim=128,
        growth="constant",
        num_heads=4,
        window_levels=3,
        local_blocks_per_level=1,
        global_blocks=1,
        mlp_ratio=4,
    ):
        super().__init__()

        if num_nested_levels < 0:
            raise ValueError("num_nested_levels must be >= 0")

        if local_blocks_per_level < 0:
            raise ValueError("local_blocks_per_level must be >= 0")

        if global_blocks < 1:
            raise ValueError("global_blocks must be >= 1")

        self.in_channels = in_channels
        self.num_nested_levels = num_nested_levels
        self.base_embed_dim = base_embed_dim
        self.growth = growth
        self.num_heads = num_heads
        self.window_levels = window_levels

        # Channel dimensions at each resolution level.
        #
        # Length is num_nested_levels + 1.
        #
        # Example with M = 4:
        #   channel_dims[0] = channels after input projection
        #   channel_dims[1] = channels after merge 1
        #   channel_dims[2] = channels after merge 2
        #   channel_dims[3] = channels after merge 3
        #   channel_dims[4] = channels after merge 4
        self.channel_dims = make_channel_dims(
            base_embed_dim=base_embed_dim,
            num_nested_levels=num_nested_levels,
            growth=growth,
        )

        for dim in self.channel_dims:
            if dim % num_heads != 0:
                raise ValueError(
                    f"Channel dimension {dim} must be divisible by num_heads={num_heads}."
                )

        # Project input channels C -> base_embed_dim.
        #
        # Applied independently to every fine nested token.
        self.input_proj = nn.Linear(in_channels, self.channel_dims[0])

        # One local stage per nested resolution level.
        #
        # Stage i operates before merge i.
        # Its channel dimension is channel_dims[i].
        self.local_stages = nn.ModuleList()

        for level in range(num_nested_levels):
            dim = self.channel_dims[level]

            blocks = nn.ModuleList(
                [
                    NestedLocalWindowBlock(
                        dim=dim,
                        num_heads=num_heads,
                        window_levels=window_levels,
                        mlp_ratio=mlp_ratio,
                    )
                    for _ in range(local_blocks_per_level)
                ]
            )

            self.local_stages.append(blocks)

        # One patch merge per nested level.
        #
        # Merge i maps:
        #   channel_dims[i] -> channel_dims[i + 1]
        self.patch_merges = nn.ModuleList()

        for level in range(num_nested_levels):
            in_dim = self.channel_dims[level]
            out_dim = self.channel_dims[level + 1]

            self.patch_merges.append(
                NestedPatchMerge4(
                    in_dim=in_dim,
                    out_dim=out_dim,
                )
            )

        # Final global attention over the N basic-patch tokens.
        final_dim = self.channel_dims[-1]

        self.global_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=final_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                )
                for _ in range(global_blocks)
            ]
        )

        self.norm = nn.LayerNorm(final_dim)
        self.head = nn.Linear(final_dim, num_outputs)

    def forward(self, x):
        """
        Input:
            x: (B, C, N, 4, 4, ..., 4)

        Output:
            y: (B, num_outputs)
        """
        expected_ndim = 3 + self.num_nested_levels

        if x.ndim != expected_ndim:
            raise ValueError(
                f"Expected input with {expected_ndim} dims: "
                f"(B, C, N, 4, ..., 4), got shape {tuple(x.shape)}."
            )

        B, C, N = x.shape[:3]

        if C != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, got {C}."
            )

        for size in x.shape[3:]:
            if size != 4:
                raise ValueError(
                    "Every nested resolution dimension must have size 4."
                )

        # Move channels to the end:
        #
        #   (B, C, N, 4, 4, ..., 4)
        #       ->
        #   (B, N, 4, 4, ..., 4, C)
        x = x.movedim(1, -1).contiguous()

        # Project C -> base_embed_dim:
        #
        #   (B, N, 4, 4, ..., 4, C)
        #       ->
        #   (B, N, 4, 4, ..., 4, D0)
        x = self.input_proj(x)

        # Hierarchical local processing.
        #
        # At level i:
        #
        #   x has shape:
        #       (B, N, 4, ..., 4, channel_dims[i])
        #
        #   local attention keeps the same shape.
        #
        #   patch merge removes one nested dimension and changes channels:
        #       channel_dims[i] -> channel_dims[i + 1]
        for level in range(self.num_nested_levels):
            for block in self.local_stages[level]:
                x = block(x)

            x = self.patch_merges[level](x)

        # After all merges:
        #
        #   x: (B, N, final_dim)
        #
        # Apply final global attention over N tokens.
        for block in self.global_blocks:
            x = block(x)

        x = self.norm(x)

        # Pool over the N basic patches:
        #
        #   (B, N, final_dim) -> (B, final_dim)
        x = x.mean(dim=1)

        # Classification or regression head:
        #
        #   (B, final_dim) -> (B, num_outputs)
        x = self.head(x)

        return x
