import math
import torch
import torch.nn as nn


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
        x: (B, S, D)

    where:
        B = batch-like dimension
        S = sequence length
        D = embedding dimension
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

    Example with M = 4 and window_levels = 3:

        x: (B, N, 4, 4, 4, 4, D)

    The block applies attention over the last 3 nested dimensions:

        4 × 4 × 4 = 64 tokens

    So internally it becomes:

        (B * N * 4, 64, D)

    This is analogous to an 8 × 8 local attention window in a 2D image,
    because 8 × 8 = 64.
    """

    def __init__(
        self,
        dim,
        num_heads,
        window_levels=3,
        mlp_ratio=4,
    ):
        super().__init__()

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

        # Use at most self.window_levels, but if fewer levels remain,
        # attend over all remaining nested levels.
        levels_used = min(self.window_levels, num_nested_levels)

        original_shape = x.shape
        D = x.shape[-1]

        # The local attention window is made from the last levels_used
        # nested dimensions, each of size 4.
        #
        # Example:
        #   x shape = (B, N, 4, 4, 4, 4, D)
        #   levels_used = 3
        #   window_shape = (4, 4, 4)
        #   sequence length = 4 * 4 * 4 = 64
        window_shape = x.shape[-levels_used - 1 : -1]

        for size in window_shape:
            assert size == 4, "Every nested resolution dimension must have size 4."

        sequence_length = math.prod(window_shape)

        # Flatten every local nested window into a sequence.
        #
        # Example:
        #   (B, N, 4, 4, 4, 4, D)
        #       ->
        #   (B * N * 4, 64, D)
        #
        # This is not reshaping to an image-like representation.
        # It only creates local sequences for attention.
        x = x.reshape(-1, sequence_length, D)

        x = self.block(x)

        # Restore the nested tensor shape.
        x = x.reshape(original_shape)

        return x


class NestedPatchMerge4(nn.Module):
    """
    Merges the last nested resolution dimension.

    Input:
        x: (B, N, 4, 4, ..., 4, D)

    Output:
        x: (B, N, 4, 4, ..., D)

    The last nested dimension has size 4.

    For each parent token, we concatenate its 4 child tokens:

        4 children × D channels = 4D channels

    Then we project:

        4D -> D

    This removes one nested resolution level.
    """

    def __init__(self, dim):
        super().__init__()

        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, dim)

    def forward(self, x):
        """
        x: (B, N, 4, 4, ..., 4, D)
        """
        if x.ndim < 4:
            raise ValueError(
                "NestedPatchMerge4 needs at least one nested resolution dimension."
            )

        assert x.shape[-2] == 4, "The last nested dimension must have size 4."

        D = x.shape[-1]

        # Everything except the final nested dimension and channel dimension.
        #
        # Example:
        #   x:      (B, N, 4, 4, 4, D)
        #   prefix: (B, N, 4, 4)
        prefix_shape = x.shape[:-2]

        # Concatenate the 4 children into the channel dimension.
        #
        #   (B, N, 4, 4, 4, D)
        #       ->
        #   (B, N, 4, 4, 4D)
        x = x.reshape(*prefix_shape, 4 * D)

        x = self.norm(x)
        x = self.reduction(x)

        return x


class NestedHierarchicalLocalWindowTransformer(nn.Module):
    """
    Hierarchical Local Window Transformer for nested tensors.

    Expected input:

        x: (B, C, N, 4, 4, ..., 4)

    where:
        B = batch size
        C = input channels
        N = number of top-level/basic patches
        M = number of nested resolution levels

    The model internally uses:

        x: (B, N, 4, 4, ..., 4, D)

    Processing:

        nested local attention
        -> merge last nested level
        -> nested local attention
        -> merge last nested level
        -> ...
        -> final tensor of shape (B, N, D)
        -> global attention over N tokens
        -> pooling
        -> head

    The final global attention has an N × N attention matrix internally.
    """

    def __init__(
        self,
        in_channels,
        num_outputs,
        embed_dim=128,
        num_heads=4,
        window_levels=3,
        local_blocks_per_level=1,
        global_blocks=1,
        mlp_ratio=4,
    ):
        super().__init__()

        assert embed_dim % num_heads == 0

        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.window_levels = window_levels

        # Projects input channels C -> D.
        # This is applied independently to every nested token.
        self.input_proj = nn.Linear(in_channels, embed_dim)

        # Shared local blocks.
        # These are reused at every nested resolution level.
        self.local_blocks = nn.ModuleList(
            [
                NestedLocalWindowBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    window_levels=window_levels,
                    mlp_ratio=mlp_ratio,
                )
                for _ in range(local_blocks_per_level)
            ]
        )

        # Shared patch merge.
        # This is reused until all nested dimensions are removed.
        self.patch_merge = NestedPatchMerge4(embed_dim)

        # Final attention over the N top-level tokens.
        self.global_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                )
                for _ in range(global_blocks)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_outputs)

    def forward(self, x):
        """
        Input:
            x: (B, C, N, 4, 4, ..., 4)

        Output:
            y: (B, num_outputs)
        """
        if x.ndim < 3:
            raise ValueError(
                "Expected input with shape (B, C, N, 4, 4, ..., 4)."
            )

        B, C, N = x.shape[:3]
        nested_shape = x.shape[3:]

        assert C == self.in_channels

        for size in nested_shape:
            assert size == 4, "Every nested resolution dimension must have size 4."

        # Move channels to the end:
        #
        #   (B, C, N, 4, 4, ..., 4)
        #       ->
        #   (B, N, 4, 4, ..., 4, C)
        x = x.movedim(1, -1).contiguous()

        # Project C -> D:
        #
        #   (B, N, 4, 4, ..., 4, C)
        #       ->
        #   (B, N, 4, 4, ..., 4, D)
        x = self.input_proj(x)

        # Repeatedly:
        #   1. apply local nested attention
        #   2. merge away the last nested resolution level
        #
        # Stop when the tensor is:
        #   (B, N, D)
        while x.ndim > 3:
            for block in self.local_blocks:
                x = block(x)

            x = self.patch_merge(x)

        # Now:
        #
        #   x: (B, N, D)
        #
        # Apply final global attention over the N basic-patch tokens.
        for block in self.global_blocks:
            x = block(x)

        x = self.norm(x)

        # Pool over the N basic patches:
        #
        #   (B, N, D) -> (B, D)
        x = x.mean(dim=1)

        # Classification or regression head:
        #
        #   (B, D) -> (B, num_outputs)
        x = self.head(x)

        return x


if __name__ == "__main__":
    # Example:
    #
    # B = 2
    # C = 3 input channels
    # N = 16 top-level/basic patches
    # M = 4 nested resolution levels
    #
    # Input shape:
    #   (B, C, N, 4, 4, 4, 4)
    #
    # Total number of fine tokens per sample:
    #   N * 4^M = 16 * 4^4 = 4096

    model = NestedHierarchicalLocalWindowTransformer(
        in_channels=3,
        num_outputs=1,          # 1 for scalar regression; use K for K-class classification
        embed_dim=128,
        num_heads=4,
        window_levels=3,        # 4^3 = 64-token local attention window
        local_blocks_per_level=1,
        global_blocks=1,
        mlp_ratio=4,
    )

    x = torch.randn(2, 3, 16, 4, 4, 4, 4)

    y = model(x)

    print(y.shape)
    # torch.Size([2, 1])