"""Convolutional encoder for pyracorr auto- and cross-correlations."""

from __future__ import annotations

import healpy as hp
import torch
from pyracorr import PyracorrFastFootprint
from torch import nn

from .transformer_corrs import get_footprint_indices


class ResidualBlock1D(nn.Module):
    """A pair of length-preserving convolutions with a residual connection."""

    def __init__(self, channels: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.layers = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.layers(x))


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

class ConvolutionalResidualCorrNetwork(nn.Module):
    """Encode part-sky maps with a small CNN over their correlations.

    Inputs have shape ``(batch, pixels, channels)``. The input maps and their
    weights are prepared in the same way as for
    :class:`~.transformer_corrs.ShiftedWindowTransformerCorrNetwork`, then
    pyracorr computes every auto- and cross-correlation. Strided convolutions
    reduce the correlation-bin axis, residual blocks refine the
    representation, and global average pooling produces a fixed-size input for
    the embedding head.
    """

    tag = "corr_cnn"

    def __init__(
        self,
        *,
        indices: list[int] | torch.Tensor,
        nside: int,
        nside_down: int,
        num_channels: int,
        spins: list[int],
        embed_dim: int,
        device: torch.device | str | None = None,
        weight_function: callable | None = None,
        preprocess_function: callable | None = None,
        inner_channels: int = 32,
        downsampling_layers: int = 3,
        residual_layers: int = 3,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.weight_function = weight_function
        self.preprocess_function = preprocess_function
        self.embed_dim = int(embed_dim)
        self.num_channels = int(num_channels)
        if self.num_channels <= 0:
            raise ValueError("num_channels must be positive.")
        if inner_channels <= 0:
            raise ValueError("inner_channels must be positive.")
        if downsampling_layers < 1:
            raise ValueError("downsampling_layers must be positive.")
        if residual_layers < 0:
            raise ValueError("residual_layers cannot be negative.")
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer.")

        if len(spins) != self.num_channels:
            raise ValueError("spins must have the same length as num_channels.")

        self.nside = int(nside)
        self.nside_down = int(nside_down)
        level = hp.nside2order(self.nside)
        footprint_level = hp.nside2order(self.nside_down)

        self.num_corrs = 2 * (level + 1)
        self.correlator = PyracorrFastFootprint(
            L=level,
            spins=spins,
            R_footprint=footprint_level,
            matmul_precision="high",
            footprint_indices=get_footprint_indices(
                indices, level, footprint_level
            ),
            recompute_pairs=False,
            doublesets=True,
            pairs_filename=f"pairs_L{level:02d}.h5",
        ).to(device)

        self.num_channel_pairs = self.num_channels * (self.num_channels + 1) // 2
        padding = kernel_size // 2
        downsampling: list[nn.Module] = []
        in_channels = self.num_channel_pairs
        for _ in range(downsampling_layers):
            downsampling.extend(
                [
                    nn.Conv1d(
                        in_channels,
                        inner_channels,
                        kernel_size,
                        stride=2,
                        padding=padding,
                        bias=False,
                    ),
                    nn.BatchNorm1d(inner_channels),
                    nn.GELU(),
                ]
            )
            in_channels = inner_channels
        self.downsampling = nn.Sequential(*downsampling)
        self.residual_blocks = nn.Sequential(
            *[
                ResidualBlock1D(inner_channels, kernel_size, dropout)
                for _ in range(residual_layers)
            ]
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.regression_head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(inner_channels, self.embed_dim),
        )

        self.correlation_batch_norm = InputBatchNorm(
            self.num_corrs, self.num_channel_pairs
        )
        self.register_buffer(
            "upper_triangular_idx",
            torch.triu_indices(self.num_channels, self.num_channels),
            persistent=False,
        )

    def forward(self, maps: torch.Tensor) -> torch.Tensor:
        """Return one convolutional embedding for every set of input maps."""
        if maps.ndim != 3:
            raise ValueError(
                "maps must have shape (batch_size, num_pixels, num_channels)."
            )
        if maps.shape[-1] != self.num_channels:
            raise ValueError(
                f"Expected {self.num_channels} input channels, "
                f"got {maps.shape[-1]}."
            )

        weights = self.weight_function(maps)
        maps = self.preprocess_function(maps)

        maps = torch.movedim(maps, -1, 1).contiguous()
        weights = torch.movedim(weights, -1, 1).contiguous()
        correlations = self.correlator(maps, weights)
        correlations = correlations[
            :,
            self.upper_triangular_idx[0],
            self.upper_triangular_idx[1],
            :,
        ].transpose(1, 2)
        correlations = self.correlation_batch_norm(correlations)

        features = self.downsampling(correlations.transpose(1, 2))
        features = self.residual_blocks(features)
        return self.regression_head(self.pool(features))
