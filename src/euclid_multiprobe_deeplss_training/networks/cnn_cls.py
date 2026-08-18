"""Convolutional encoder for auto- and cross-power spectra."""

from __future__ import annotations

import torch
from torch import nn

from ..utils.cls_cuhpx import PartSkyCls


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

class ConvolutionalResidualClsNetwork(nn.Module):
    """Encode part-sky maps with a small CNN over their power spectra.

    Inputs have shape ``(batch, pixels, channels)``. ``PartSkyCls`` first
    computes every auto- and cross-spectrum. Strided convolutions then reduce
    the ell axis, residual blocks refine the representation, and global average
    pooling produces a fixed-size input for the embedding head.
    """

    tag = "cls_cnn"

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
        inner_channels: int = 32,
        downsampling_layers: int = 3,
        residual_layers: int = 3,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.unstack_function = unstack_function
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

        self.lmax = 3 * int(nside) if lmax is None else int(lmax)
        self.cls = PartSkyCls(
            torch.as_tensor(indices, dtype=torch.long),
            nside=nside,
            lmax=self.lmax,
            sub_batch_size=sub_batch_size,
            device=device,
        )

        self.num_spectra = self.num_channels * (self.num_channels + 1) // 2
        padding = kernel_size // 2
        downsampling: list[nn.Module] = []
        in_channels = self.num_spectra
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

        self.spectrum_batch_norm = InputBatchNorm(self.lmax, self.num_spectra)

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

        channel_maps = self.unstack_function(maps)
        spectra = self.cls(*channel_maps)
        spectra = self.spectrum_batch_norm(spectra)
        features = self.downsampling(spectra.transpose(1, 2))
        features = self.residual_blocks(features)
        return self.regression_head(self.pool(features))
