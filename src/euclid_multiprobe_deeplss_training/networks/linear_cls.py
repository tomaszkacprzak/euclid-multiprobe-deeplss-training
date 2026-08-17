"""Power-spectrum encoder with a linear projection head."""

from __future__ import annotations

import torch
from torch import nn

from ..utils.cls_cuhpx import PartSkyCls


class LinearClsNetwork(nn.Module):
    """Encode part-sky maps through their auto- and cross-power spectra.

    The pipeline input is expected in ``(batch, pixels, channels)`` format.
    Each channel is treated as a scalar map.  :class:`PartSkyCls` computes all
    channel pairs.  When ``window_size`` is set, a fixed depthwise convolution
    averages non-overlapping windows along the ell dimension before the spectra
    are flattened for the learned linear projection.
    """

    tag = "cls_linear"

    def __init__(
        self,
        *,
        indices: list[int] | torch.Tensor,
        nside: int,
        num_channels: int,
        embed_dim: int,
        lmax: int | None = None,
        sub_batch_size: int = 1,
        unstack_function: callable | None = None,
        window_size: int | None = None,
    ) -> None:
        super().__init__()

        self.unstack_function = unstack_function

        self.num_channels = int(num_channels)
        if self.num_channels <= 0:
            raise ValueError("num_channels must be positive.")

        self.lmax = 3 * int(nside) if lmax is None else int(lmax)
        self.cls = PartSkyCls(
            torch.as_tensor(indices, dtype=torch.long),
            nside=nside,
            lmax=self.lmax,
            sub_batch_size=sub_batch_size,
        )
        num_spectra = self.num_channels * (self.num_channels + 1) // 2

        self.downsample: nn.Conv1d | None = None
        downsampled_lmax = self.lmax
        if window_size is not None:
            window_size = int(window_size)
            if window_size <= 0:
                raise ValueError("window_size must be positive.")
            if window_size > self.lmax:
                raise ValueError("window_size cannot exceed lmax.")

            # Treat the spectra as channels and apply the same, fixed averaging
            # kernel independently to each one.  Using the window as the stride
            # produces non-overlapping bins along the ell dimension.
            self.downsample = nn.Conv1d(
                num_spectra,
                num_spectra,
                kernel_size=window_size,
                stride=window_size,
                groups=num_spectra,
                bias=False,
            )
            with torch.no_grad():
                self.downsample.weight.fill_(1.0 / window_size)
            self.downsample.weight.requires_grad_(False)
            downsampled_lmax = self.lmax // window_size

        self.linear = nn.Linear(downsampled_lmax * num_spectra, embed_dim)

    def forward(self, maps: torch.Tensor) -> torch.Tensor:
        """Return one embedding for every item in ``maps``."""
        if maps.ndim != 3:
            raise ValueError("maps must have shape (batch_size, num_pixels, num_channels).")
        if maps.shape[-1] != self.num_channels:
            raise ValueError(f"Expected {self.num_channels} input channels, got {maps.shape[-1]}.")

        channel_maps = self.unstack_function(maps)

        spectra = self.cls(*channel_maps)
        if self.downsample is not None:
            spectra = self.downsample(spectra.transpose(1, 2)).transpose(1, 2)
        return self.linear(spectra.flatten(start_dim=1))
