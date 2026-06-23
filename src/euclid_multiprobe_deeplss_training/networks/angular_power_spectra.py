"""Angular power spectra features for partial-sky HEALPix maps."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import healpy as hp
import numpy as np
import torch
import torch.nn as nn


class AngularPowerSpectra(nn.Module):
    """Convert partial-sky HEALPix maps to auto- and cross-angular spectra.

    Inputs use the project-standard shape ``(batch, pixels, channels)``.  Each
    example is expanded to a zero-filled full-sky HEALPix map using the stored
    full-sky pixel indices, transformed with cuHPX ``SHTCUDA``, and reduced to
    all channel auto spectra followed by all unique cross spectra.
    """

    def __init__(
        self,
        nside: int,
        pixel_file: str | Path | torch.Tensor | np.ndarray | list[int],
        *,
        lmax: int | None = None,
        mmax: int | None = None,
        quad_weights: str = "ring",
        pixel_dataset: str | None = None,
    ) -> None:
        super().__init__()
        self.nside = int(nside)
        self.lmax = int(2 * nside + 1 if lmax is None else lmax)
        self.mmax = int(self.lmax if mmax is None else mmax)
        self.quad_weights = quad_weights

        pixel_indices = torch.as_tensor(_load_pixel_indices(pixel_file, pixel_dataset), dtype=torch.long)
        if pixel_indices.ndim != 1:
            raise ValueError("pixel_file must contain a one-dimensional list of HEALPix pixel indices")

        npix_full = hp.nside2npix(self.nside)
        if torch.any(pixel_indices < 0) or torch.any(pixel_indices >= npix_full):
            raise ValueError(f"pixel indices must be in [0, {npix_full}) for nside={self.nside}")

        self.register_buffer("pixel_indices", pixel_indices, persistent=False)
        self._sht: nn.Module | None = None
        self._isht: nn.Module | None = None

    @property
    def num_ells(self) -> int:
        """Number of output ell bins."""
        return self.lmax + 1

    @staticmethod
    def num_cls_channels(num_map_channels: int) -> int:
        """Return the number of auto plus unique cross spectra channels."""
        return num_map_channels * (num_map_channels + 1) // 2

    def forward(self, maps: torch.Tensor) -> torch.Tensor:
        """Return spectra with shape ``(batch, ell, cls_channels)``."""
        if maps.ndim != 3:
            raise ValueError(f"Expected maps with shape (batch, pixels, channels), got {tuple(maps.shape)}")

        batch_size, num_pixels, num_channels = maps.shape
        if num_pixels != self.pixel_indices.numel():
            raise ValueError(f"Input contains {num_pixels} pixels, but pixel_file contains {self.pixel_indices.numel()} indices")

        sht, _isht = self._transforms(maps.device)
        del _isht  # Created with SHTCUDA so callers get matching cuHPX transforms if needed later.

        spectra = []
        full_shape = (hp.nside2npix(self.nside), num_channels)
        pixel_indices = self.pixel_indices.to(device=maps.device)
        for example in maps:
            full_map = example.new_zeros(full_shape)
            full_map.index_copy_(0, pixel_indices, example)

            alms = [sht(full_map[:, channel]) for channel in range(num_channels)]
            cls = []
            for channel in range(num_channels):
                cls.append(_alm_cross_power(alms[channel], alms[channel], self.lmax, self.mmax).real)
            for first in range(num_channels):
                for second in range(first + 1, num_channels):
                    cls.append(_alm_cross_power(alms[first], alms[second], self.lmax, self.mmax).real)
            spectra.append(torch.stack(cls, dim=-1))

        return torch.stack(spectra, dim=0)

    def _transforms(self, device: torch.device) -> tuple[nn.Module, nn.Module]:
        if self._sht is None or self._isht is None:
            try:
                from cuhpx import SHTCUDA, iSHTCUDA
            except ImportError as exc:
                raise ImportError("AngularPowerSpectra requires the cuhpx package with SHTCUDA and iSHTCUDA") from exc

            self._sht = SHTCUDA(self.nside, lmax=self.lmax, mmax=self.mmax, quad_weights=self.quad_weights)
            self._isht = iSHTCUDA(self.nside, lmax=self.lmax, mmax=self.mmax)
        if hasattr(self._sht, "to"):
            self._sht = self._sht.to(device)
        if hasattr(self._isht, "to"):
            self._isht = self._isht.to(device)
        return self._sht, self._isht


def _load_pixel_indices(pixel_file: str | Path | torch.Tensor | np.ndarray | list[int], pixel_dataset: str | None) -> Any:
    if isinstance(pixel_file, torch.Tensor | np.ndarray | list):
        return pixel_file

    path = Path(pixel_file)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path)
    if suffix == ".npz":
        loaded = np.load(path)
        key = pixel_dataset or next(iter(loaded.files))
        return loaded[key]
    if suffix in {".h5", ".hdf5"}:
        with h5py.File(path, "r") as handle:
            key = pixel_dataset or next(iter(handle.keys()))
            return handle[key][...]
    return np.loadtxt(path, dtype=np.int64)


def _alm_cross_power(first: torch.Tensor, second: torch.Tensor, lmax: int, mmax: int) -> torch.Tensor:
    if first.shape != second.shape:
        raise ValueError(f"Alm tensors must have matching shapes, got {tuple(first.shape)} and {tuple(second.shape)}")
    if first.ndim < 2:
        raise ValueError("Expected cuHPX SHTCUDA to return at least a two-dimensional (ell, m) tensor")

    ell_count = min(lmax + 1, first.shape[-2])
    m_count = min(mmax + 1, first.shape[-1])
    dtype = torch.promote_types(first.real.dtype, second.real.dtype)
    cls = torch.empty(ell_count, dtype=dtype, device=first.device)

    for ell in range(ell_count):
        max_m = min(ell, m_count - 1)
        terms = first[..., ell, : max_m + 1] * second[..., ell, : max_m + 1].conj()
        weighted = terms[..., 0] if max_m == 0 else terms[..., 0] + 2 * terms[..., 1:].sum(dim=-1)
        cls[ell] = weighted.real / (2 * ell + 1)
    return cls
