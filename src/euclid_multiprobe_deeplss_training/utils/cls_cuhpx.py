"""Angular power spectra for scalar full-sky HEALPix maps using cuHPX."""

from __future__ import annotations

from typing import Literal

import cuhpx
import torch
from cuhpx import SHTCUDA
from torch import nn


class ClsCuHPX(nn.Module):
    """Compute a scalar angular power spectrum with a cuHPX SHT.

    ``lmax`` and ``mmax`` follow the cuHPX convention: they are array extents,
    not inclusive maximum multipoles.  Thus, the returned multipoles are
    ``ell = 0, ..., lmax - 1``.  A complete full-sky spectrum needs every
    non-negative m mode, so ``mmax`` must equal ``lmax``.

    Parameters
    ----------
    nside:
        HEALPix NSIDE of each input map.
    lmax:
        Number of multipoles in the result.
    mmax:
        Number of non-negative m modes supplied to cuHPX. Defaults to
        ``lmax`` and must equal it.
    input_order:
        Pixel ordering of the input maps. cuHPX consumes RING-ordered maps;
        NESTED maps are remapped before the transform.
    quad_weights, norm, csphase:
        Options passed directly to :class:`cuhpx.SHTCUDA`.

    Notes
    -----
    Inputs have shape ``(batch_size, 12 * nside**2)`` and outputs have shape
    ``(batch_size, lmax)``.  The implementation is differentiable with
    respect to the input wherever the cuHPX transform is differentiable.
    """

    def __init__(
        self,
        nside: int,
        lmax: int,
        mmax: int | None = None,
        *,
        input_order: Literal["ring", "nest"] = "ring",
        quad_weights: str = "ring",
        norm: str = "ortho",
        csphase: bool = True,
    ) -> None:
        super().__init__()

        self.nside = int(nside)
        self.lmax = int(lmax)
        self.mmax = self.lmax if mmax is None else int(mmax)
        self.num_pixels = 12 * self.nside**2
        self.input_order = input_order

        if self.nside <= 0:
            raise ValueError("nside must be positive.")
        if self.lmax <= 0:
            raise ValueError("lmax must be positive.")
        if self.mmax != self.lmax:
            raise ValueError("mmax must equal lmax for a complete full-sky spectrum.")
        if input_order not in {"ring", "nest"}:
            raise ValueError("input_order must be either 'ring' or 'nest'.")

        self.sht = SHTCUDA(
            nside=self.nside,
            lmax=self.lmax,
            mmax=self.mmax,
            quad_weights=quad_weights,
            norm=norm,
            csphase=csphase,
        )

        ell = torch.arange(self.lmax).unsqueeze(1)
        emm = torch.arange(self.mmax).unsqueeze(0)
        valid_lm = emm <= ell
        m_weights = torch.full((self.mmax,), 2.0)
        m_weights[0] = 1.0
        self.register_buffer("_cl_weights", valid_lm * m_weights, persistent=False)
        self.register_buffer(
            "_cl_denominator",
            2.0 * torch.arange(self.lmax) + 1.0,
            persistent=False,
        )

    def forward(self, maps: torch.Tensor) -> torch.Tensor:
        """Return ``C_ell`` for a batch of full-sky scalar maps."""
        if maps.ndim != 2:
            raise ValueError("maps must have shape (batch_size, num_pixels).")
        if maps.shape[1] != self.num_pixels:
            raise ValueError(
                f"Expected {self.num_pixels} pixels for nside={self.nside}, "
                f"got {maps.shape[1]}."
            )
        if not maps.is_floating_point():
            raise TypeError("maps must be a floating-point tensor.")
        if not maps.is_cuda:
            raise ValueError("maps must be a CUDA tensor for cuHPX SHTCUDA.")

        if self.input_order == "nest":
            maps = cuhpx.nest2ring(maps.contiguous(), self.nside)

        alms = self.sht(maps.contiguous())
        if alms.shape[-2:] != (self.lmax, self.mmax):
            raise RuntimeError(
                "cuHPX returned an unexpected alm shape: "
                f"expected (..., {self.lmax}, {self.mmax}), got {tuple(alms.shape)}."
            )

        power = alms.real.square() + alms.imag.square()
        weights = self._cl_weights.to(device=power.device, dtype=power.dtype)
        denominator = self._cl_denominator.to(device=power.device, dtype=power.dtype)
        return (power * weights).sum(dim=-1) / denominator
