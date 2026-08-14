"""Angular auto-power spectra for scalar and spin-2 HEALPix maps using cuHPX."""

from __future__ import annotations

from typing import Literal

import cuhpx
import torch
from cuhpx import SHTCUDA
from torch import nn

from .eb_cuhpx import CuHPXScalarRouteEB


class AutoClsCuHPX(nn.Module):
    """Compute scalar or spin-2 angular auto-power spectra with cuHPX.

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
        Pixel ordering of the input maps. Inputs are remapped as needed for
        the scalar and spin-2 transforms.
    quad_weights, norm, csphase:
        Options passed directly to :class:`cuhpx.SHTCUDA`.

    Notes
    -----
    Scalar inputs have shape ``(batch_size, 12 * nside**2)`` and produce
    ``(batch_size, lmax)`` spectra. Spin-2 inputs have shape
    ``(batch_size, 2, 12 * nside**2)``, with the two channels representing
    the HEALPix/CMB Q/U convention, and produce ``(batch_size, 2, lmax)``;
    output channels contain the E- and B-mode auto spectra respectively.
    Spin-2 transforms use :class:`CuHPXScalarRouteEB` and therefore require
    ``lmax >= 3``. The implementation is differentiable with respect to the
    input wherever the cuHPX transforms are differentiable.
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
        self.quad_weights = quad_weights

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

        # (L,) -> (L, 1).
        ell = torch.arange(self.lmax).unsqueeze(1)
        # (M,) -> (1, M).
        emm = torch.arange(self.mmax).unsqueeze(0)
        # (L, 1), (1, M) -> (L, M).
        valid_lm = emm <= ell
        # Scalar -> (M,).
        m_weights = torch.full((self.mmax,), 2.0)
        # (M,) -> (M,).
        m_weights[0] = 1.0
        # (L, M), (M,) -> (L, M).
        self.register_buffer("_cl_weights", valid_lm * m_weights, persistent=False)
        self.register_buffer(
            "_cl_denominator",
            # (L,) -> (L,).
            2.0 * torch.arange(self.lmax) + 1.0,
            persistent=False,
        )
        # Construct lazily so scalar-only use retains its existing requirements.
        self._spin_sht: CuHPXScalarRouteEB | None = None

    def forward(self, maps: torch.Tensor) -> torch.Tensor:
        """Return scalar ``C_ell`` or spin-2 E/B ``C_ell`` auto spectra."""
        self._validate_maps(maps)
        if maps.ndim == 2:
            return self._forward_scalar(maps)
        return self._forward_spin2(maps)

    def _validate_maps(self, maps: torch.Tensor) -> None:
        """Validate the shared scalar and spin-2 input requirements."""
        if maps.ndim not in {2, 3}:
            raise ValueError("maps must have shape (batch_size, num_pixels) or (batch_size, 2, num_pixels).")
        if maps.ndim == 3 and maps.shape[1] != 2:
            raise ValueError("spin-2 maps must have exactly two channels.")
        if maps.shape[-1] != self.num_pixels:
            raise ValueError(f"Expected {self.num_pixels} pixels for nside={self.nside}, got {maps.shape[-1]}.")
        if not maps.is_floating_point():
            raise TypeError("maps must be a floating-point tensor.")
        if not maps.is_cuda:
            raise ValueError("maps must be a CUDA tensor for cuHPX SHTCUDA.")

    def _forward_scalar(self, maps: torch.Tensor) -> torch.Tensor:
        """Calculate auto spectra for scalar maps."""
        if self.input_order == "nest":
            # (B, P) -> (B, P).
            maps = cuhpx.nest2ring(maps.contiguous(), self.nside)

        # (B, P) -> (B, L, M).
        alms = self.sht(maps.contiguous())
        self._validate_alm_shape(alms)
        return self._auto_spectrum(alms)

    def _forward_spin2(self, maps: torch.Tensor) -> torch.Tensor:
        """Calculate separate E- and B-mode auto spectra for spin-2 maps."""
        if self.lmax < 3:
            raise ValueError("lmax must be at least 3 for spin-2 maps.")

        if self.input_order == "ring":
            # (B, 2, P) -> (B, 2, P).
            maps = cuhpx.ring2nest(maps.contiguous(), self.nside)

        if self._spin_sht is None:
            self._spin_sht = CuHPXScalarRouteEB(
                nside=self.nside,
                lmax=self.lmax - 1,
                output_type="alm",
                quad_weights=self.quad_weights,
                device=maps.device,
                dtype=maps.dtype,
            )

        # (B, 2, P) -> (B, 2, L, M).
        eb_alms = self._spin_sht(maps.contiguous())
        self._validate_alm_shape(eb_alms)
        return self._auto_spectrum(eb_alms)

    def _validate_alm_shape(self, alms: torch.Tensor) -> None:
        """Check the common trailing cuHPX alm dimensions."""
        if alms.shape[-2:] != (self.lmax, self.mmax):
            raise RuntimeError(f"cuHPX returned an unexpected alm shape: expected (..., {self.lmax}, {self.mmax}), got {tuple(alms.shape)}.")

    def _auto_spectrum(self, alms: torch.Tensor) -> torch.Tensor:
        """Reduce scalar or channel-wise alms to their auto spectra."""
        # (..., L, M) -> (..., L, M).
        power = alms.real.square() + alms.imag.square()
        # (L, M) -> (L, M).
        weights = self._cl_weights.to(device=power.device, dtype=power.dtype)
        # (L,) -> (L,).
        denominator = self._cl_denominator.to(device=power.device, dtype=power.dtype)
        # (..., L, M) -> (..., L).
        return (power * weights).sum(dim=-1) / denominator
