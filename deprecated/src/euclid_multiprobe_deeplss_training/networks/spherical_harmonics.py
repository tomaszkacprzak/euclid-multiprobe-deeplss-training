from typing import Literal

import cuhpx
import torch
import torch.nn as nn
from cuhpx import SHTCUDA, iSHTCUDA

try:
    # Low-level CUDA extension. Some installs expose it; some only expose
    # the public cuhpx.nest2ring wrapper.
    from cuhpx import cuhpx_remap
except Exception:
    cuhpx_remap = None


class SphCuHpx(nn.Module):
    """Shared cuHPX spherical-harmonic support for spin-2 shear modules.

    The base class owns all common dimensions, HEALPix ordering metadata,
    cuHPX SHT/iSHT operators, spectral buffers, ring-geometry buffers, and
    flattened alm indexing used by the concrete modules.

    Public input layout
    -------------------
    g1, g2 : torch.Tensor
        Shape (batch_size, num_pix, num_channels), in HEALPix NESTED order
        by default.

    Output
    ------
    cl_ee, cl_bb : torch.Tensor
        Each has shape (batch_size, num_ells, num_channels).

    lmax / mmax convention
    ----------------------
    This follows the cuHPX bandlimit convention:

        ell = 0, ..., lmax - 1
        m   = 0, ..., mmax - 1

    Therefore the output ell axis has length `lmax`.

    If you think in healpy-style inclusive ell_max, pass

        lmax = ell_max + 1
        mmax = m_max + 1

    By default this uses lmax = 3*nside, corresponding to ell_max = 3*nside - 1.
    """

    def __init__(
        self,
        nside: int,
        lmax: int | None = None,
        mmax: int | None = None,
        *,
        batch_size: int | None = None,
        num_channels: int | None = None,
        input_order: Literal["nest", "ring"] = "nest",
        quad_weights: str = "ring",
        norm: str = "ortho",
        csphase: bool = True,
        sht_iter: int = 3,
    ):
        super().__init__()

        if nside <= 0 or (nside & (nside - 1)) != 0:
            raise ValueError("nside must be a positive power of two.")
        if batch_size is not None and batch_size <= 0:
            raise ValueError("batch_size must be positive when provided.")
        if num_channels is not None and num_channels <= 0:
            raise ValueError("num_channels must be positive when provided.")
        if input_order not in ("nest", "ring"):
            raise ValueError("input_order must be either 'nest' or 'ring'.")

        self.batch_size = int(batch_size) if batch_size is not None else None
        self.num_channels = int(num_channels) if num_channels is not None else None
        self.nside = int(nside)
        self.npix = 12 * self.nside * self.nside
        self.input_order = input_order
        self.sht_iter = int(sht_iter)
        if self.sht_iter < 0:
            raise ValueError("sht_iter must be non-negative.")

        # cuHPX convention: lmax is the number of ell rows, not inclusive ell_max.
        # Default ell_max = 3*nside - 1 -> num_ells = 3*nside.
        self.lmax = int(lmax) if lmax is not None else 3 * self.nside
        self.mmax = int(mmax) if mmax is not None else self.lmax

        if self.lmax < 3:
            raise ValueError("Need lmax >= 3 so that ell=2 exists.")
        if self.mmax < 1:
            raise ValueError("Need mmax >= 1.")
        if self.mmax > self.lmax:
            raise ValueError(
                "This implementation assumes mmax <= lmax. "
                "Use mmax=lmax for the full m range."
            )

        # cuHPX scalar transforms. cuHPX's README shows SHTCUDA/iSHTCUDA usage
        # with lmax and mmax passed at construction.
        self.sht = SHTCUDA(
            self.nside,
            lmax=self.lmax,
            mmax=self.mmax,
            quad_weights=quad_weights,
            norm=norm,
            csphase=csphase,
        )
        self.isht = iSHTCUDA(
            self.nside,
            lmax=self.lmax,
            mmax=self.mmax,
            quad_weights=quad_weights,
            norm=norm,
            csphase=csphase,
        )

        self._register_spectral_buffers()
        self._register_ring_geometry_buffers()
        self._register_alm_index_buffers()

    def _register_spectral_buffers(self) -> None:
        L = self.lmax
        M = self.mmax

        ell = torch.arange(L, dtype=torch.float64)
        emm = torch.arange(M, dtype=torch.float64)

        ell_lm = ell[:, None]          # (L, 1)
        m_lm = emm[None, :]            # (1, M)

        valid_lm = m_lm <= ell_lm      # (L, M)

        # alpha_{ell,m} used in
        #
        #   sin(theta) d_theta Y_{ell m}
        #       = ell alpha_{ell+1,m} Y_{ell+1,m}
        #       - (ell+1) alpha_{ell,m} Y_{ell-1,m}
        #
        # alpha_{ell,m} =
        # sqrt((ell^2 - m^2) / ((2 ell - 1)(2 ell + 1))).
        ell_alpha = torch.arange(L + 1, dtype=torch.float64)[:, None]  # (L+1, 1)
        denom = (2.0 * ell_alpha - 1.0) * (2.0 * ell_alpha + 1.0)
        numer = ell_alpha.square() - m_lm.square()
        alpha = torch.sqrt(torch.clamp(numer / denom, min=0.0))
        alpha[0, :] = 0.0

        # Spin-2 lowering normalization:
        #
        #   N_ell^2 = (ell-1) ell (ell+1) (ell+2).
        #
        # E_lm = xE_lm / N_ell, B_lm = xB_lm / N_ell for ell >= 2.
        spin2_norm2 = (ell - 1.0) * ell * (ell + 1.0) * (ell + 2.0)
        spin2_norm2 = torch.where(ell >= 2.0, spin2_norm2, torch.ones_like(spin2_norm2))

        # Real-map alm convention: cuHPX stores m >= 0.
        # Sum over negative m by doubling m > 0.
        m_weight = torch.full((M,), 2.0, dtype=torch.float64)
        m_weight[0] = 1.0
        cl_m_weight = valid_lm.to(torch.float64) * m_weight[None, :]

        cl_den = 2.0 * ell + 1.0
        ell_ge_2 = ell >= 2.0

        # cuHPX stores coefficients with an m-dependent longitudinal phase
        # relative to the HEALPix/healpy alm convention used by downstream
        # shear products.  Scalar power spectra are phase-invariant, but
        # returning complex E/B modes requires converting the phase explicitly.
        healpy_m_phase = torch.where(
            (emm.to(torch.long) % 2) == 0,
            torch.full_like(emm, -1.0),
            torch.ones_like(emm),
        )

        self.register_buffer("ell", ell, persistent=False)
        self.register_buffer("m", emm, persistent=False)
        self.register_buffer("ell_lm", ell_lm, persistent=False)
        self.register_buffer("m_lm", m_lm, persistent=False)
        self.register_buffer("valid_lm", valid_lm, persistent=False)
        self.register_buffer("alpha_lm", alpha, persistent=False)
        self.register_buffer("spin2_norm2", spin2_norm2, persistent=False)
        self.register_buffer("cl_m_weight", cl_m_weight, persistent=False)
        self.register_buffer("cl_den", cl_den, persistent=False)
        self.register_buffer("ell_ge_2", ell_ge_2, persistent=False)
        self.register_buffer("healpy_m_phase", healpy_m_phase, persistent=False)

    def _register_ring_geometry_buffers(self) -> None:
        theta = self._healpix_ring_theta(self.nside)

        sin_theta = torch.sin(theta)
        cos_theta = torch.cos(theta)

        csc_theta = 1.0 / sin_theta
        cot_theta = cos_theta / sin_theta
        csc2_theta = csc_theta.square()

        # Shape for broadcasting against maps shaped (B, C, component, Npix).
        self.register_buffer("csc_theta", csc_theta.view(1, 1, 1, -1), persistent=False)
        self.register_buffer("cot_theta", cot_theta.view(1, 1, 1, -1), persistent=False)
        self.register_buffer("csc2_theta", csc2_theta.view(1, 1, 1, -1), persistent=False)

    @staticmethod
    def _healpix_ring_theta(nside: int) -> torch.Tensor:
        """
        Vectorized HEALPix RING colatitude theta for every pixel, in RING order.
        No healpy dependency.
        """
        r = torch.arange(1, 4 * nside, dtype=torch.float64)  # 1-based ring index
        n = torch.tensor(float(nside), dtype=torch.float64)

        north = r < nside
        equat = (r >= nside) & (r <= 3 * nside)
        south = r > 3 * nside

        z = torch.empty_like(r)
        z[north] = 1.0 - r[north].square() / (3.0 * n.square())
        z[equat] = (2.0 * n - r[equat]) * 2.0 / (3.0 * n)
        z[south] = -1.0 + (4.0 * n - r[south]).square() / (3.0 * n.square())

        r_int = torch.arange(1, 4 * nside, dtype=torch.long)
        nphi = torch.where(
            r_int < nside,
            4 * r_int,
            torch.where(
                r_int <= 3 * nside,
                torch.full_like(r_int, 4 * nside),
                4 * (4 * nside - r_int),
            ),
        )

        theta_ring = torch.acos(torch.clamp(z, -1.0, 1.0))
        theta_pix = torch.repeat_interleave(theta_ring, nphi)

        expected = 12 * nside * nside
        if theta_pix.numel() != expected:
            raise RuntimeError(
                f"Internal HEALPix theta construction failed: "
                f"got {theta_pix.numel()} pixels, expected {expected}."
            )

        return theta_pix

    @staticmethod
    def _real_dtype_of(x: torch.Tensor) -> torch.dtype:
        return x.real.dtype if torch.is_complex(x) else x.dtype

    def _as_real_buffer(self, name: str, ref: torch.Tensor) -> torch.Tensor:
        buf = getattr(self, name)
        return buf.to(device=ref.device, dtype=self._real_dtype_of(ref))

    def _as_bool_buffer(self, name: str, ref: torch.Tensor) -> torch.Tensor:
        buf = getattr(self, name)
        return buf.to(device=ref.device)

    def _nest_to_ring_last_axis(self, x: torch.Tensor) -> torch.Tensor:
        """
        x is shaped (..., npix). Output is same shape, RING ordered.

        Use the low-level native batch function when exposed; otherwise use
        cuhpx.nest2ring, whose Python wrapper dispatches to the batch kernel
        for non-1D tensors.
        """
        x = x.contiguous()

        if cuhpx_remap is not None and hasattr(cuhpx_remap, "nest2ring_batch"):
            return cuhpx_remap.nest2ring_batch(x, self.nside, x.size(-1))

        return cuhpx.nest2ring(x, self.nside)

    def _sht_analysis(self, maps: torch.Tensor) -> torch.Tensor:
        """Return scalar alm coefficients with healpy-style iterative refinement.

        HEALPix quadrature is not exact at finite nside.  healpy.map2alm
        compensates by default with three residual-correction iterations; doing
        the same here prevents low-m leakage from dominating the spin-lowered
        E/B modes.
        """
        alm = self.sht(maps)
        for _ in range(self.sht_iter):
            residual = maps - self.isht(alm).real
            alm = alm + self.sht(residual.contiguous())
        return alm

    def _sin_theta_dtheta_alm(self, alm: torch.Tensor) -> torch.Tensor:
        """
        Given scalar alm coefficients a_{ell m}, return coefficients b_{ell m}
        for the scalar field

            sin(theta) * d_theta f.

        This uses only vectorized ell-neighbor recurrences.
        """
        alpha = self._as_real_buffer("alpha_lm", alm)      # (L+1, M)
        ell = self._as_real_buffer("ell", alm).view(-1, 1)

        out = torch.zeros_like(alm)

        # Target ell j receives contribution from source ell j-1:
        #
        #   (j - 1) alpha_{j,m} a_{j-1,m}
        out[..., 1:, :] = out[..., 1:, :] + (
            (ell[1:] - 1.0) * alpha[1:self.lmax, :]
        ) * alm[..., :-1, :]

        # Target ell j receives contribution from source ell j+1:
        #
        #   - (j + 2) alpha_{j+1,m} a_{j+1,m}
        out[..., :-1, :] = out[..., :-1, :] - (
            (ell[:-1] + 2.0) * alpha[1:self.lmax, :]
        ) * alm[..., 1:, :]

        valid = self._as_real_buffer("valid_lm", alm)
        return out * valid

    def _dr_di_from_scalar_alm(self, alm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        From scalar alm of a real scalar field f, compute scalar maps

            D_R f = lap f + 2 cot(theta) d_theta f
                  - 2 csc(theta)^2 d_phi^2 f - 2 f

            D_I f = 2 csc(theta) [d_theta d_phi f + cot(theta) d_phi f]

        using scalar harmonic operations only.

        alm shape:
            (..., lmax, mmax)

        Returns:
            dr, di with shape (..., npix)
        """
        real_dtype = self._real_dtype_of(alm)

        ell = self._as_real_buffer("ell_lm", alm)          # (L, 1)
        emm = self._as_real_buffer("m_lm", alm)            # (1, M)
        valid = self._as_real_buffer("valid_lm", alm)      # (L, M)

        alm = alm * valid

        imag_unit = torch.tensor(1j, dtype=alm.dtype, device=alm.device)

        lap_alm = -(ell * (ell + 1.0)) * alm
        dphi_alm = imag_unit * emm * alm
        dphi2_alm = -(emm.square()) * alm

        sin_dtheta_alm = self._sin_theta_dtheta_alm(alm)
        sin_dtheta_dphi_alm = self._sin_theta_dtheta_alm(dphi_alm)

        # Stack six scalar inverse-SHTs into one cuHPX iSHT call.
        # Input to iSHT: (..., op, ell, m)
        ops_alm = torch.stack(
            (
                alm,
                lap_alm,
                dphi2_alm,
                dphi_alm,
                sin_dtheta_alm,
                sin_dtheta_dphi_alm,
            ),
            dim=-3,
        ).contiguous()

        maps = self.isht(ops_alm).real
        f_bl, lap_f, dphi2_f, dphi_f, sin_dtheta_f, sin_dtheta_dphi_f = maps.unbind(dim=-2)

        csc = self.csc_theta.to(device=alm.device, dtype=real_dtype)
        cot = self.cot_theta.to(device=alm.device, dtype=real_dtype)
        csc2 = self.csc2_theta.to(device=alm.device, dtype=real_dtype)

        dtheta_f = csc * sin_dtheta_f
        dtheta_dphi_f = csc * sin_dtheta_dphi_f

        dr = lap_f + 2.0 * cot * dtheta_f - 2.0 * csc2 * dphi2_f - 2.0 * f_bl
        di = 2.0 * csc * (dtheta_dphi_f + cot * dphi_f)

        return dr, di

    def _lowered_alm_to_cl(self, lowered_alm: torch.Tensor) -> torch.Tensor:
        """
        Convert spin-lowered scalar coefficients x_lm to C_ell.

        lowered_alm shape:
            (..., lmax, mmax)

        Returns:
            cl shape (..., lmax)
        """
        real_dtype = self._real_dtype_of(lowered_alm)

        weight = self.cl_m_weight.to(device=lowered_alm.device, dtype=real_dtype)
        den = self.cl_den.to(device=lowered_alm.device, dtype=real_dtype)
        spin2_norm2 = self.spin2_norm2.to(device=lowered_alm.device, dtype=real_dtype)
        ell_ge_2 = self.ell_ge_2.to(device=lowered_alm.device)

        abs2 = lowered_alm.real.square() + lowered_alm.imag.square()

        # Sum over stored m >= 0, doubling m > 0.
        cl_lowered = (abs2 * weight).sum(dim=-1) / den

        # x_lm = N_ell E_lm or N_ell B_lm.
        cl = cl_lowered / spin2_norm2

        return torch.where(ell_ge_2, cl, torch.zeros_like(cl))


    def _register_alm_index_buffers(self) -> None:
        valid_indices = torch.nonzero(self.valid_lm, as_tuple=False)
        self.register_buffer("alm_ell_indices", valid_indices[:, 0], persistent=False)
        self.register_buffer("alm_m_indices", valid_indices[:, 1], persistent=False)
        self.num_alm = int(valid_indices.size(0))

    def _validate_shear_inputs(self, g1: torch.Tensor, g2: torch.Tensor) -> None:
        if g1.shape != g2.shape:
            raise ValueError(f"g1 and g2 must have the same shape, got {g1.shape} and {g2.shape}.")
        if g1.ndim != 3:
            raise ValueError("g1 and g2 must have shape (batch_size, num_pix, num_channels).")
        if g1.shape[1] != self.npix:
            raise ValueError(
                f"Expected num_pix={self.npix} for nside={self.nside}, "
                f"got num_pix={g1.shape[1]}."
            )
        if self.batch_size is not None and g1.shape[0] != self.batch_size:
            raise ValueError(f"Expected batch_size={self.batch_size}, got {g1.shape[0]}.")
        if self.num_channels is not None and g1.shape[2] != self.num_channels:
            raise ValueError(f"Expected num_channels={self.num_channels}, got {g1.shape[2]}.")
        if not g1.is_cuda or not g2.is_cuda:
            raise RuntimeError("cuHPX SHTCUDA requires CUDA tensors.")
        if g1.dtype not in (torch.float32, torch.float64):
            raise TypeError("g1 and g2 must be float32 or float64 tensors.")
        if g2.dtype != g1.dtype:
            raise TypeError("g1 and g2 must have the same dtype.")

    def _shear_to_lowered_eb_alm(self, g1: torch.Tensor, g2: torch.Tensor) -> torch.Tensor:
        self._validate_shear_inputs(g1, g2)

        g = torch.stack((g1, g2), dim=-1)          # (B, Npix, C, 2)
        g = g.permute(0, 2, 3, 1).contiguous()     # (B, C, 2, Npix)

        if self.input_order == "nest":
            g = self._nest_to_ring_last_axis(g)

        alm_g = self._sht_analysis(g)              # (B, C, 2, L, M)
        dr, di = self._dr_di_from_scalar_alm(alm_g)

        x_e_map = dr[:, :, 0, :] + di[:, :, 1, :]
        x_b_map = dr[:, :, 1, :] - di[:, :, 0, :]
        x_maps = torch.stack((x_e_map, x_b_map), dim=2).contiguous()

        return self._sht_analysis(x_maps)          # (B, C, 2, L, M)


class AngularPowerSpectrum(SphCuHpx):
    """
    Full-sky E/B power spectrum of a spin-2 shear field g = g1 + i g2,
    implemented using only scalar spherical-harmonic transforms.

    Public input layout
    -------------------
    g1, g2 : torch.Tensor
        Shape (batch_size, num_pix, num_channels), in HEALPix NESTED order
        by default.

    Output
    ------
    cl_ee, cl_bb : torch.Tensor
        Each has shape (batch_size, num_ells, num_channels).
    """

    def __init__(
        self,
        nside: int,
        lmax: int | None = None,
        mmax: int | None = None,
        *,
        input_order: Literal["nest", "ring"] = "nest",
        quad_weights: str = "ring",
        norm: str = "ortho",
        csphase: bool = True,
        sht_iter: int = 3,
        return_stacked: bool = False,
    ):
        super().__init__(
            nside=nside,
            lmax=lmax,
            mmax=mmax,
            input_order=input_order,
            quad_weights=quad_weights,
            norm=norm,
            csphase=csphase,
            sht_iter=sht_iter,
        )
        self.return_stacked = bool(return_stacked)

    def forward(
        self,
        g1: torch.Tensor,
        g2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        """
        Parameters
        ----------
        g1, g2:
            Shape (batch_size, num_pix, num_channels).

        Returns
        -------
        If return_stacked=False:
            cl_ee, cl_bb, each shape (batch_size, lmax, num_channels).

        If return_stacked=True:
            cl_eb_stack with shape (batch_size, 2, lmax, num_channels),
            where index 0 is EE and index 1 is BB.
        """
        lowered_alm = self._shear_to_lowered_eb_alm(g1, g2)

        # Convert x_lm to C_ell^EE and C_ell^BB.
        cl = self._lowered_alm_to_cl(lowered_alm)   # (B, C, 2, L)

        cl_ee = cl[:, :, 0, :].permute(0, 2, 1).contiguous()  # (B, L, C)
        cl_bb = cl[:, :, 1, :].permute(0, 2, 1).contiguous()  # (B, L, C)

        if self.return_stacked:
            return torch.stack((cl_ee, cl_bb), dim=1)          # (B, 2, L, C)

        return cl_ee, cl_bb


class ShearToEBMode(SphCuHpx):
    """Convert a spin-2 shear field into flattened E- and B-mode alm coefficients."""

    def __init__(
        self,
        batch_size: int,
        nside: int,
        num_channels: int,
        lmax: int,
        mmax: int,
        *,
        input_order: Literal["nest", "ring"] = "nest",
        quad_weights: str = "ring",
        norm: str = "ortho",
        csphase: bool = True,
        sht_iter: int = 3,
    ):
        super().__init__(
            nside=nside,
            lmax=lmax,
            mmax=mmax,
            batch_size=batch_size,
            num_channels=num_channels,
            input_order=input_order,
            quad_weights=quad_weights,
            norm=norm,
            csphase=csphase,
            sht_iter=sht_iter,
        )

    def forward(self, g1: torch.Tensor, g2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return flattened E- and B-mode alm tensors for ``g1`` and ``g2``."""
        lowered_alm = self._shear_to_lowered_eb_alm(g1, g2)

        spin2_norm = torch.sqrt(
            self.spin2_norm2.to(device=lowered_alm.device, dtype=self._real_dtype_of(lowered_alm))
        )
        spin2_norm = spin2_norm.view(1, 1, 1, self.lmax, 1)
        eb_alm = lowered_alm / spin2_norm

        ell_ge_2 = self.ell_ge_2.to(device=eb_alm.device).view(1, 1, 1, self.lmax, 1)
        eb_alm = torch.where(ell_ge_2, eb_alm, torch.zeros_like(eb_alm))

        # Convert from cuHPX's internal complex-alm phase to the
        # HEALPix/healpy phase expected by callers. Without this, modes with
        # even m have the opposite sign while odd m appear to agree.
        m_phase = self.healpy_m_phase.to(device=eb_alm.device, dtype=self._real_dtype_of(eb_alm))
        eb_alm = eb_alm * m_phase.view(1, 1, 1, 1, self.mmax)

        ell_idx = self.alm_ell_indices.to(device=eb_alm.device)
        m_idx = self.alm_m_indices.to(device=eb_alm.device)
        flattened = eb_alm[..., ell_idx, m_idx]     # (B, C, 2, num_alm)

        e_alm = flattened[:, :, 0, :].permute(0, 2, 1).contiguous()
        b_alm = flattened[:, :, 1, :].permute(0, 2, 1).contiguous()
        return e_alm, b_alm
