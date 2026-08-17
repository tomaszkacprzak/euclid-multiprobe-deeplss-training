"""
Differentiable scalar-route E/B decomposition for NESTED HEALPix maps.

The module uses cuHPX scalar SHTCUDA/iSHTCUDA transforms only. For every
(g1, g2) pair it constructs the six weighted real maps in one tensor, performs
one batched scalar analysis, combines the resulting two-dimensional alm arrays,
and performs one batched scalar synthesis.

Conventions
-----------
* Input and output maps use HEALPix NESTED ordering.
* The real and imaginary input components are g1 and g2, using the
  HEALPix/CMB Q/U convention.
* ``lmax`` is the usual inclusive maximum multipole. cuHPX is therefore
  initialized with coefficient extents L = M = lmax + 1.
"""

from __future__ import annotations

from typing import Literal

import cuhpx
import torch
from cuhpx import SHTCUDA, iSHTCUDA
from torch import nn


class CuHPXScalarRouteEB(nn.Module):
    """
    Convert batched full-sky shear maps (g1, g2) to scalar (E, B) maps or alms.

    Parameters
    ----------
    nside:
        HEALPix NSIDE. NESTED ordering requires a power-of-two NSIDE.
    lmax:
        Inclusive maximum multipole, using the conventional HEALPix meaning.
        Internally cuHPX receives ``lmax + 1`` because its alm arrays have
        shape ``(..., L, M)`` with ell=0,...,L-1 and m=0,...,M-1.
    quad_weights:
        Scalar-analysis quadrature used by cuHPX. Use ``"ring"`` or ``"none"``.
    output_type:
        Return complex NESTED E/B maps when ``"map"``, or two-dimensional E/B
        alm arrays with shape ``(batch_size, 2, lmax + 1, lmax + 1)`` when
        ``"alm"``.
    device:
        CUDA device on which the module is constructed. Construct the module
        directly on its target GPU, especially in multi-GPU applications.
    dtype:
        Real dtype of the precomputed scalar-route weights. The forward input
        must have the same dtype. Float64 is recommended when polar
        cancellation accuracy is important.

    Input
    -----
    x:
        Complex CUDA tensor with shape ``(batch_size, 12*nside**2)`` in NESTED
        ordering. Its real component is g1 and its imaginary component is g2.

    Output
    ------
    When ``output_type="map"``, a complex tensor with the same shape and
    ordering as the input, whose real and imaginary components are E and B. When
    ``output_type="alm"``, a complex tensor with shape
    ``(batch_size, 2, lmax + 1, lmax + 1)`` whose channels 0 and 1 are the E
    and B modes.

    Notes
    -----
    There are no Python loops in initialization or forward. The forward pass
    calls the scalar cuHPX SHT exactly once on a tensor of shape
    ``(batch_size, 6, num_pixels)`` and calls the scalar inverse SHT exactly
    once on a tensor of shape ``(batch_size, 2, L, M)``.
    """

    def __init__(
        self,
        nside: int,
        lmax: int,
        *,
        output_type: Literal["map", "alm"] = "map",
        quad_weights: str = "ring",
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()

        nside = int(nside)
        lmax = int(lmax)

        if nside <= 0 or (nside & (nside - 1)) != 0:
            raise ValueError(
                "nside must be a positive power of two for NESTED ordering."
            )
        if lmax < 2:
            raise ValueError("lmax must be at least 2 for a spin-2 field.")
        if lmax > 2 * nside:
            raise ValueError(
                "Current cuHPX real transforms store a complete non-negative "
                "m range only through m=2*nside. Require lmax <= 2*nside "
                "when using the square M=L=lmax+1 coefficient layout."
            )
        if quad_weights not in {"ring", "none"}:
            raise ValueError("quad_weights must be either 'ring' or 'none'.")
        if output_type not in {"map", "alm"}:
            raise ValueError("output_type must be either 'map' or 'alm'.")
        if dtype not in {torch.float32, torch.float64}:
            raise TypeError("dtype must be torch.float32 or torch.float64.")
        if not torch.cuda.is_available():
            raise RuntimeError("CuHPXScalarRouteEB requires a CUDA-capable GPU.")

        target_device = (
            torch.device("cuda", torch.cuda.current_device())
            if device is None
            else torch.device(device)
        )
        if target_device.type != "cuda":
            raise ValueError("device must be a CUDA device.")
        if target_device.index is None:
            target_device = torch.device("cuda", torch.cuda.current_device())

        self.nside = nside
        self.lmax = lmax
        self.L = lmax + 1
        self.M = lmax + 1
        self.num_pixels = 12 * nside * nside
        self.output_type = output_type
        self.quad_weights = quad_weights


        # cuHPX constructs some internal tensors on the current CUDA device.
        with torch.cuda.device(target_device):
            self.sht = SHTCUDA(
                nside=nside,
                lmax=self.L,
                mmax=self.M,
                quad_weights=quad_weights,
                norm="ortho",
                csphase=True,
            )
            self.isht = iSHTCUDA(
                nside=nside,
                lmax=self.L,
                mmax=self.M,
                quad_weights=quad_weights,
                norm="ortho",
                csphase=True,
            )

            pixel_weights = self._make_ring_pixel_weights(
                nside=nside,
                device=target_device,
                dtype=dtype,
            )
            harmonic_weights = self._make_harmonic_weights(
                L=self.L,
                M=self.M,
                device=target_device,
                dtype=dtype,
            )
            nest_index_for_ring, ring_index_for_nest = (
                self._make_ordering_indices(
                    nside=nside,
                    num_pixels=self.num_pixels,
                    device=target_device,
                )
            )

        # These buffers are deterministic functions of nside/lmax and need not
        # enlarge checkpoints.
        self.register_buffer("pixel_weights", pixel_weights, persistent=False)
        self.register_buffer(
            "harmonic_weights", harmonic_weights, persistent=False
        )
        self.register_buffer(
            "nest_index_for_ring",
            nest_index_for_ring,
            persistent=False,
        )
        self.register_buffer(
            "ring_index_for_nest",
            ring_index_for_nest,
            persistent=False,
        )

    @staticmethod
    def _make_ring_pixel_weights(
        *,
        nside: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Return [w0, w1, w2] with shape (3, num_pixels) in RING ordering.

        w0 = 1 / sin(theta)^2
        w1 = cos(theta) / sin(theta)^2
        w2 = cos(theta)^2 / sin(theta)^2
        """
        ring = torch.arange(
            1,
            4 * nside,
            device=device,
            dtype=torch.int64,
        )
        ring_f64 = ring.to(torch.float64)
        ns = float(nside)

        z_north = 1.0 - ring_f64.square() / (3.0 * ns * ns)
        z_equatorial = 4.0 / 3.0 - 2.0 * ring_f64 / (3.0 * ns)
        z_south = (
            (4.0 * ns - ring_f64).square() / (3.0 * ns * ns) - 1.0
        )
        z_ring = torch.where(
            ring < nside,
            z_north,
            torch.where(ring <= 3 * nside, z_equatorial, z_south),
        )

        nphi = torch.where(
            ring < nside,
            4 * ring,
            torch.where(
                ring <= 3 * nside,
                torch.full_like(ring, 4 * nside),
                4 * (4 * nside - ring),
            ),
        )
        z = torch.repeat_interleave(z_ring, nphi)

        expected_pixels = 12 * nside * nside
        if z.numel() != expected_pixels:
            raise RuntimeError(
                f"Internal HEALPix ring construction produced {z.numel()} "
                f"pixels; expected {expected_pixels}."
            )

        inv_sin2 = (1.0 - z.square()).reciprocal()
        weights = torch.stack(
            (
                inv_sin2,
                z * inv_sin2,
                z.square() * inv_sin2,
            ),
            dim=0,
        )
        return weights.to(dtype=dtype)

    @staticmethod
    def _make_harmonic_weights(
        *,
        L: int,
        M: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Precompute scalar-route factors in cuHPX's two-dimensional alm layout.

        The returned tensor has shape (5, L, M), with planes

            0: R_l K_lm
            1: R_l K_lm m
            2: R_l l(l-1)
            3: R_l 2m(l-1)
            4: R_l [2m^2-l(l+1)]

        All entries with l < 2 or m > l are exactly zero.
        """
        ell = torch.arange(L, device=device, dtype=torch.float64).view(L, 1)
        m = torch.arange(M, device=device, dtype=torch.float64).view(1, M)
        valid = (ell >= 2.0) & (m <= ell)

        ell_safe = ell.clamp_min(2.0)
        r_l = torch.rsqrt(
            (ell_safe - 1.0)
            * ell_safe
            * (ell_safe + 1.0)
            * (ell_safe + 2.0)
        ).expand(L, M)
        r_l = r_l * valid.to(torch.float64)

        k_lm = 2.0 * torch.sqrt(
            (2.0 * ell_safe + 1.0)
            / (2.0 * ell_safe - 1.0)
            * (ell_safe.square() - m.square()).clamp_min(0.0)
        )

        rk = r_l * k_lm
        harmonic_weights = torch.stack(
            (
                rk,
                rk * m,
                r_l * ell * (ell - 1.0),
                r_l * 2.0 * m * (ell - 1.0),
                r_l * (2.0 * m.square() - ell * (ell + 1.0)),
            ),
            dim=0,
        )
        return harmonic_weights.to(dtype=dtype)

    @staticmethod
    def _make_ordering_indices(
        *,
        nside: int,
        num_pixels: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Precompute differentiable index_select permutations.

        cuHPX's remapper is used once here on integer pixel labels. Forward
        then uses torch.index_select, so NESTED/RING conversion remains inside
        the PyTorch autograd graph as a pure permutation.
        """
        nested_pixel_id = torch.arange(
            num_pixels,
            device=device,
            dtype=torch.int64,
        )

        # At ring pixel r, this contains the corresponding NESTED pixel index.
        nest_index_for_ring = cuhpx.nest2ring(nested_pixel_id, nside).long()

        # Invert the permutation without a Python loop.
        ring_index_for_nest = torch.empty_like(nest_index_for_ring)
        ring_index_for_nest.scatter_(
            0,
            nest_index_for_ring,
            nested_pixel_id,
        )
        return nest_index_for_ring, ring_index_for_nest

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(
                "Expected x with shape (batch_size, num_pixels); "
                f"received {tuple(x.shape)}."
            )
        if x.shape[1] != self.num_pixels:
            raise ValueError(
                "Expected x with shape "
                f"(batch_size, {self.num_pixels}); received {tuple(x.shape)}."
            )
        if not x.is_cuda:
            raise ValueError("x must be a CUDA tensor.")
        if x.dtype not in {torch.complex64, torch.complex128}:
            raise TypeError("x must have dtype torch.complex64 or torch.complex128.")
        if x.device != self.pixel_weights.device:
            raise ValueError(
                f"x is on {x.device}, but the module is on "
                f"{self.pixel_weights.device}."
            )
        expected_dtype = (
            torch.complex64
            if self.pixel_weights.dtype == torch.float32
            else torch.complex128
        )
        if x.dtype != expected_dtype:
            raise TypeError(
                f"x has dtype {x.dtype}, but the module weights have dtype "
                f"{self.pixel_weights.dtype}. Construct with the desired dtype "
                "or cast the whole module and input consistently."
            )

        batch_size = x.shape[0]

        # Input maps are NESTED; cuHPX's scalar SHT consumes RING ordering.
        x_ring = torch.index_select(
            x,
            dim=-1,
            index=self.nest_index_for_ring,
        )

        # Splitting into real scalar maps is necessary at the boundary to the
        # real-valued scalar cuHPX transform. Shape: (B, 2, 3, P) -> (B, 6, P).
        # Flattening preserves the order
        # [w0*g1, w1*g1, w2*g1, w0*g2, w1*g2, w2*g2].
        maps6 = (
            torch.view_as_real(x_ring).movedim(-1, 1).unsqueeze(2)
            * self.pixel_weights.view(1, 1, 3, self.num_pixels)
        ).reshape(batch_size, 6, self.num_pixels).contiguous()

        # The only forward scalar SHT call.
        alms6 = self.sht(maps6).reshape(
            batch_size,
            2,
            3,
            self.L,
            self.M,
        )

        q = alms6[:, 0]
        u = alms6[:, 1]

        # Only p=0 and p=1 require the neighboring (l-1,m) coefficient.
        # This creates four shifted fields rather than shifting all six.
        alms01_lminus1 = torch.cat(
            (
                torch.zeros_like(alms6[:, :, :2, :1, :]),
                alms6[:, :, :2, :-1, :],
            ),
            dim=-2,
        )
        q_lminus1 = alms01_lminus1[:, 0]
        u_lminus1 = alms01_lminus1[:, 1]

        q0 = q[:, 0]
        q1 = q[:, 1]
        q2 = q[:, 2]
        u0 = u[:, 0]
        u1 = u[:, 1]
        u2 = u[:, 2]

        q0_lminus1 = q_lminus1[:, 0]
        q1_lminus1 = q_lminus1[:, 1]
        u0_lminus1 = u_lminus1[:, 0]
        u1_lminus1 = u_lminus1[:, 1]

        rk, rkm, rl2, r2m, rh = self.harmonic_weights.unbind(dim=0)

        # Direct six-transform formulas in the HEALPix/CMB E/B convention.
        e_alm = (
            -rk * q1_lminus1
            - 1.0j * rkm * u0_lminus1
            - rl2 * q2
            + 1.0j * r2m * u1
            - rh * q0
        )
        b_alm = (
            -rk * u1_lminus1
            + 1.0j * rkm * q0_lminus1
            - rl2 * u2
            - 1.0j * r2m * q1
            - rh * u0
        )

        eb_alm = torch.stack((e_alm, b_alm), dim=1).contiguous()

        if self.output_type == "alm":
            return eb_alm
 
        elif self.output_type == "map":
            # The only inverse scalar SHT call.
            eb_ring = self.isht(eb_alm)

            # Return E/B in the same NESTED ordering as the input.
            eb_nest = torch.index_select(
                eb_ring,
                dim=-1,
                index=self.ring_index_for_nest,
            )

            return torch.complex(eb_nest[:, 0], eb_nest[:, 1])

        else:
            raise ValueError(f"Invalid output_type: {self.output_type}")

    def extra_repr(self) -> str:
        return (
            f"nside={self.nside}, lmax={self.lmax}, "
            f"num_pixels={self.num_pixels}, "
            f"quad_weights={self.quad_weights!r}, "
            f"dtype={self.pixel_weights.dtype}, "
            f"device={self.pixel_weights.device}"
        )
