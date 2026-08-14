from __future__ import annotations

import pytest


def test_cuhpx_scalar_route_matches_healpy_spin_transform() -> None:
    """Recover synfast E/B maps through both cuHPX and healpy routes."""
    hp = pytest.importorskip("healpy")
    np = pytest.importorskip("numpy")
    torch = pytest.importorskip("torch")
    pytest.importorskip("cuhpx")

    if not torch.cuda.is_available():
        pytest.skip("CuHPXScalarRouteEB requires a CUDA-capable GPU.")

    from euclid_multiprobe_deeplss_training.utils.eb_cuhpx import (
        CuHPXScalarRouteEB,
    )

    nside = 32
    lmax = nside
    ell = np.arange(lmax + 1, dtype=np.float64)
    e_power = np.zeros(lmax + 1, dtype=np.float64)
    b_power = np.zeros_like(e_power)
    e_power[2:] = 5.0e-5 / (ell[2:] * (ell[2:] + 1.0))
    b_power[2:] = 1.0e-5 / (ell[2:] * (ell[2:] + 1.0))

    # Start with independent scalar E and B maps made by synfast, then use
    # healpy to turn their scalar alms into the spin-2 shear field (g1, g2).
    np.random.seed(1729)
    e_input = hp.synfast(e_power, nside=nside, lmax=lmax, pol=False)
    b_input = hp.synfast(b_power, nside=nside, lmax=lmax, pol=False)
    e_input_alm = hp.map2alm(e_input, lmax=lmax, pol=False)
    b_input_alm = hp.map2alm(b_input, lmax=lmax, pol=False)
    g1_ring, g2_ring = hp.alm2map_spin(
        [e_input_alm, b_input_alm],
        nside=nside,
        spin=2,
        lmax=lmax,
    )

    # Reference route: healpy's spin analysis followed by scalar synthesis.
    e_reference_alm, b_reference_alm = hp.map2alm_spin(
        [g1_ring, g2_ring],
        spin=2,
        lmax=lmax,
    )
    eb_reference_ring = np.stack(
        (
            hp.alm2map(e_reference_alm, nside=nside, lmax=lmax),
            hp.alm2map(b_reference_alm, nside=nside, lmax=lmax),
        )
    )
    eb_reference_nested = hp.reorder(eb_reference_ring, r2n=True)

    # CUDA route: the module accepts a batched, channel-first NESTED map.
    shear_nested = np.stack(
        (hp.reorder(g1_ring, r2n=True), hp.reorder(g2_ring, r2n=True))
    )[None]
    shear = torch.as_tensor(shear_nested, device="cuda", dtype=torch.float64)
    transform = CuHPXScalarRouteEB(
        nside=nside,
        lmax=lmax,
        device=shear.device,
        dtype=shear.dtype,
    )

    with torch.no_grad():
        eb_cuhpx = transform(shear)[0].cpu().numpy()

    assert eb_cuhpx.shape == eb_reference_nested.shape
    np.testing.assert_allclose(
        eb_cuhpx,
        eb_reference_nested,
        rtol=2.0e-2,
        atol=2.0e-4,
    )
