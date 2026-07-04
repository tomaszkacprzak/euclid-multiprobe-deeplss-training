from __future__ import annotations

import pytest


def test_angular_power_spectrum_matches_healpy_spin2_synfast() -> None:
    np = pytest.importorskip("numpy")
    hp = pytest.importorskip("healpy")
    torch = pytest.importorskip("torch")
    pytest.importorskip("cuhpx")

    if not torch.cuda.is_available():
        pytest.skip("AngularPowerSpectrum requires CUDA tensors via cuHPX SHTCUDA.")

    from euclid_multiprobe_deeplss_training.networks.spherical_harmonics import AngularPowerSpectrum

    nside = 128
    lmax = 3 * nside
    ell_max = lmax - 1

    ell = np.arange(lmax, dtype=np.float64)
    input_ee = np.zeros(lmax, dtype=np.float64)
    input_bb = np.zeros(lmax, dtype=np.float64)
    ell_ge_2 = ell >= 2
    input_ee[ell_ge_2] = 2.0e-5 / (ell[ell_ge_2] * (ell[ell_ge_2] + 1.0))
    input_bb[ell_ge_2] = 1.0e-5 / (ell[ell_ge_2] * (ell[ell_ge_2] + 1.0))

    # healpy.synfast with pol=True consumes [TT, EE, BB, TE] and returns
    # [T, Q, U].  Q and U are the spin-2 shear components used as g1 and g2.
    t_map, q_map, u_map = hp.synfast(
        [np.zeros(lmax, dtype=np.float64), input_ee, input_bb, np.zeros(lmax, dtype=np.float64)],
        nside=nside,
        lmax=ell_max,
        pol=True,
        new=True,
        verbose=False,
    )

    expected_cls = hp.anafast([t_map, q_map, u_map], lmax=ell_max, pol=True)
    expected_ee = torch.as_tensor(expected_cls[1], device="cuda", dtype=torch.float64)
    expected_bb = torch.as_tensor(expected_cls[2], device="cuda", dtype=torch.float64)

    g1 = torch.as_tensor(q_map, device="cuda", dtype=torch.float64).reshape(1, -1, 1)
    g2 = torch.as_tensor(u_map, device="cuda", dtype=torch.float64).reshape(1, -1, 1)

    angular_power_spectrum = AngularPowerSpectrum(
        nside=nside,
        lmax=lmax,
        mmax=lmax,
        input_order="ring",
    ).to("cuda")

    actual_ee, actual_bb = angular_power_spectrum(g1, g2)

    lmin = 3
    assert actual_ee.shape == (1, lmax, 1)
    assert actual_bb.shape == (1, lmax, 1)
    torch.testing.assert_close(actual_ee[0, lmin:, 0], expected_ee[lmin:], rtol=1.0e-2, atol=1.0e-6)
    torch.testing.assert_close(actual_bb[0, lmin:, 0], expected_bb[lmin:], rtol=1.0e-2, atol=1.0e-6)

def test_shear_to_eb_mode_matches_healpy_toy_alms() -> None:
    np = pytest.importorskip("numpy")
    hp = pytest.importorskip("healpy")
    torch = pytest.importorskip("torch")
    pytest.importorskip("cuhpx")

    if not torch.cuda.is_available():
        pytest.skip("ShearToEBMode requires CUDA tensors via cuHPX SHTCUDA.")

    from euclid_multiprobe_deeplss_training.networks.spherical_harmonics import ShearToEBMode

    batch_size = 1
    nside = 32
    num_channels = 1
    lmax = 16
    mmax = lmax
    ell_max = lmax - 1

    rng = np.random.default_rng(1234)
    alm_size = hp.Alm.getsize(ell_max, mmax=ell_max)
    input_t_alm = np.zeros(alm_size, dtype=np.complex128)
    expected_e_alm_healpy = np.zeros(alm_size, dtype=np.complex128)
    expected_b_alm_healpy = np.zeros(alm_size, dtype=np.complex128)

    for ell in range(2, 7):
        for emm in range(ell + 1):
            idx = hp.Alm.getidx(ell_max, ell, emm)
            expected_e_alm_healpy[idx] = rng.normal(scale=1.0e-4) + 1j * rng.normal(scale=1.0e-4)
            expected_b_alm_healpy[idx] = rng.normal(scale=5.0e-5) + 1j * rng.normal(scale=5.0e-5)
            if emm == 0:
                expected_e_alm_healpy[idx] = expected_e_alm_healpy[idx].real + 0j
                expected_b_alm_healpy[idx] = expected_b_alm_healpy[idx].real + 0j

    _, q_map_ring, u_map_ring = hp.alm2map(
        [input_t_alm, expected_e_alm_healpy, expected_b_alm_healpy],
        nside=nside,
        lmax=ell_max,
        pol=True,
        verbose=False,
    )

    q_map_nest = hp.reorder(q_map_ring, r2n=True)
    u_map_nest = hp.reorder(u_map_ring, r2n=True)
    g1 = torch.as_tensor(q_map_nest, device="cuda", dtype=torch.float64).reshape(batch_size, -1, num_channels)
    g2 = torch.as_tensor(u_map_nest, device="cuda", dtype=torch.float64).reshape(batch_size, -1, num_channels)

    shear_to_eb = ShearToEBMode(
        batch_size=batch_size,
        nside=nside,
        num_channels=num_channels,
        lmax=lmax,
        mmax=mmax,
    ).to("cuda")

    actual_e_alm, actual_b_alm = shear_to_eb(g1, g2)

    ell_indices, m_indices = np.nonzero(np.arange(mmax)[None, :] <= np.arange(lmax)[:, None])
    expected_e_flat = np.asarray(
        [expected_e_alm_healpy[hp.Alm.getidx(ell_max, int(ell), int(emm))] for ell, emm in zip(ell_indices, m_indices, strict=True)]
    )
    expected_b_flat = np.asarray(
        [expected_b_alm_healpy[hp.Alm.getidx(ell_max, int(ell), int(emm))] for ell, emm in zip(ell_indices, m_indices, strict=True)]
    )

    expected_e = torch.as_tensor(expected_e_flat, device="cuda", dtype=torch.complex128).reshape(batch_size, -1, num_channels)
    expected_b = torch.as_tensor(expected_b_flat, device="cuda", dtype=torch.complex128).reshape(batch_size, -1, num_channels)

    assert actual_e_alm.shape == expected_e.shape
    assert actual_b_alm.shape == expected_b.shape
    torch.testing.assert_close(actual_e_alm, expected_e, rtol=5.0e-2, atol=1.0e-7)
    torch.testing.assert_close(actual_b_alm, expected_b, rtol=5.0e-2, atol=1.0e-7)
