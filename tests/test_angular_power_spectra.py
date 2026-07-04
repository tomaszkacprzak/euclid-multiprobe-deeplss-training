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
