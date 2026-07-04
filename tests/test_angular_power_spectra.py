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
    lmax = lmax - 1

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
        lmax=lmax,
        pol=True,
        new=True,
        verbose=False,
    )

    expected_cls = hp.anafast([t_map, q_map, u_map], lmax=lmax, pol=True)
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

    batch_size = 4
    nside = 128
    num_channels = 2
    lmax = 2 * nside + 1
    mmax = lmax
    lmin = 0

    alm_size = hp.Alm.getsize(lmax, mmax=lmax)
    input_t_alm = np.zeros(alm_size, dtype=np.complex128)

    ell = np.arange(lmax, dtype=np.float64)
    input_ee = np.zeros(lmax, dtype=np.float64)
    input_bb = np.zeros(lmax, dtype=np.float64)
    ell_ge_2 = ell >= 2
    input_ee[ell_ge_2] = 5.0e-5 / (ell[ell_ge_2] * (ell[ell_ge_2] + 1.0))
    input_bb[ell_ge_2] = 1.0e-5 / (ell[ell_ge_2] * (ell[ell_ge_2] + 1.0))

    np.random.seed(12343)
    q_maps_nest = np.empty((batch_size, hp.nside2npix(nside), num_channels), dtype=np.float64)
    u_maps_nest = np.empty_like(q_maps_nest)
    expected_e_alm_healpy = np.empty((batch_size, num_channels, alm_size), dtype=np.complex128)
    expected_b_alm_healpy = np.empty_like(expected_e_alm_healpy)

    for batch_idx in range(batch_size):
        for channel_idx in range(num_channels):
            e_map_ring = hp.synfast(input_ee, nside=nside, lmax=lmax, pol=False, new=True, verbose=False)
            b_map_ring = hp.synfast(input_bb, nside=nside, lmax=lmax, pol=False, new=True, verbose=False)

            e_alm = hp.map2alm(e_map_ring, lmax=lmax, pol=False, use_weights=False, use_pixel_weights=False)
            b_alm = hp.map2alm(b_map_ring, lmax=lmax, pol=False, use_weights=False, use_pixel_weights=False)
            expected_e_alm_healpy[batch_idx, channel_idx] = e_alm
            expected_b_alm_healpy[batch_idx, channel_idx] = b_alm

            _, q_map_ring, u_map_ring = hp.alm2map(
                [input_t_alm, e_alm, b_alm],
                nside=nside,
                lmax=lmax,
                pol=True,
                verbose=False,
            )
            q_maps_nest[batch_idx, :, channel_idx] = hp.reorder(q_map_ring, r2n=True)
            u_maps_nest[batch_idx, :, channel_idx] = hp.reorder(u_map_ring, r2n=True)

    g1 = torch.as_tensor(q_maps_nest, device="cuda", dtype=torch.float64)
    g2 = torch.as_tensor(u_maps_nest, device="cuda", dtype=torch.float64)

    sht_iter = 1
    shear_to_eb = ShearToEBMode(
        batch_size=batch_size,
        nside=nside,
        num_channels=num_channels,
        lmax=lmax,
        mmax=mmax,
        sht_iter=sht_iter,
        quad_weights="ring",
    ).to("cuda")

    print()
    print("sht_iter = ", sht_iter)
    print("g1.shape = ", g1.shape)
    print("g2.shape = ", g2.shape)

    actual_e_alm, actual_b_alm = shear_to_eb(g1, g2)

    # ShearToEBMode flattens valid coefficients in row-major (ell, m) order
    # from an (lmax, mmax) grid, unlike healpy's compact m-major alm array.
    ell_indices, m_indices = np.nonzero(np.arange(mmax)[None, :] <= np.arange(lmax)[:, None])
    mode_mask = ell_indices >= lmin
    healpy_indices = [hp.Alm.getidx(lmax, int(ell), int(emm)) for ell, emm in zip(ell_indices, m_indices, strict=True)]
    expected_e_flat = expected_e_alm_healpy[:, :, healpy_indices]
    expected_b_flat = expected_b_alm_healpy[:, :, healpy_indices]

    expected_e = torch.as_tensor(
        expected_e_flat[:, :, mode_mask].transpose(0, 2, 1), device="cuda", dtype=torch.complex128
    )
    expected_b = torch.as_tensor(
        expected_b_flat[:, :, mode_mask].transpose(0, 2, 1), device="cuda", dtype=torch.complex128
    )
    selected_mode_indices = torch.as_tensor(np.nonzero(mode_mask)[0], device="cuda")
    actual_e_alm = actual_e_alm.index_select(1, selected_mode_indices)
    actual_b_alm = actual_b_alm.index_select(1, selected_mode_indices)
    actual_e_alm = actual_e_alm.cpu().numpy()
    actual_b_alm = actual_b_alm.cpu().numpy()
    expected_e = expected_e.cpu().numpy()
    expected_b = expected_b.cpu().numpy()

    print("actual_e_alm.shape = ", actual_e_alm.shape)
    print("expected_e.shape = ", expected_e.shape)
    print("actual_b_alm.shape = ", actual_b_alm.shape)
    print("expected_b.shape = ", expected_b.shape)

    example_id = 1
    channel_id = 1
    print("E comparison:")
    for i in range(20):
        print(f"l={ell_indices[i]:>5d} m={m_indices[i]: >5d} actual_e_alm[{i}] = {actual_e_alm[example_id,i,channel_id]: .10f} vs expected_e[{i}] = {expected_e[example_id,i,channel_id]: .10f} delta = {np.abs(actual_e_alm[example_id,i,channel_id] - expected_e[example_id,i,channel_id]): .10f}")

    print("B comparison:")
    for i in range(20):
        print(f"l={ell_indices[i]:>5d} m={m_indices[i]: >5d} actual_b_alm[{i}] = {actual_b_alm[example_id,i,channel_id]: .10f} vs expected_b[{i}] = {expected_b[example_id,i,channel_id]: .10f} delta = {np.abs(actual_b_alm[example_id,i,channel_id] - expected_b[example_id,i,channel_id]): .10f}")


    assert actual_e_alm.shape == expected_e.shape
    assert actual_b_alm.shape == expected_b.shape
    torch.testing.assert_close(actual_e_alm, expected_e, rtol=0.0, atol=1.0e-3)
    torch.testing.assert_close(actual_b_alm, expected_b, rtol=0.0, atol=1.0e-3)
