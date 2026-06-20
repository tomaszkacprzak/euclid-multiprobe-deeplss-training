from __future__ import annotations

import pytest


def test_healpy_downsampling_sum_preserves_channels_with_expected_resolution() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("healpy")

    from euclid_multiprobe_deeplss_training.networks.smoothing import HealpyDownsampling

    x = torch.stack(
        [
            torch.arange(48, dtype=torch.float32),
            torch.arange(100, 148, dtype=torch.float32),
        ],
        dim=-1,
    ).unsqueeze(0)

    downsample = HealpyDownsampling(nside=2, nside_base=1, nside_lower=[1, 2], dim=-1, operator="sum")

    y = downsample(x)

    expected_channel_0 = torch.arange(48, dtype=torch.float32).reshape(12, 4).sum(dim=1).repeat_interleave(4)
    expected_channel_1 = torch.arange(100, 148, dtype=torch.float32)
    expected = torch.stack([expected_channel_0, expected_channel_1], dim=-1).unsqueeze(0)

    assert y.shape == x.shape
    assert torch.equal(y, expected)


def test_healpy_downsampling_mean_averages_to_lower_nside() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("healpy")

    from euclid_multiprobe_deeplss_training.networks.smoothing import HealpyDownsampling

    x = torch.arange(48, dtype=torch.float32).reshape(1, 48, 1)
    downsample = HealpyDownsampling(nside=2, nside_base=1, nside_lower=[1], dim=-1, operator="mean")

    y = downsample(x)

    expected = torch.arange(48, dtype=torch.float32).reshape(12, 4).mean(dim=1).repeat_interleave(4).reshape(1, 48, 1)
    assert torch.equal(y, expected)


def test_healpy_downsampling_mean_matches_healpy_ud_grade() -> None:
    np = pytest.importorskip("numpy")
    torch = pytest.importorskip("torch")
    hp = pytest.importorskip("healpy")

    from euclid_multiprobe_deeplss_training.networks.smoothing import HealpyDownsampling

    nside = 64
    nside_base = 4
    nside_lower = 16
    input_map = np.arange(hp.nside2npix(nside), dtype=np.float64)
    x = torch.from_numpy(input_map).reshape(1, -1, 1)

    downsample = HealpyDownsampling(nside=nside, nside_base=nside_base, nside_lower=[nside_lower], dim=-1, operator="mean")

    y = downsample(x)

    ud_grade_lower = hp.ud_grade(input_map, nside_out=nside_lower, order_in="NESTED", order_out="NESTED", power=0)
    expected = hp.ud_grade(ud_grade_lower, nside_out=nside, order_in="NESTED", order_out="NESTED", power=0)
    expected = torch.from_numpy(expected).reshape(1, -1, 1)

    assert torch.allclose(y, expected)


def test_healpy_downsampling_rejects_unknown_operator() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("healpy")

    from euclid_multiprobe_deeplss_training.networks.smoothing import HealpyDownsampling

    with pytest.raises(ValueError, match="Invalid operator: median"):
        HealpyDownsampling(nside=2, nside_base=1, nside_lower=[1], dim=-1, operator="median")
