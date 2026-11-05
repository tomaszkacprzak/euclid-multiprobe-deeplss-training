from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
hp = pytest.importorskip("healpy")
NestChannelScalecut = pytest.importorskip("euclid_multiprobe_deeplss_training.networks.smoothing").NestChannelScalecut


def test_scalecut_sum_preserves_channels_with_expected_resolution() -> None:
    x = torch.stack(
        [torch.arange(48, dtype=torch.float32), torch.arange(100, 148, dtype=torch.float32)], dim=-1
    ).unsqueeze(0)
    scalecut = NestChannelScalecut(nside=2, nside_base=1, nside_lower=[1, 2], operator="sum")

    y = scalecut(x, torch.ones_like(x))

    expected_channel_0 = torch.arange(48, dtype=torch.float32).reshape(12, 4).sum(dim=1).repeat_interleave(4)
    expected_channel_1 = torch.arange(100, 148, dtype=torch.float32)
    expected = torch.stack([expected_channel_0, expected_channel_1], dim=-1).unsqueeze(0)
    assert y.shape == x.shape
    assert torch.equal(y, expected)


def test_scalecut_mean_averages_to_lower_nside() -> None:
    x = torch.arange(48, dtype=torch.float32).reshape(1, 48, 1)
    scalecut = NestChannelScalecut(nside=2, nside_base=1, nside_lower=[1], operator="mean")

    y = scalecut(x, torch.ones_like(x))

    expected = torch.arange(48, dtype=torch.float32).reshape(12, 4).mean(dim=1).repeat_interleave(4).reshape(1, 48, 1)
    assert torch.equal(y, expected)


def test_scalecut_mean_matches_healpy_ud_grade() -> None:
    nside = 64
    nside_base = 4
    nside_lower = 16
    input_map = torch.arange(hp.nside2npix(nside), dtype=torch.float64).numpy()
    x = torch.from_numpy(input_map).reshape(1, -1, 1)
    scalecut = NestChannelScalecut(nside=nside, nside_base=nside_base, nside_lower=[nside_lower], operator="mean")

    y = scalecut(x, torch.ones_like(x))

    lower_map = hp.ud_grade(input_map, nside_out=nside_lower, order_in="NESTED", order_out="NESTED", power=0)
    expected = hp.ud_grade(lower_map, nside_out=nside, order_in="NESTED", order_out="NESTED", power=0)
    torch.testing.assert_close(y, torch.from_numpy(expected).reshape(1, -1, 1))


@pytest.mark.parametrize("operator", ["sum", "mean"])
def test_scalecut_nonuniform_weights_mixed_resolutions_and_empty_groups(operator: str) -> None:
    # Two complete base patches per batch; the second patch has no statistical weight.
    x = torch.arange(1, 33, dtype=torch.float64).reshape(2, 8, 2)
    w = torch.tensor([1, 0, 2, 1, 0, 0, 0, 0], dtype=torch.float64).reshape(1, 8, 1)
    scalecut = NestChannelScalecut(2, 1, [1, 2], operator=operator)

    y = scalecut(x, w)

    expected = torch.zeros_like(x)
    if operator == "sum":
        expected[0, :4, 0] = 18
        expected[1, :4, 0] = 82
        expected[0, :4, 1] = torch.tensor([2, 0, 12, 8])
        expected[1, :4, 1] = torch.tensor([18, 0, 44, 24])
    else:
        expected[0, :4, 0] = 4.5
        expected[1, :4, 0] = 20.5
        expected[0, :4, 1] = torch.tensor([2, 0, 6, 8])
        expected[1, :4, 1] = torch.tensor([18, 0, 22, 24])
    torch.testing.assert_close(y, expected)


@pytest.mark.parametrize("operator", ["sum", "mean"])
def test_scalecut_masks_nonfinite_values_with_zero_weight_and_has_finite_gradients(operator: str) -> None:
    x = torch.tensor([2, float("nan"), 4, float("inf"), float("nan"), -float("inf"), float("nan"), float("inf")],
                     dtype=torch.float64).reshape(1, 8, 1).requires_grad_()
    w = torch.tensor([1, 0, 2, 0, 0, 0, 0, 0], dtype=torch.float64).reshape(1, 8, 1).requires_grad_()
    scalecut = NestChannelScalecut(2, 1, [1], operator=operator)

    y = scalecut(x, w)
    y.sum().backward()

    expected = torch.zeros_like(x)
    expected[:, :4] = 10 if operator == "sum" else 10 / 3
    torch.testing.assert_close(y, expected)
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(w.grad).all()
    assert torch.equal(x.grad[w == 0], torch.zeros_like(x.grad[w == 0]))


@pytest.mark.parametrize("operator", ["sum", "mean"])
@pytest.mark.parametrize("invalid", [float("nan"), float("inf")])
def test_scalecut_preserves_nonfinite_values_with_positive_weight(operator: str, invalid: float) -> None:
    x = torch.ones(1, 4, 1)
    x[0, 0, 0] = invalid
    y = NestChannelScalecut(2, 1, [1], operator=operator)(x, torch.ones_like(x))
    assert (~torch.isfinite(y)).all()


@pytest.mark.parametrize("operator", ["sum", "mean"])
def test_scalecut_gradients_match_finite_differences(operator: str) -> None:
    x = torch.linspace(0.3, 2.7, 16, dtype=torch.float64).reshape(1, 8, 2).requires_grad_()
    # Shared weights exercise gradient accumulation across channels and repeated output pixels.
    w = torch.linspace(0.5, 1.5, 8, dtype=torch.float64).reshape(1, 8, 1).requires_grad_()
    scalecut = NestChannelScalecut(2, 1, [1, 2], operator=operator)
    assert torch.autograd.gradcheck(scalecut, (x, w))


@pytest.mark.parametrize("operator", ["sum", "mean"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.complex64, torch.complex128])
def test_scalecut_preserves_real_and_complex_dtypes(operator: str, dtype) -> None:
    x = torch.tensor([1, 2, 3, 4], dtype=dtype).reshape(1, 4, 1)
    if x.is_complex():
        x = x * (1 + 2j)
    w = torch.tensor([1, 2, 0, 1], dtype=x.real.dtype).reshape(1, 4, 1)

    y = NestChannelScalecut(2, 1, [1], operator=operator)(x, w)

    expected = x[:, 0] + 2 * x[:, 1] + x[:, 3]
    if operator == "mean":
        expected = expected / 4
    assert y.dtype == dtype
    assert y.device == x.device
    torch.testing.assert_close(y, expected.unsqueeze(1).expand_as(y))


@pytest.mark.parametrize("operator", ["sum", "mean"])
def test_scalecut_applies_weights_when_input_and_base_resolution_match(operator: str) -> None:
    x = torch.tensor([1, 2, 3, 4], dtype=torch.float64).reshape(1, 4, 1)
    w = torch.tensor([0, 2, 3, 0], dtype=torch.float64).reshape(1, 4, 1)

    y = NestChannelScalecut(1, 1, [1], operator=operator)(x, w)

    expected = torch.tensor([0, 4, 9, 0] if operator == "sum" else [0, 2, 3, 0], dtype=x.dtype).reshape_as(x)
    torch.testing.assert_close(y, expected)


@pytest.mark.parametrize("shape", [(), (4, 1), (2, 1, 2), (2, 4, 1), (2, 4, 2)])
def test_scalecut_broadcasts_statistical_weights(shape: tuple[int, ...]) -> None:
    x = torch.ones(2, 4, 2)
    w = torch.full(shape, 2.0)

    y = NestChannelScalecut(2, 1, [1, 2])(x, w)

    expected = torch.ones_like(x)
    expected[..., 0] = 8
    expected[..., 1] = 2
    torch.testing.assert_close(y, expected)


def test_scalecut_requires_explicit_weights() -> None:
    scalecut = NestChannelScalecut(2, 1, [1])
    with pytest.raises(TypeError, match="w"):
        scalecut(torch.ones(1, 4, 1))
    with pytest.raises(TypeError, match="tensor"):
        scalecut(torch.ones(1, 4, 1), None)


@pytest.mark.parametrize("weight", [-1.0, float("nan"), float("inf"), -float("inf")])
def test_scalecut_rejects_invalid_statistical_weights(weight: float) -> None:
    with pytest.raises(ValueError, match="finite nonnegative"):
        NestChannelScalecut(2, 1, [1])(torch.ones(1, 4, 1), torch.tensor(weight))


def test_scalecut_rejects_complex_weights() -> None:
    with pytest.raises(ValueError, match="real statistical weights"):
        NestChannelScalecut(2, 1, [1])(torch.ones(1, 4, 1), torch.tensor(1 + 0j))


def test_scalecut_rejects_weights_that_do_not_broadcast() -> None:
    with pytest.raises(ValueError, match="broadcast"):
        NestChannelScalecut(2, 1, [1])(torch.ones(1, 4, 1), torch.ones(1, 3, 1))


def test_scalecut_rejects_weights_on_another_device() -> None:
    with pytest.raises(ValueError, match="same device"):
        NestChannelScalecut(2, 1, [1])(torch.ones(1, 4, 1), torch.ones(1, device="meta"))


@pytest.mark.parametrize(
    ("shape", "message"),
    [((4, 1), "shape"), ((1, 4, 2), "one channel"), ((1, 6, 1), "complete NESTED"), ((1, 0, 1), "complete NESTED")],
)
def test_scalecut_rejects_invalid_input_shapes(shape: tuple[int, ...], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        NestChannelScalecut(2, 1, [1])(torch.ones(shape), torch.tensor(1.0))


@pytest.mark.parametrize(
    ("nside", "nside_base", "nside_lower"),
    [(0, 1, [1]), (3, 1, [1]), (2, 3, [1]), (2, 4, [1]), (2, 1, [0]), (2, 1, [3]), (2, 1, [4]), (4, 2, [1]), (2, 1, [])],
)
def test_scalecut_rejects_invalid_nside_hierarchies(nside: int, nside_base: int, nside_lower: list[int]) -> None:
    with pytest.raises(ValueError, match="nside"):
        NestChannelScalecut(nside, nside_base, nside_lower)


def test_scalecut_rejects_unknown_operator() -> None:
    with pytest.raises(ValueError, match="Invalid operator: median"):
        NestChannelScalecut(nside=2, nside_base=1, nside_lower=[1], operator="median")


@pytest.mark.parametrize("operator", ["sum", "mean"])
def test_scalecut_upper_cut_matches_hand_calculated_weighted_maps(operator: str) -> None:
    x = torch.arange(1, 17, dtype=torch.float64).reshape(1, 16, 1)
    w = torch.tensor([1, 1, 1, 1, 1, 2, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0], dtype=x.dtype).reshape_as(x)

    y = NestChannelScalecut(4, 1, [2], operator=operator, nside_upper=[1])(x, w)

    # The four target pixels have sums [10, 25, 9, 0] and weights [4, 4, 1, 0].
    # The coarse sum 44 is shared across FOUR target pixels, not 16 input pixels.
    if operator == "sum":
        expected = torch.tensor([-1, 14, -2, -11], dtype=x.dtype)
    else:
        expected = torch.tensor([2.5, 6.25, 9, 0], dtype=x.dtype) - 44 / 9
    torch.testing.assert_close(y.flatten(), expected.repeat_interleave(4))
    # An empty target pixel still has the coarse background subtracted.
    assert y[0, -1, 0] < 0


@pytest.mark.parametrize("operator", ["sum", "mean"])
def test_scalecut_upper_cut_handles_batches_channels_and_original_resolution(operator: str) -> None:
    x = (torch.arange(128, dtype=torch.float64).reshape(2, 32, 2) - 27) / 7
    # Weights broadcast across channels, with different weights in each batch.
    w = (torch.arange(64, dtype=x.dtype).reshape(2, 32, 1) % 7) / 3
    nside_lower, nside_upper = [2, 4], [1, 2]

    y = NestChannelScalecut(4, 1, nside_lower, operator=operator, nside_upper=nside_upper)(x, w)

    # A direct loop over target pixels provides a reference independent of the
    # vectorized implementation, including channel 1 whose target is the input grid.
    expected = torch.empty_like(x)
    for batch in range(2):
        for channel in range(2):
            target_size = (4 // nside_lower[channel]) ** 2
            coarse_size = (4 // nside_upper[channel]) ** 2
            for start in range(0, 32, target_size):
                coarse_start = start // coarse_size * coarse_size
                target = slice(start, start + target_size)
                coarse = slice(coarse_start, coarse_start + coarse_size)
                target_sum = sum(float(x[batch, p, channel] * w[batch, p, 0]) for p in range(*target.indices(32)))
                coarse_sum = sum(float(x[batch, p, channel] * w[batch, p, 0]) for p in range(*coarse.indices(32)))
                if operator == "mean":
                    target_weight = sum(float(w[batch, p, 0]) for p in range(*target.indices(32)))
                    coarse_weight = sum(float(w[batch, p, 0]) for p in range(*coarse.indices(32)))
                    value = (target_sum / target_weight if target_weight else 0) - (
                        coarse_sum / coarse_weight if coarse_weight else 0
                    )
                else:
                    value = target_sum - coarse_sum / (coarse_size // target_size)
                expected[batch, target, channel] = value
    torch.testing.assert_close(y, expected)


@pytest.mark.parametrize("operator", ["sum", "mean"])
def test_scalecut_upper_cut_has_zero_coarse_aggregate(operator: str) -> None:
    x = torch.arange(1, 65, dtype=torch.float64).reshape(2, 32, 1) ** 2
    w = torch.linspace(0.25, 3.5, 64, dtype=x.dtype).reshape_as(x)

    y = NestChannelScalecut(4, 1, [2], operator=operator, nside_upper=[1])(x, w)

    # Count each target pixel once; the returned map repeats it four times.
    target_values = y[:, ::4, 0].reshape(2, 2, 4)
    if operator == "mean":
        target_weights = w[..., 0].reshape(2, 2, 4, 4).sum(-1)
        target_values = target_values * target_weights
    torch.testing.assert_close(target_values.sum(-1), torch.zeros(2, 2, dtype=x.dtype), atol=2e-11, rtol=0)


def test_scalecut_upper_mean_annihilates_coarse_piecewise_constant_maps() -> None:
    x = torch.tensor([2.0] * 16 + [-3.0] * 16, dtype=torch.float64).reshape(1, 32, 1)
    w = torch.arange(1, 33, dtype=x.dtype).reshape_as(x)

    y = NestChannelScalecut(4, 1, [2], operator="mean", nside_upper=[1])(x, w)

    torch.testing.assert_close(y, torch.zeros_like(y), atol=0, rtol=0)


@pytest.mark.parametrize("operator", ["sum", "mean"])
def test_scalecut_equal_upper_and_lower_cuts_produce_zero(operator: str) -> None:
    x = torch.linspace(-3, 5, 64, dtype=torch.float64).reshape(1, 32, 2)
    w = torch.arange(32, dtype=x.dtype).reshape(1, 32, 1) % 5

    y = NestChannelScalecut(4, 1, [2, 4], operator=operator, nside_upper=[2, 4])(x, w)

    torch.testing.assert_close(y, torch.zeros_like(y), atol=0, rtol=0)


@pytest.mark.parametrize("operator", ["sum", "mean"])
def test_scalecut_none_upper_cut_preserves_existing_behavior(operator: str) -> None:
    x = torch.arange(64, dtype=torch.float64).reshape(1, 32, 2)
    w = torch.arange(32, dtype=x.dtype).reshape(1, 32, 1) % 5

    expected = NestChannelScalecut(4, 1, [2, 4], operator=operator)(x, w)
    y = NestChannelScalecut(4, 1, [2, 4], operator=operator, nside_upper=None)(x, w)

    torch.testing.assert_close(y, expected, atol=0, rtol=0)


@pytest.mark.parametrize("operator", ["sum", "mean"])
def test_scalecut_upper_cut_masks_nonfinite_values_and_has_finite_gradients(operator: str) -> None:
    x = torch.tensor([2, float("nan"), 4, float("inf")] + [float("nan")] * 28, dtype=torch.float64)
    x = x.reshape(1, 32, 1).requires_grad_()
    w = torch.tensor([1, 0, 2] + [0] * 29, dtype=x.dtype).reshape_as(x).requires_grad_()

    y = NestChannelScalecut(4, 1, [2], operator=operator, nside_upper=[1])(x, w)

    expected = torch.zeros_like(x)
    if operator == "sum":
        expected[:, :4] = 7.5
        expected[:, 4:16] = -2.5
    else:
        expected[:, 4:16] = -10 / 3
    torch.testing.assert_close(y, expected)
    y.square().sum().backward()
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(w.grad).all()
    assert torch.count_nonzero(x.grad[w == 0]) == 0


@pytest.mark.parametrize("operator", ["sum", "mean"])
def test_scalecut_upper_cut_gradients_match_finite_differences(operator: str) -> None:
    x = torch.linspace(0.3, 2.7, 32, dtype=torch.float64).reshape(1, 16, 2).requires_grad_()
    w = torch.linspace(0.5, 1.5, 16, dtype=x.dtype).reshape(1, 16, 1).requires_grad_()
    scalecut = NestChannelScalecut(4, 1, [2, 4], operator=operator, nside_upper=[1, 2])

    assert torch.autograd.gradcheck(scalecut, (x, w))


@pytest.mark.parametrize("operator", ["sum", "mean"])
def test_scalecut_upper_cut_preserves_complex_maps(operator: str) -> None:
    x = torch.arange(1, 17, dtype=torch.float64).to(torch.complex128).reshape(1, 16, 1) * (1 + 2j)
    w = torch.tensor([1, 1, 1, 1, 1, 2, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0], dtype=x.real.dtype).reshape_as(x)

    y = NestChannelScalecut(4, 1, [2], operator=operator, nside_upper=[1])(x, w)

    expected_real = torch.tensor([-1, 14, -2, -11], dtype=x.real.dtype)
    if operator == "mean":
        expected_real = torch.tensor([2.5, 6.25, 9, 0], dtype=x.real.dtype) - 44 / 9
    expected = (expected_real * (1 + 2j)).repeat_interleave(4).reshape_as(x)
    assert y.dtype == x.dtype
    torch.testing.assert_close(y, expected)


@pytest.mark.parametrize("nside_upper", [[], [1, 1]])
def test_scalecut_rejects_upper_cut_channel_count_mismatch(nside_upper: list[int]) -> None:
    with pytest.raises(ValueError, match="nside_upper"):
        NestChannelScalecut(4, 1, [2], nside_upper=nside_upper)


@pytest.mark.parametrize("nside_upper", [[0], [3], [1.5], [float("inf")], [8]])
def test_scalecut_rejects_invalid_upper_nsides(nside_upper: list[float]) -> None:
    with pytest.raises(ValueError, match="nside_upper"):
        NestChannelScalecut(4, 1, [2], nside_upper=nside_upper)


@pytest.mark.parametrize(("nside_base", "nside_lower", "nside_upper"), [(2, [4], [1]), (1, [2], [4])])
def test_scalecut_rejects_upper_cut_outside_hierarchy(
    nside_base: int, nside_lower: list[int], nside_upper: list[int]
) -> None:
    with pytest.raises(ValueError, match="nside"):
        NestChannelScalecut(4, nside_base, nside_lower, nside_upper=nside_upper)
