from __future__ import annotations

import pytest


def test_print_evaluation_statistics_reports_each_target_and_prediction_feature(monkeypatch) -> None:
    torch = pytest.importorskip("torch")

    from euclid_multiprobe_deeplss_training import training

    messages: list[str] = []
    monkeypatch.setattr(training.LOGGER, "info", messages.append)

    training._print_evaluation_statistics(
        [torch.tensor([[0.0, 2.0], [2.0, float("nan")]])],
        [torch.tensor([[0.0, 4.0], [4.0, float("inf")]])],
    )

    assert messages == [
        "Evaluation target and prediction statistics:",
        "Feature 0 targets: min=0, max=2, mean=1, std=1, exact_zeros=1, non_finite=0",
        "Feature 0 predictions: min=0, max=4, mean=2, std=2, exact_zeros=1, non_finite=0",
        "Feature 1 targets: min=2, max=2, mean=2, std=0, exact_zeros=0, non_finite=1",
        "Feature 1 predictions: min=4, max=4, mean=4, std=0, exact_zeros=0, non_finite=1",
    ]


def test_print_evaluation_statistics_handles_feature_with_no_finite_values(monkeypatch) -> None:
    torch = pytest.importorskip("torch")

    from euclid_multiprobe_deeplss_training import training

    messages: list[str] = []
    monkeypatch.setattr(training.LOGGER, "info", messages.append)

    training._print_evaluation_statistics(
        [torch.tensor([[float("nan")], [float("inf")]])],
        [torch.tensor([[float("-inf")], [float("nan")]])],
    )

    assert messages[1] == (
        "Feature 0 targets: min=nan, max=nan, mean=nan, std=nan, "
        "exact_zeros=0, non_finite=2"
    )
    assert messages[2] == (
        "Feature 0 predictions: min=nan, max=nan, mean=nan, std=nan, "
        "exact_zeros=0, non_finite=2"
    )
