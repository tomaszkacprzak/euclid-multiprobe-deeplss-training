from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")


def test_parameter_names_from_physics_model_uses_all_params() -> None:
    from euclid_multiprobe_deeplss_training.plots import parameter_names_from_physics_model

    class PhysicsModel:
        all_params = ["omega_m", "sigma8"]

    assert parameter_names_from_physics_model(PhysicsModel()) == ["omega_m", "sigma8"]


def test_plot_targets_vs_predictions_labels_panels() -> None:
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from euclid_multiprobe_deeplss_training.plots import plot_targets_vs_predictions

    targets = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    predictions = np.array([[0.1, 0.25], [0.35, 0.45], [0.45, 0.65]])

    fig = plot_targets_vs_predictions(targets, predictions, ["omega_m", "sigma8"])

    assert fig.axes[0].get_xlabel() == "Target: omega_m"
    assert fig.axes[0].get_ylabel() == "Prediction: omega_m"
    assert fig.axes[1].get_xlabel() == "Target: sigma8"
    assert fig.axes[1].get_ylabel() == "Prediction: sigma8"
    plt.close(fig)
