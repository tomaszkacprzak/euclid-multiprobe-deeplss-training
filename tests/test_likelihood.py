from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
from euclid_multiprobe_deeplss_training.likelihood.likelihood_mdn import GaussianMixtureMDN  # noqa: E402
from euclid_multiprobe_deeplss_training.likelihood.likelihood_training import (  # noqa: E402
    build_likelihood,
    plot_likelihood_fit,
    train_likelihood,
)


def test_mdn_returns_one_finite_log_likelihood_per_sample() -> None:
    model = GaussianMixtureMDN(3, num_components=2, num_layers=1, hidden_dim=8)
    theta_obs = torch.randn(5, 3)
    theta_true = torch.randn(5, 3)

    result = model(theta_obs, theta_true)

    assert result.shape == (5,)
    assert torch.isfinite(result).all()


def test_fit_updates_mdn_and_reports_each_epoch(capsys) -> None:
    model = GaussianMixtureMDN(2, num_components=2, num_layers=1, hidden_dim=8)
    theta_true = torch.randn(12, 2)
    theta_obs = theta_true + 0.1 * torch.randn(12, 2)

    history = model.fit(theta_obs[:8], theta_true[:8], theta_obs[8:], theta_true[8:], num_epochs=2, batch_size=4, device="cpu")

    assert len(history["training"]) == len(history["validation"]) == 2
    assert capsys.readouterr().out.count("validation log likelihood") == 2


def test_mdn_checkpoint_round_trip(tmp_path) -> None:
    model = build_likelihood(2, {"model_type": "mdn", "model_args": {"num_components": 2}})
    path = tmp_path / "likelihood.pt"
    model.save(path)
    restored = build_likelihood(2, {"model_type": "mdn", "model_args": {"num_components": 2}})

    restored.load(path, map_location="cpu")

    for expected, actual in zip(model.parameters(), restored.parameters(), strict=True):
        assert torch.equal(expected, actual)


def test_plot_likelihood_fit_has_sample_and_surface_panel_per_parameter() -> None:
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    model = GaussianMixtureMDN(2, num_components=2, num_layers=1, hidden_dim=8)
    labels = torch.randn(5, 2)
    predictions = labels + 0.1 * torch.randn(5, 2)

    figure = plot_likelihood_fit(model, predictions, labels)

    panels = [axis for axis in figure.axes if axis.get_xlabel().startswith("Label")]
    assert len(panels) == 4
    for index, panel in enumerate(panels[:2]):
        assert panel.get_xlabel() == f"Label {index}"
        assert panel.get_ylabel() == f"Prediction {index}"
        assert panel.collections[0].get_offsets().shape == (5, 2)
        assert panel.collections[0].get_array().shape == (5,)
    for index, panel in enumerate(panels[2:]):
        assert panel.get_xlabel() == f"Label {index}"
        assert panel.get_ylabel() == f"Prediction {index}"
        assert panel.collections[0].get_array().size == 100 * 100


def test_train_likelihood_saves_plot_next_to_checkpoint(tmp_path) -> None:
    pytest.importorskip("matplotlib")
    h5py = pytest.importorskip("h5py")
    input_file = tmp_path / "predictions.h5"
    output_file = tmp_path / "models" / "likelihood.pt"
    labels = torch.randn(8, 2)
    with h5py.File(input_file, "w") as handle:
        handle["labels"] = labels.numpy()
        handle["predictions"] = (labels + 0.1 * torch.randn(8, 2)).numpy()

    train_likelihood(
        {
            "model_type": "mdn",
            "model_args": {"num_components": 2, "num_layers": 1, "hidden_dim": 8},
            "num_epochs": 1,
            "batch_size": 4,
        },
        input_file=input_file,
        output_file=output_file,
        device="cpu",
    )

    assert output_file.is_file()
    assert output_file.with_suffix(".png").is_file()
