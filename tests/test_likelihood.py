from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
from euclid_multiprobe_deeplss_training.likelihood.likelihood_cnf import ConditionalNormalizingFlowFM  # noqa: E402
from euclid_multiprobe_deeplss_training.likelihood.likelihood_mdn import GaussianMixtureMDN  # noqa: E402
from euclid_multiprobe_deeplss_training.likelihood.likelihood_training import (  # noqa: E402
    build_likelihood,
    plot_likelihood_fit,
    plot_posterior_samples,
    sample_posterior_metropolis_hastings,
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


def test_cnffm_returns_one_finite_log_likelihood_per_sample() -> None:
    model = ConditionalNormalizingFlowFM(3, num_layers=1, hidden_dim=8, ode_steps=2)
    theta_obs = torch.randn(5, 3)
    theta_true = torch.randn(5, 3)

    result = model(theta_obs, theta_true)

    assert result.shape == (5,)
    assert torch.isfinite(result).all()


def test_cnffm_zero_velocity_has_standard_normal_likelihood() -> None:
    model = ConditionalNormalizingFlowFM(2, num_layers=1, hidden_dim=8, ode_steps=2)
    for parameter in model.parameters():
        torch.nn.init.zeros_(parameter)
    theta_obs = torch.randn(5, 2)
    theta_true = torch.randn(5, 2)

    result = model(theta_obs, theta_true)
    expected = torch.distributions.Independent(torch.distributions.Normal(torch.zeros_like(theta_obs), torch.ones_like(theta_obs)), 1).log_prob(
        theta_obs
    )

    assert torch.allclose(result, expected)


def test_cnffm_builder_and_flow_matching_loss() -> None:
    model = build_likelihood(2, {"model_type": "cnffm", "model_args": {"num_layers": 1, "hidden_dim": 8, "ode_steps": 2}})
    theta_obs = torch.randn(4, 2)
    theta_true = torch.randn(4, 2)

    loss = model.training_loss(theta_obs, theta_true)
    loss.backward()

    assert isinstance(model, ConditionalNormalizingFlowFM)
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_metropolis_hastings_samples_inside_box_prior() -> None:
    class QuadraticLikelihood(GaussianMixtureMDN):
        evaluated_observations = []
        evaluated_parameters = []

        def log_likelihood(self, theta_obs, theta_true):
            self.evaluated_observations.append(theta_obs.clone())
            self.evaluated_parameters.append(theta_true.clone())
            return -(theta_obs - theta_true).square().sum(dim=-1)

    model = QuadraticLikelihood(2, num_components=1, num_layers=1, hidden_dim=4)
    observation = torch.tensor([0.25, 0.75])
    initial = torch.tensor([0.5, 0.5])

    samples = sample_posterior_metropolis_hastings(model, observation, initial, num_samples=20, burn_in=5, seed=7)

    assert samples.shape == (20, 2)
    assert torch.all(samples >= 0)
    assert torch.all(samples <= 1)
    assert all(torch.equal(evaluated, observation.unsqueeze(0)) for evaluated in model.evaluated_observations)
    assert all(torch.all((parameters >= 0) & (parameters <= 1)) for parameters in model.evaluated_parameters)


def test_plot_posterior_samples_has_one_histogram_per_parameter() -> None:
    pytest.importorskip("matplotlib")
    samples = torch.tensor([[0.25, 0.75], [0.5, 0.5]])
    theta_true = torch.tensor([0.4, 0.6])

    figure = plot_posterior_samples(samples, theta_true)

    assert len(figure.axes) == 2
    assert [axis.get_xlabel() for axis in figure.axes] == ["Parameter 0", "Parameter 1"]


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
