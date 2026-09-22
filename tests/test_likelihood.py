from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
from euclid_multiprobe_deeplss_training.likelihood.training import build_likelihood  # noqa: E402

from euclid_multiprobe_deeplss_training.likelihood.likelihood_cnf import ConditionalNormalizingFlowFM  # noqa: E402
from euclid_multiprobe_deeplss_training.likelihood.likelihood_mdn import GaussianMixtureMDN  # noqa: E402


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

    history = model.fit(
        theta_obs[:8], theta_true[:8], theta_obs[8:], theta_true[8:], num_epochs=2, batch_size=4, device="cpu"
    )

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
    expected = torch.distributions.Independent(
        torch.distributions.Normal(torch.zeros_like(theta_obs), torch.ones_like(theta_obs)), 1
    ).log_prob(theta_obs)

    assert torch.allclose(result, expected)


def test_cnffm_builder_and_flow_matching_loss() -> None:
    model = build_likelihood(
        2, {"model_type": "cnffm", "model_args": {"num_layers": 1, "hidden_dim": 8, "ode_steps": 2}}
    )
    theta_obs = torch.randn(4, 2)
    theta_true = torch.randn(4, 2)

    loss = model.training_loss(theta_obs, theta_true)
    loss.backward()

    assert isinstance(model, ConditionalNormalizingFlowFM)
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert all(parameter.grad is not None for parameter in model.parameters())
