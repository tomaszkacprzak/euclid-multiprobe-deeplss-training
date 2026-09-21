from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
from euclid_multiprobe_deeplss_training.likelihood.likelihood_mdn import GaussianMixtureMDN  # noqa: E402
from euclid_multiprobe_deeplss_training.likelihood.training import build_likelihood  # noqa: E402


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
