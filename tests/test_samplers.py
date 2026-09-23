from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchebm")

from euclid_multiprobe_deeplss_training.likelihood.likelihood_mdn import GaussianMixtureMDN  # noqa: E402
from euclid_multiprobe_deeplss_training.samplers import ConditionalLikelihoodEnergy  # noqa: E402


class QuadraticLikelihood(GaussianMixtureMDN):
    def log_likelihood(self, theta_obs, theta_true):
        return -(theta_obs - theta_true).square().sum(dim=-1)


def test_conditional_likelihood_energy_is_finite_and_penalizes_boundaries() -> None:
    model = QuadraticLikelihood(1, num_components=1, num_layers=1, hidden_dim=4)
    energy_model = ConditionalLikelihoodEnergy(model)
    observations = torch.tensor([[0.5], [0.5], [0.5]])
    unconstrained = torch.tensor([[0.0], [-100.0], [100.0]])

    energy = energy_model(unconstrained, theta_obs=observations)

    assert torch.isfinite(energy).all()
    assert energy[1] > energy[0]
    assert energy[2] > energy[0]


def test_conditional_likelihood_energy_transforms_unit_box_endpoints_to_finite_values() -> None:
    parameters = torch.tensor([[0.0, 0.25, 0.75, 1.0]])

    unconstrained = ConditionalLikelihoodEnergy.to_unconstrained_space(parameters)
    restored = ConditionalLikelihoodEnergy.to_parameter_space(unconstrained)

    assert torch.isfinite(unconstrained).all()
    assert torch.all((restored > 0) & (restored < 1))
    assert torch.allclose(restored[:, 1:3], parameters[:, 1:3])
