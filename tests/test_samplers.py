from __future__ import annotations

import sys
import types

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")
pytest.importorskip("torchebm")

from euclid_multiprobe_deeplss_training.likelihood.likelihood_mdn import GaussianMixtureMDN  # noqa: E402
from euclid_multiprobe_deeplss_training.samplers import BaseBatchSampler, ConditionalLikelihoodEnergy  # noqa: E402


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


def test_triangle_chain_plot_unscales_samples_and_uses_parameter_names(monkeypatch) -> None:
    calls = {}

    class FakeTriangleChain:
        def __init__(self, **kwargs):
            calls["init"] = kwargs

        def contour_cl(self, chain, **kwargs):
            calls["chain"] = chain
            calls["contour"] = kwargs
            return "figure", "axes"

    monkeypatch.setitem(sys.modules, "trianglechain", types.SimpleNamespace(TriangleChain=FakeTriangleChain))
    samples = torch.tensor([[0.0, 0.25], [1.0, 0.75]])

    result = BaseBatchSampler.plot_triangle_chain(
        samples,
        ["Om", "H0"],
        {"Om": [0.1, 0.5], "H0": [64.0, 82.0]},
    )

    assert result == ("figure", "axes")
    np.testing.assert_allclose(calls["chain"], [[0.1, 68.5], [0.5, 77.5]])
    assert calls["init"] == {
        "labels": ["Om", "H0"],
        "fill": True,
        "de_kwargs": {"levels": [0.68, 0.95]},
    }
    assert calls["contour"] == {
        "show_values": True,
        "bestfit_method": "median",
        "levels_method": "percentile",
        "credible_interval": 0.68,
    }


def test_triangle_chain_plot_requires_one_parameter_name_per_column() -> None:
    with pytest.raises(ValueError, match="one parameter name"):
        BaseBatchSampler.plot_triangle_chain(torch.ones(3, 2), ["Om"], {"Om": [0.1, 0.5]})
