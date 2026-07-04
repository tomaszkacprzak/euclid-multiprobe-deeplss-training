from __future__ import annotations

import pytest


def test_full_cov_mixture_density_regressor_shapes_and_loss() -> None:
    torch = pytest.importorskip("torch")
    from torch.distributions import MixtureSameFamily

    from euclid_multiprobe_deeplss_training.losses.vimm import FullCovMixtureDensityRegressor

    torch.manual_seed(0)
    batch_size = 4
    x_dim = 3
    y_dim = 2
    n_components = 3
    model = FullCovMixtureDensityRegressor(
        x_dim=x_dim,
        y_dim=y_dim,
        hidden_dim=8,
        n_components=n_components,
        min_scale=1e-4,
    )

    x = torch.randn(batch_size, x_dim)
    y = torch.randn(batch_size, y_dim)

    dist = model(x)

    assert isinstance(dist, MixtureSameFamily)
    assert dist.batch_shape == torch.Size([batch_size])
    assert dist.event_shape == torch.Size([y_dim])
    assert dist.mixture_distribution.logits.shape == (batch_size, n_components)
    assert dist.component_distribution.loc.shape == (batch_size, n_components, y_dim)
    assert dist.component_distribution.scale_tril.shape == (batch_size, n_components, y_dim, y_dim)
    assert torch.all(dist.component_distribution.scale_tril.diagonal(dim1=-2, dim2=-1) > 0)

    log_prob = dist.log_prob(y)
    nll = model.nll(x, y)
    prediction = model.predict_mean(x)

    assert log_prob.shape == (batch_size,)
    assert nll.shape == ()
    assert torch.isfinite(nll)
    assert torch.allclose(nll, -log_prob.mean())
    assert prediction.shape == (batch_size, y_dim)
    assert torch.isfinite(prediction).all()
