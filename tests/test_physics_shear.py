from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from euclid_multiprobe_deeplss_training.physics.onthefly_linear_shear import OntheflyPhysicsModelLinearShear
from euclid_multiprobe_deeplss_training.training import load_physics_model_class


class _Smoother(nn.Module):
    """Match nestconv's different weighted and scalar return signatures."""

    def __init__(self, factor, weighted):
        super().__init__()
        self.register_buffer("factor", torch.tensor(factor))
        self.weighted = weighted

    def forward(self, values, weights=None):
        if self.weighted:
            assert weights is not None
            return values * self.factor, weights
        assert weights is None
        return values * self.factor


@pytest.mark.parametrize("small,large,factor", [(True, True, 0.375), (True, False, 0.5), (False, True, 0.75)])
def test_shear_scale_cuts_process_every_channel(small, large, factor):
    model = OntheflyPhysicsModelLinearShear.__new__(OntheflyPhysicsModelLinearShear)
    nn.Module.__init__(model)
    model.num_channels = 18
    model.small_scale_selection = {"WL": 4, "GC": 16} if small else None
    model.large_scale_selection = 6 if large else None
    model.small_scale_smoothers = nn.ModuleList([_Smoother(0.5, j < 6) for j in range(18)])
    model.large_scale_smoothers = nn.ModuleList([_Smoother(0.25, j < 6) for j in range(18)])
    inputs = [torch.ones(1, 12, dtype=torch.complex64 if j < 6 else torch.float32) * (j + 1) for j in range(18)]
    weights = [torch.ones(1, 12) if j < 6 else None for j in range(18)]

    actual = model.apply_smoothing(inputs, weights)

    assert len(actual) == 18
    for channel, original in zip(actual, inputs, strict=True):
        torch.testing.assert_close(channel, original * factor)
        assert channel.dtype == original.dtype


def test_shear_batch_one_with_smoothing_and_scaling(monkeypatch):
    created = []

    class FakeNestconv(_Smoother):
        def __init__(self, *, weighted, **kwargs):
            super().__init__(0.5, weighted)
            created.append(self)

        def stage_info(self):
            return []

    monkeypatch.setitem(sys.modules, "nestconv", SimpleNamespace(nestconv=FakeNestconv))
    config = {
        "master_seed": 42,
        "survey": {
            "WL": {"n_gal": [14580.] * 6, "z_med": [0.62] * 6},
            "GC": {"n_gal": [2008.8] * 6, "z_med": [0.62] * 6},
        },
    }
    model = OntheflyPhysicsModelLinearShear(
        config, device="cpu", nside=1, indices_pixels=np.arange(12),
        num_samples_prior=4, scalers=True,
        small_scale_selection={"WL": 4, "GC": 16}, large_scale_selection=6,
    )
    assert len(created) == 4  # Two small-scale and two large-scale geometries.
    assert model.small_scale_smoothers[6] is model.small_scale_smoothers[17]
    assert all(smoother is not None for smoother in model.large_scale_smoothers)
    assert len(list(model.buffers())) == 4
    hard_params = torch.tensor([[0.3, 0.8, 0.05, 70., 0.96, -1., 1e13, 0.]])
    channels, targets = model((torch.zeros(1, 12, 6, 6), torch.tensor([[0, 0, 0]]), hard_params))

    assert len(channels) == 18
    assert targets.shape == (1, 26)
    assert torch.isfinite(targets).all()
    for index, channel in enumerate(channels):
        assert channel.shape == (1, 12)
        assert channel.is_complex() == (index < 6)
        assert torch.isfinite(channel).all()
    # Counts are scaled by 1e-5 (sources) and 1e-4 (lenses).
    assert channels[6].mean() < 1000
    assert channels[12].mean() < 1000
    assert channels[12].mean() > 0


@pytest.mark.parametrize("name,class_name", [
    ("onthefly_linear_shear", "OntheflyPhysicsModelLinearShear"),
    ("onthefly_linear_convergence", "OntheflyPhysicsModelLinearConvergence"),
])
def test_configured_physics_models_resolve(name, class_name):
    assert load_physics_model_class(name).__name__ == class_name
