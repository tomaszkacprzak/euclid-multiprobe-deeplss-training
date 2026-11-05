from __future__ import annotations

import torch

from euclid_multiprobe_deeplss_training.dataloaders import OntheflyPipeline
from euclid_multiprobe_deeplss_training.networks.smoothing import NestChannelScalecut, NestDownsampler


def _pipeline(batch, physics_model, *, smoother=None, downsampler=None):
    # Exercise batch processing without creating a WebDataset or worker processes.
    pipeline = object.__new__(OntheflyPipeline)
    pipeline.loader = [batch]
    pipeline.device = "cpu"
    pipeline.physics_model = physics_model
    pipeline.smoother = smoother
    pipeline.downsampler = downsampler
    return pipeline


def test_scalecut_uses_weights_from_unsmoothed_physics_channels() -> None:
    counts = torch.tensor([1.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    maps = torch.stack((torch.arange(8.0), counts), dim=-1)[None, ..., None]
    indices = torch.tensor([[7]])
    targets = torch.tensor([[2.0]])
    physics_channels = maps[..., 0].clone()
    physics_channels[..., 0] += 10
    calls = []

    class PhysicsModel:
        def __call__(self, batch):
            calls.append("physics")
            torch.testing.assert_close(batch[0], maps)
            return physics_channels.clone(), batch[2]

        def get_weight_maps(self, inputs):
            calls.append("weights")
            torch.testing.assert_close(inputs, physics_channels)
            weights = torch.ones_like(inputs)
            weights[..., 0] = inputs[..., 1]
            return weights

    smoother = NestChannelScalecut(nside=2, nside_base=1, nside_lower=[1, 1], operator="mean")
    pipeline = _pipeline((maps, indices, targets), PhysicsModel(), smoother=smoother)

    actual_inputs, actual_targets, actual_indices = next(iter(pipeline))

    # The first coarse pixel uses source counts; an empty coarse pixel gives zero.
    expected = torch.tensor([[[11.6, 1.25]] * 4 + [[0.0, 0.0]] * 4])
    torch.testing.assert_close(actual_inputs, expected)
    torch.testing.assert_close(actual_targets, targets)
    torch.testing.assert_close(actual_indices, indices)
    assert calls == ["physics", "weights"]


def test_raw_downsampler_reduces_resolution_before_physics() -> None:
    maps = torch.arange(16.0).reshape(1, 16, 1, 1)
    indices = torch.tensor([[7]])
    targets = torch.tensor([[2.0]])
    expected_maps = torch.tensor([1.5, 5.5, 9.5, 13.5]).reshape(1, 4, 1, 1)

    class PhysicsModel:
        def __call__(self, batch):
            torch.testing.assert_close(batch[0], expected_maps)
            return batch[0][..., 0], batch[2]

    downsampler = NestDownsampler(nside=4, nside_base=1, nside_lower=2)
    pipeline = _pipeline((maps, indices, targets), PhysicsModel(), downsampler=downsampler)

    actual_inputs, actual_targets, actual_indices = next(iter(pipeline))

    torch.testing.assert_close(actual_inputs, expected_maps[..., 0])
    torch.testing.assert_close(actual_targets, targets)
    torch.testing.assert_close(actual_indices, indices)


def test_unsmoothed_pipeline_does_not_require_weight_maps() -> None:
    maps = torch.arange(8.0).reshape(1, 8, 1, 1)
    indices = torch.tensor([[7]])
    targets = torch.tensor([[2.0]])

    class PhysicsModel:
        def __call__(self, batch):
            return batch[0][..., 0] + 10, batch[2]

    pipeline = _pipeline((maps, indices, targets), PhysicsModel())

    actual_inputs, actual_targets, actual_indices = next(iter(pipeline))

    torch.testing.assert_close(actual_inputs, maps[..., 0] + 10)
    torch.testing.assert_close(actual_targets, targets)
    torch.testing.assert_close(actual_indices, indices)
