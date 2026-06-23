from __future__ import annotations

import pytest


def test_modelprofile_profiles_nested_transformer_batches(monkeypatch) -> None:
    torch = pytest.importorskip("torch")

    from euclid_multiprobe_deeplss_training import modelprofile

    class FakePhysicsModel:
        num_channels = 3
        num_targets = 2

        def __init__(self, forward_model, *, scalers, device):
            assert scalers is True
            self.forward_model = forward_model
            self.device = device

        def to(self, device):
            self.device = device
            return self

    class FakeSmoothingModel:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def to(self, device):
            self.device = device
            return self

    class FakePipeline:
        num_pixels = 48

        def __init__(self, records_pattern, physics_model, smoothing_model, *, batch_size, num_workers, pin_memory, device):
            assert records_pattern == "records/*.tar"
            assert physics_model.forward_model["analysis"]["n_side_down"] == 1
            assert smoothing_model.kwargs["nside_base"] == 1
            assert batch_size == 2
            assert num_workers == 0
            assert pin_memory is True
            assert device == "cpu"

        def __iter__(self):
            yield torch.ones(2, 48, 3), torch.zeros(2, 1)

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inputs = []

        def forward(self, x):
            self.inputs.append(tuple(x.shape))
            return torch.full((x.shape[0], 2), 3.0, device=x.device)

    fake_model = FakeModel()

    monkeypatch.setattr(modelprofile.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(modelprofile, "OntheflyPhysicsModelLinear", FakePhysicsModel)
    monkeypatch.setattr(modelprofile, "NestChannelDownsampler", FakeSmoothingModel)
    monkeypatch.setattr(modelprofile, "OntheflyPipeline", FakePipeline)

    def fake_build_model(model_name, *, num_channels, num_targets, num_pixels, nside, nside_down):
        assert model_name == "nested_transformer"
        assert num_channels == 3
        assert num_targets == 2
        assert num_pixels == 48
        assert nside == 2
        assert nside_down == 1
        return fake_model

    monkeypatch.setattr(modelprofile, "build_model", fake_build_model)
    monkeypatch.setattr(modelprofile, "print_profiler_stats", lambda prof: None)

    outputs = modelprofile.modelprofile(
        {
            "records_pattern": "records/*.tar",
            "forward_model": {"analysis": {"n_side": 2, "n_side_down": 1}},
            "batch_size": 2,
            "validation_fraction": 0.0,
            "use_wandb": False,
            "in_channels": 3,
            "num_targets": 2,
        }
    )

    assert fake_model.inputs == [(2, 3, 12, 4)]
    assert len(outputs) == 1
    assert outputs[0].shape == (2, 2)
