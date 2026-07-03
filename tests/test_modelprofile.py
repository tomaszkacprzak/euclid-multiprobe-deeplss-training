from __future__ import annotations

import pytest


def test_model_specification_includes_layers_that_create_parameters_during_forward() -> None:
    torch = pytest.importorskip("torch")

    from euclid_multiprobe_deeplss_training.modelprofile import _register_model_specification_hooks

    class LazyWrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = None

        def forward(self, x):
            if self.linear is None:
                self.linear = torch.nn.Linear(x.shape[-1], 2).to(device=x.device, dtype=x.dtype)
            return self.linear(x)

    model = LazyWrapper()
    rows, hooks = _register_model_specification_hooks(model)
    try:
        model(torch.ones(3, 4))
    finally:
        for handle in hooks:
            handle.remove()

    assert rows == [
        (
            "",
            "LazyWrapper",
            "(3, 4)",
            "float32",
            "(3, 2)",
            "float32",
            10,
        )
    ]


def test_model_specification_omits_parameterless_leaf_modules() -> None:
    torch = pytest.importorskip("torch")

    from euclid_multiprobe_deeplss_training.modelprofile import _register_model_specification_hooks

    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(4, 2))
    rows, hooks = _register_model_specification_hooks(model)
    try:
        model(torch.ones(3, 2, 2))
    finally:
        for handle in hooks:
            handle.remove()

    assert [row[1] for row in rows] == ["Linear"]


def test_modelprofile_profiles_nested_transformer_batches(monkeypatch) -> None:
    torch = pytest.importorskip("torch")

    from euclid_multiprobe_deeplss_training import modelprofile

    class FakePhysicsModel:
        def __init__(self, forward_model, *, device):
            self.forward_model = forward_model
            self.device = device
            self.num_channels = 3
            self.num_targets = 2

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
        def __init__(self, records_pattern, physics_model, smoothing_model, *, batch_size, num_workers, pin_memory, device):
            assert records_pattern == "records/*.tar"
            assert physics_model.forward_model["analysis"]["n_side_down"] == 1
            assert smoothing_model.kwargs["nside_base"] == 1
            assert batch_size == 2
            assert num_workers == 0
            assert pin_memory is True
            assert device == "cpu"
            self.num_pixels = 48

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
    build_model_calls = []

    monkeypatch.setattr(modelprofile.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(modelprofile, "OntheflyPhysicsModelLinear", FakePhysicsModel)
    monkeypatch.setattr(modelprofile, "NestChannelDownsampler", FakeSmoothingModel)
    monkeypatch.setattr(modelprofile, "OntheflyPipeline", FakePipeline)
    def fake_build_model(model_name, **kwargs):
        build_model_calls.append((model_name, kwargs))
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
            "model_args": {"base_embed_dim": 512},
        }
    )

    assert build_model_calls == [
        (
            "nested_transformer",
            {
                "num_channels": 3,
                "num_targets": 2,
                "num_pixels": 48,
                "nside": 2,
                "nside_down": 1,
                "model_args": {"base_embed_dim": 512},
            },
        )
    ]
    assert fake_model.inputs == [(2, 3, 12, 4)]
    assert len(outputs) == 1
    assert outputs[0].shape == (2, 2)
