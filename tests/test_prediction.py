from __future__ import annotations

import sys
import types

import torch
from torch import nn


def test_predict_builds_models_with_training_physics_configuration(monkeypatch, tmp_path) -> None:
    from euclid_multiprobe_deeplss_training import dataloaders, prediction
    from euclid_multiprobe_deeplss_training.training import TrainingConfig

    captured: dict[str, object] = {}

    class FakeH5File:
        def __init__(self, path, mode):
            captured["output_path"] = path
            captured["output_mode"] = mode
            captured["datasets"] = {}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def create_dataset(self, name, data):
            captured["datasets"][name] = data

    fake_h5py = types.ModuleType("h5py")
    fake_h5py.File = FakeH5File
    monkeypatch.setitem(sys.modules, "h5py", fake_h5py)

    class FakePhysicsModel(nn.Module):
        num_channels = 2
        num_targets = 1
        channel_spins = [0, 2]

        def __init__(self, forward_model, **kwargs):
            super().__init__()
            captured["forward_model"] = forward_model
            captured["physics_model_kwargs"] = kwargs

    class FakePipeline:
        num_pixels = 3

        def __init__(self, **kwargs):
            captured["pipeline_kwargs"] = kwargs

        def __iter__(self):
            yield torch.zeros(1, 3, 2), torch.ones(1, 1), torch.tensor([7])

    class FakePredictionModel(nn.Module):
        def predict(self, inputs):
            return torch.ones(inputs.shape[0], 1, device=inputs.device)

    encoder = nn.Identity()

    def fake_build_encoder(*args, **kwargs):
        captured["encoder_args"] = args
        captured["encoder_kwargs"] = kwargs
        return encoder

    def fake_build_loss(*args, **kwargs):
        captured["loss_args"] = args
        captured["loss_kwargs"] = kwargs
        return FakePredictionModel()

    monkeypatch.setattr(prediction, "load_physics_model_class", lambda _name: FakePhysicsModel)
    monkeypatch.setattr(prediction, "load_pixel_indices", lambda _config: [1, 2, 3])
    monkeypatch.setattr(prediction, "build_encoder", fake_build_encoder)
    monkeypatch.setattr(prediction, "build_loss", fake_build_loss)
    monkeypatch.setattr(dataloaders, "OntheflyPipeline", FakePipeline)

    config = TrainingConfig(
        records_pattern="records-{0..1}.tar",
        encoder_name="corrs_transformer",
        physics_model_args={"custom_physics_option": 42},
        forward_model={"analysis": {"n_side": 8, "n_side_down": 4}},
    )
    checkpoint = tmp_path / "checkpoint.pt"
    output = tmp_path / "predictions.h5"
    torch.save({"model_state_dict": {}}, checkpoint)

    prediction.predict(
        config,
        checkpoint=checkpoint,
        output_file=output,
        num_examples=1,
        device="cpu",
    )

    assert captured["physics_model_kwargs"]["custom_physics_option"] == 42
    physics_model = captured["pipeline_kwargs"]["physics_model"]
    assert captured["encoder_kwargs"]["physics_model"] is physics_model
    assert captured["loss_kwargs"]["encoder"] is encoder
    assert captured["output_path"] == output
    assert captured["datasets"]["indices"].tolist() == [7]
    assert captured["datasets"]["predictions"].tolist() == [[1.0]]
