from __future__ import annotations

import h5py
import pytest
import torch
from torch import nn


@pytest.mark.parametrize("channel_list", [False, True])
@pytest.mark.parametrize("num_examples", [None, 3, 4])
def test_predict_builds_models_with_training_physics_configuration(monkeypatch, tmp_path, channel_list, num_examples) -> None:
    from euclid_multiprobe_deeplss_training import dataloaders, prediction
    from euclid_multiprobe_deeplss_training.utils.config import Config

    captured = {}

    class FakePhysicsModel(nn.Module):
        num_channels = 2
        num_targets = 1
        channel_spins = [0, 2]
        params = ["omega_m"]

        def __init__(self, forward_model, **kwargs):
            super().__init__()
            captured["physics_model_kwargs"] = kwargs

    class FakePipeline:
        num_pixels = 3

        def __init__(self, **kwargs):
            captured["pipeline_kwargs"] = kwargs

        def __iter__(self):
            for ids in ([7, 8], [9]):
                if channel_list:
                    inputs = [torch.zeros(len(ids), 3), torch.full((len(ids), 5), 1 + 2j)]
                else:
                    inputs = torch.zeros(len(ids), 3, 2)
                yield inputs, torch.ones(len(ids), 1), torch.tensor(ids)

    class FakePredictionModel(nn.Module):
        def predict(self, inputs):
            assert not self.training
            assert not torch.is_grad_enabled()
            if channel_list:
                assert isinstance(inputs, list)
                assert inputs[1].is_complex()
                torch.testing.assert_close(inputs[1].imag, torch.full((len(inputs[0]), 5), 2.0))
                inputs = inputs[0]
            assert inputs.device.type == "cpu"
            return torch.ones(inputs.shape[0], 1, device=inputs.device)

    encoder = nn.Identity()

    def fake_build_encoder(*args, **kwargs):
        captured["encoder_kwargs"] = kwargs
        return encoder

    def fake_build_loss(*args, **kwargs):
        captured["loss_kwargs"] = kwargs
        return FakePredictionModel()

    monkeypatch.setattr(prediction, "load_physics_model_class", lambda _name: FakePhysicsModel)
    monkeypatch.setattr(prediction, "load_pixel_file", lambda _config: ([1, 2, 3], None, None, None))
    monkeypatch.setattr(prediction, "build_encoder", fake_build_encoder)
    monkeypatch.setattr(prediction, "build_loss", fake_build_loss)
    monkeypatch.setattr(dataloaders, "OntheflyPipeline", FakePipeline)

    config = Config.from_mapping({
        "training": {
            "records_pattern": "records-{0..1}.tar", "encoder_name": "corrs_transformer",
            "physics_model": "fake", "physics_model_args": {"custom_physics_option": 42},
            "batch_size": 2, "num_workers": 0, "embed_dim": 2, "encoder_args": {}, "loss_function": "mse",
        },
        "forward_model": {"analysis": {"n_side": 8, "n_side_down": 4}},
    })
    checkpoint = tmp_path / "checkpoint.pt"
    output = tmp_path / "predictions.h5"
    torch.save({"model_state_dict": {}}, checkpoint)
    prediction.predict(config, checkpoint=checkpoint, output_file=output, num_examples=num_examples, device="cpu")

    assert captured["physics_model_kwargs"]["custom_physics_option"] == 42
    assert captured["pipeline_kwargs"]["device"] == torch.device("cpu")
    assert captured["encoder_kwargs"]["physics_model"] is captured["pipeline_kwargs"]["physics_model"]
    assert captured["loss_kwargs"]["encoder"] is encoder
    expected_ids = [7, 8, 9] if num_examples is None or num_examples == 3 else [7, 8, 9, 7, 8]
    with h5py.File(output, "r") as handle:
        assert handle["indices"][:].tolist() == expected_ids
        assert handle["predictions"][:].tolist() == [[1.0]] * len(expected_ids)
        assert handle["labels"][:].tolist() == [[1.0]] * len(expected_ids)
        assert handle.attrs["parameter_names"].tolist() == ["omega_m"]

    monkeypatch.setattr(FakePipeline, "__iter__", lambda _self: iter(()))
    with pytest.raises(ValueError, match="did not produce any examples"):
        prediction.predict(config, checkpoint=checkpoint, output_file=output, num_examples=num_examples, device="cpu")
