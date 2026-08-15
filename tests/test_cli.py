from __future__ import annotations

import sys
import types

import pytest

from euclid_multiprobe_deeplss_training.cli import main


def test_info_command_prints_package_name(capsys) -> None:
    exit_code = main(["info"])

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "euclid-multiprobe-deeplss-training" in captured.out


def test_train_command_passes_cli_overrides(monkeypatch) -> None:
    calls = []
    fake_training = types.ModuleType("euclid_multiprobe_deeplss_training.training")

    def fake_train_from_config(config_path, **kwargs):
        calls.append((config_path, kwargs))
        return {"step": 0}

    fake_training.train_from_config = fake_train_from_config
    monkeypatch.setitem(sys.modules, "euclid_multiprobe_deeplss_training.training", fake_training)

    exit_code = main(
        [
            "--config",
            "configs/example.yaml",
            "train",
            "--resume-from-checkpoint",
            "checkpoint.pt",
            "--checkpoint-dir",
            "checkpoints",
            "--max-steps",
            "10",
            "--device",
            "cpu",
            "--wandb-mode",
            "disabled",
        ]
    )

    assert exit_code == 0
    assert calls == [
        (
            "configs/example.yaml",
            {
                "resume_from_checkpoint": "checkpoint.pt",
                "checkpoint_dir": "checkpoints",
                "max_steps": 10,
                "device": "cpu",
                "wandb_mode": "disabled",
                "tag": "test-run",
            },
        )
    ]


def test_train_from_config_loads_forward_model_config(tmp_path, monkeypatch) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("wandb")

    from euclid_multiprobe_deeplss_training import training

    forward_model_path = tmp_path / "forward_model.yaml"
    forward_model_path.write_text("survey: euclid\nparams:\n  omega_m: 0.3\n", encoding="utf-8")
    config_path = tmp_path / "training.yaml"
    config_path.write_text(
        "\n".join(
            [
                "records_pattern: records/*.tar",
                "config_forward_model: forward_model.yaml",
                "max_steps: 0",
                "use_wandb: false",
            ]
        ),
        encoding="utf-8",
    )

    captured = {}

    def fake_train(config_or_path, *, device=None):
        captured["config"] = config_or_path
        captured["device"] = device
        return {"step": 0}

    monkeypatch.setattr(training, "train", fake_train)

    result = training.train_from_config(config_path, device="cpu")

    assert result == {"step": 0}
    assert captured["device"] == "cpu"
    assert captured["config"]["forward_model"] == {"survey": "euclid", "params": {"omega_m": 0.3}}


def test_reduce_mean_without_initialized_process_group_returns_local_copy() -> None:
    torch = pytest.importorskip("torch")

    from euclid_multiprobe_deeplss_training import training

    value = torch.tensor(3.5, requires_grad=True)

    reduced = training.reduce_mean(value)

    assert reduced.item() == pytest.approx(3.5)
    assert reduced is not value
    assert not reduced.requires_grad


def test_train_passes_forward_model_to_onthefly_pipeline(monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("wandb")

    from euclid_multiprobe_deeplss_training import training

    calls = []

    class FakePhysicsModel:
        def __init__(self, forward_model):
            self.forward_model = forward_model

    class FakePipeline:
        def __init__(self, records_pattern, physics_model, *, batch_size, num_workers):
            calls.append((records_pattern, physics_model.forward_model, batch_size, num_workers))

        def __iter__(self):
            yield torch.tensor([[1.0, 2.0]]), torch.tensor([[0.5]])

    monkeypatch.setattr(training, "OntheflyPhysicsModelLinear", FakePhysicsModel)
    monkeypatch.setattr(training, "OntheflyPipeline", FakePipeline)

    training.train(
        {
            "records_pattern": "records/*.tar",
            "forward_model": {"survey": "euclid"},
            "max_steps": 1,
            "validation_fraction": 0.0,
            "use_wandb": False,
            "num_targets": 1,
            "hidden_channels": 2,
            "num_blocks": 1,
        },
        device="cpu",
    )

    assert calls == [("records/*.tar", {"survey": "euclid"}, 32, 0)]


def test_datastats_command_requires_config() -> None:
    with pytest.raises(ValueError, match="datastats command requires --config"):
        main(["datastats"])


def test_datastats_command_passes_config(monkeypatch) -> None:
    calls = []
    fake_datastats = types.ModuleType("euclid_multiprobe_deeplss_training.datastats")

    def fake_datastats_from_config(config_path):
        calls.append(config_path)
        return []

    fake_datastats.datastats_from_config = fake_datastats_from_config
    monkeypatch.setitem(sys.modules, "euclid_multiprobe_deeplss_training.datastats", fake_datastats)

    exit_code = main(["--config", "configs/example.yaml", "datastats"])

    assert exit_code == 0
    assert calls == ["configs/example.yaml"]


def test_modelprofile_command_requires_config() -> None:
    with pytest.raises(ValueError, match="modelprofile command requires --config"):
        main(["modelprofile"])


def test_modelprofile_command_passes_config(monkeypatch) -> None:
    calls = []
    fake_modelprofile = types.ModuleType("euclid_multiprobe_deeplss_training.modelprofile")

    def fake_modelprofile_from_config(config_path):
        calls.append(config_path)
        return []

    fake_modelprofile.modelprofile_from_config = fake_modelprofile_from_config
    monkeypatch.setitem(sys.modules, "euclid_multiprobe_deeplss_training.modelprofile", fake_modelprofile)

    exit_code = main(["--config", "configs/example.yaml", "modelprofile"])

    assert exit_code == 0
    assert calls == ["configs/example.yaml"]


def test_calccls_command_requires_config() -> None:
    with pytest.raises(ValueError, match="calccls command requires --config"):
        main(["calccls"])


def test_calccls_command_passes_config(monkeypatch) -> None:
    calls = []
    fake_calccls = types.ModuleType("euclid_multiprobe_deeplss_training.calccls")

    def fake_calccls_from_config(config_path, **kwargs):
        calls.append((config_path, kwargs))

    fake_calccls.calccls_from_config = fake_calccls_from_config
    monkeypatch.setitem(sys.modules, "euclid_multiprobe_deeplss_training.calccls", fake_calccls)

    exit_code = main(["--config", "configs/example.yaml", "calccls", "--output-path", "spectra/results.h5"])

    assert exit_code == 0
    assert calls == [("configs/example.yaml", {"output_path": "spectra/results.h5"})]
