from __future__ import annotations

import sys
import types

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
            },
        )
    ]
