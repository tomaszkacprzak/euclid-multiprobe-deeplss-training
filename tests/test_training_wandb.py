from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("wandb")

from euclid_multiprobe_deeplss_training import training
from euclid_multiprobe_deeplss_training.training import TrainingConfig, init_wandb


class DummyRun:
    pass


def test_init_wandb_skips_disabled_mode(monkeypatch) -> None:
    def fail_init(**_kwargs):
        raise AssertionError("wandb.init should not be called")

    monkeypatch.setattr(training.wandb, "init", fail_init)

    config = TrainingConfig(records_pattern="records/*.tar", wandb_project="project", wandb_mode="disabled")

    assert init_wandb(config) is None


def test_init_wandb_skips_use_wandb_false(monkeypatch) -> None:
    def fail_init(**_kwargs):
        raise AssertionError("wandb.init should not be called")

    monkeypatch.setattr(training.wandb, "init", fail_init)

    config = TrainingConfig(records_pattern="records/*.tar", wandb_project="project", use_wandb=False)

    assert init_wandb(config) is None


def test_init_wandb_defaults_to_offline_mode(monkeypatch) -> None:
    calls = []

    def fake_init(**kwargs):
        calls.append(kwargs)
        return DummyRun()

    monkeypatch.setattr(training.wandb, "init", fake_init)

    config = TrainingConfig(
        records_pattern="records/*.tar",
        wandb_project="project",
        wandb_run_name="run-name",
    )

    run = init_wandb(config)

    assert isinstance(run, DummyRun)
    assert calls[0]["project"] == "project"
    assert calls[0]["name"] == "run-name"
    assert calls[0]["mode"] == "offline"
    assert calls[0]["config"]["wandb_project"] == "project"
