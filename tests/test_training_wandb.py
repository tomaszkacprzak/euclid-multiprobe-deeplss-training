from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("wandb")

from euclid_multiprobe_deeplss_training import training
from euclid_multiprobe_deeplss_training.training import TrainingConfig, init_wandb


class DummyRun:
    id = "run-123"
    project = "project"
    entity = "entity"
    name = "run-name"


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


def test_init_wandb_resumes_from_checkpoint_metadata(monkeypatch) -> None:
    calls = []

    def fake_init(**kwargs):
        calls.append(kwargs)
        return DummyRun()

    monkeypatch.setattr(training.wandb, "init", fake_init)

    config = TrainingConfig(
        records_pattern="records/*.tar",
        wandb_project="configured-project",
        tag="configured-name",
    )

    run = init_wandb(
        config,
        {"id": "checkpoint-run-id", "project": "checkpoint-project", "entity": "checkpoint-entity", "name": "checkpoint-name"},
    )

    assert isinstance(run, DummyRun)
    assert calls[0]["id"] == "checkpoint-run-id"
    assert calls[0]["resume"] == "allow"
    assert calls[0]["project"] == "checkpoint-project"
    assert calls[0]["entity"] == "checkpoint-entity"
    assert calls[0]["name"] == "checkpoint-name"


def test_init_wandb_generates_and_stores_resume_metadata(monkeypatch) -> None:
    calls = []

    class MinimalRun:
        pass

    def fake_generate_id():
        return "generated-run-id"

    def fake_init(**kwargs):
        calls.append(kwargs)
        return MinimalRun()

    monkeypatch.setattr(training.wandb, "init", fake_init)
    util_stub = type("Util", (), {"generate_id": staticmethod(fake_generate_id)})()
    monkeypatch.setattr(training.wandb, "util", util_stub)

    config = TrainingConfig(
        records_pattern="records/*.tar",
        wandb_project="project",
        wandb_run_name="run-name",
    )

    run = init_wandb(config)

    assert calls[0]["id"] == "generated-run-id"
    assert "resume" not in calls[0]
    assert training._wandb_info_from_run(run) == {
        "id": "generated-run-id",
        "project": "project",
        "name": "run-name",
    }
