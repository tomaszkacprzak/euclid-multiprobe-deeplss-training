from __future__ import annotations

import pytest


def test_save_and_load_checkpoint_round_trip(tmp_path) -> None:
    torch = pytest.importorskip("torch")

    from euclid_multiprobe_deeplss_training.training import TrainingConfig, load_checkpoint, save_checkpoint

    model = torch.nn.Linear(2, 1)
    loss_fn = torch.nn.Linear(1, 1)
    optimizer = torch.optim.Adam([*model.parameters(), *loss_fn.parameters()], lr=0.01)

    inputs = torch.tensor([[1.0, 2.0]])
    targets = torch.tensor([[3.0]])
    loss = torch.nn.MSELoss()(loss_fn(model(inputs)), targets)
    loss.backward()
    optimizer.step()

    train_losses = [1.25, 0.75]
    val_losses = [0.95]
    config = TrainingConfig(records_pattern="records/*.tar", max_steps=2, learning_rate=0.01)
    checkpoint_path = tmp_path / "checkpoint.pt"

    saved_weight = model.weight.detach().clone()
    saved_bias = model.bias.detach().clone()
    saved_loss_weight = loss_fn.weight.detach().clone()
    saved_loss_bias = loss_fn.bias.detach().clone()
    save_checkpoint(checkpoint_path, model, optimizer, 2, config, train_losses, val_losses, loss_fn)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    assert set(checkpoint) == {
        "model_state_dict",
        "optimizer_state_dict",
        "loss_state_dict",
        "step",
        "config",
        "train_losses",
        "val_losses",
    }
    assert checkpoint["step"] == 2
    assert checkpoint["train_losses"] == train_losses
    assert checkpoint["val_losses"] == val_losses
    assert checkpoint["config"]["records_pattern"] == "records/*.tar"

    with torch.no_grad():
        model.weight.fill_(0.0)
        model.bias.fill_(0.0)
        loss_fn.weight.fill_(0.0)
        loss_fn.bias.fill_(0.0)

    restored_step, restored_train_losses, restored_val_losses = load_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        torch.device("cpu"),
        loss_fn,
    )

    assert restored_step == 2
    assert restored_train_losses == train_losses
    assert restored_val_losses == val_losses
    assert torch.equal(model.weight, saved_weight)
    assert torch.equal(model.bias, saved_bias)
    assert torch.equal(loss_fn.weight, saved_loss_weight)
    assert torch.equal(loss_fn.bias, saved_loss_bias)


def test_prepare_checkpoint_dir_uses_tag_and_clears_existing_contents(tmp_path) -> None:
    pytest.importorskip("torch")

    from euclid_multiprobe_deeplss_training.training import TrainingConfig, _prepare_checkpoint_dir

    checkpoint_root = tmp_path / "checkpoints"
    run_dir = checkpoint_root / "experiment-a"
    nested_dir = run_dir / "old-subdir"
    nested_dir.mkdir(parents=True)
    (run_dir / "old-checkpoint.pt").write_text("old", encoding="utf-8")
    (nested_dir / "old-nested-checkpoint.pt").write_text("old", encoding="utf-8")
    other_run_dir = checkpoint_root / "experiment-b"
    other_run_dir.mkdir()
    (other_run_dir / "checkpoint.pt").write_text("keep", encoding="utf-8")

    config = TrainingConfig(
        records_pattern="records/*.tar",
        max_steps=1,
        checkpoint_dir=str(checkpoint_root),
        tag="experiment-a",
    )

    prepared_dir = _prepare_checkpoint_dir(config)

    assert prepared_dir == run_dir
    assert run_dir.is_dir()
    assert list(run_dir.iterdir()) == []
    assert (other_run_dir / "checkpoint.pt").read_text(encoding="utf-8") == "keep"


def test_prepare_checkpoint_dir_preserves_existing_contents_when_resuming(tmp_path) -> None:
    pytest.importorskip("torch")

    from euclid_multiprobe_deeplss_training.training import TrainingConfig, _prepare_checkpoint_dir

    checkpoint_root = tmp_path / "checkpoints"
    run_dir = checkpoint_root / "experiment-a"
    run_dir.mkdir(parents=True)
    existing_checkpoint = run_dir / "checkpoint-step-1.pt"
    existing_checkpoint.write_text("keep", encoding="utf-8")

    config = TrainingConfig(
        records_pattern="records/*.tar",
        max_steps=1,
        checkpoint_dir=str(checkpoint_root),
        tag="experiment-a",
        resume_from_checkpoint=str(existing_checkpoint),
    )

    prepared_dir = _prepare_checkpoint_dir(config)

    assert prepared_dir == run_dir
    assert existing_checkpoint.read_text(encoding="utf-8") == "keep"


def test_write_reproducibility_config_to_checkpoint_dir(tmp_path) -> None:
    pytest.importorskip("torch")
    yaml = pytest.importorskip("yaml")

    from euclid_multiprobe_deeplss_training.training import TrainingConfig, _write_reproducibility_config

    checkpoint_dir = tmp_path / "checkpoints" / "experiment-a"
    checkpoint_dir.mkdir(parents=True)
    config = TrainingConfig(
        records_pattern="records/*.tar",
        checkpoint_dir=str(tmp_path / "checkpoints"),
        tag="experiment-a",
        forward_model={"analysis": {"n_side": 512, "n_side_down": 64}},
        max_steps=5,
        use_wandb=False,
    )

    config_path = _write_reproducibility_config(checkpoint_dir, config)

    assert config_path == checkpoint_dir / "config.yaml"
    saved_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert saved_config["records_pattern"] == "records/*.tar"
    assert saved_config["checkpoint_dir"] == str(tmp_path / "checkpoints")
    assert saved_config["tag"] == "experiment-a"
    assert saved_config["model_args"] == {}
    assert saved_config["forward_model"] == {"analysis": {"n_side": 512, "n_side_down": 64}}
    assert saved_config["max_steps"] == 5


def test_training_config_accepts_model_args_from_top_level_and_model_section() -> None:
    pytest.importorskip("torch")

    from euclid_multiprobe_deeplss_training.training import TrainingConfig

    top_level_config = TrainingConfig.from_mapping(
        {
            "records_pattern": "records/*.tar",
            "max_steps": 1,
            "model_args": {"base_embed_dim": 512, "growth": "128"},
        }
    )

    nested_model_config = TrainingConfig.from_mapping(
        {
            "records_pattern": "records/*.tar",
            "max_steps": 1,
            "model": {"model_args": {"num_heads": 8}},
        }
    )

    assert top_level_config.model_args == {"base_embed_dim": 512, "growth": "128"}
    assert nested_model_config.model_args == {"num_heads": 8}


def test_training_config_rejects_non_mapping_model_args() -> None:
    pytest.importorskip("torch")

    from euclid_multiprobe_deeplss_training.training import TrainingConfig

    with pytest.raises(TypeError, match="model_args must be a mapping"):
        TrainingConfig.from_mapping(
            {
                "records_pattern": "records/*.tar",
                "max_steps": 1,
                "model_args": ["base_embed_dim", 512],
            }
        )


def test_evaluation_predictions_path_uses_run_checkpoint_dir(tmp_path) -> None:
    pytest.importorskip("torch")

    from euclid_multiprobe_deeplss_training.training import TrainingConfig, _evaluation_predictions_path

    checkpoint_root = tmp_path / "checkpoints"
    config = TrainingConfig(
        records_pattern="records/*.tar",
        max_steps=1,
        checkpoint_dir=str(checkpoint_root),
        evaluation_predictions_dir=str(tmp_path / "legacy-predictions"),
        tag="experiment-a",
    )

    assert _evaluation_predictions_path(config, 2) == checkpoint_root / "experiment-a" / "evaluation-epoch-0003.h5"


def test_evaluation_predictions_path_disabled_without_checkpoint_dir() -> None:
    pytest.importorskip("torch")

    from euclid_multiprobe_deeplss_training.training import TrainingConfig, _evaluation_predictions_path

    config = TrainingConfig(
        records_pattern="records/*.tar",
        max_steps=1,
        evaluation_predictions_dir="legacy-predictions",
        tag="experiment-a",
    )

    assert _evaluation_predictions_path(config, 0) is None


def test_checkpoint_stores_and_loads_wandb_resume_metadata(tmp_path) -> None:
    torch = pytest.importorskip("torch")

    from euclid_multiprobe_deeplss_training.training import (
        TrainingConfig,
        _wandb_info_from_checkpoint,
        save_checkpoint,
    )

    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    config = TrainingConfig(records_pattern="records/*.tar", max_steps=1, wandb_project="project")
    checkpoint_path = tmp_path / "checkpoint.pt"
    wandb_info = {"id": "run-id", "project": "project", "entity": "entity", "name": "run-name"}

    save_checkpoint(checkpoint_path, model, optimizer, 1, config, [0.5], [], wandb_info=wandb_info)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    assert checkpoint["wandb"] == wandb_info
    assert _wandb_info_from_checkpoint(checkpoint_path) == wandb_info


def test_prepare_checkpoint_dir_preserves_existing_contents_when_requested(tmp_path) -> None:
    pytest.importorskip("torch")

    from euclid_multiprobe_deeplss_training.training import TrainingConfig, _prepare_checkpoint_dir

    checkpoint_root = tmp_path / "checkpoints"
    run_dir = checkpoint_root / "experiment-a"
    run_dir.mkdir(parents=True)
    existing_checkpoint = run_dir / "checkpoint-step-1.pt"
    existing_checkpoint.write_text("keep", encoding="utf-8")

    config = TrainingConfig(
        records_pattern="records/*.tar",
        max_steps=1,
        checkpoint_dir=str(checkpoint_root),
        tag="experiment-a",
    )

    prepared_dir = _prepare_checkpoint_dir(config, clear_existing=False)

    assert prepared_dir == run_dir
    assert existing_checkpoint.read_text(encoding="utf-8") == "keep"
