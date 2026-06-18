"""Small, readable training utilities for DeepLSS regression experiments."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import torch
import wandb
import yaml
from torch import nn
from torch.utils.data import DataLoader, IterableDataset, get_worker_info


@dataclass(slots=True)
class TrainingConfig:
    """Normalized training configuration loaded from a YAML file.

    The parser accepts flat YAML keys and also checks a nested ``model`` section
    for the model fields.  This keeps small paper-code experiments convenient
    while still allowing the configuration file to grow later.
    """

    records_pattern: str
    batch_size: int = 32
    num_epochs: int | None = 1
    max_steps: int | None = None
    validation_fraction: float = 0.1
    learning_rate: float = 1.0e-3
    num_workers: int = 0
    checkpoint_dir: str | None = None
    checkpoint_every_steps: int = 0
    resume_from_checkpoint: str | None = None
    wandb_project: str | None = None
    wandb_run_name: str | None = None
    wandb_mode: str | None = None
    use_wandb: bool = True
    seed: int = 0
    drop_last: bool = False
    in_channels: int = 1
    hidden_channels: int = 64
    num_targets: int = 1
    num_blocks: int = 2
    dropout: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, raw_config: Mapping[str, Any]) -> TrainingConfig:
        """Create a config object from loaded YAML data."""
        model_config = raw_config.get("model", {}) or {}
        if not isinstance(model_config, Mapping):
            raise TypeError("The optional 'model' configuration section must be a mapping.")

        names = {item.name for item in fields(cls) if item.name != "extra"}
        values: dict[str, Any] = {}
        for name in names:
            if name in raw_config:
                values[name] = raw_config[name]
            elif name in model_config:
                values[name] = model_config[name]

        if "records_pattern" not in values:
            raise KeyError("Training config requires 'records_pattern'.")

        config = cls(**values)
        config._validate()
        config.extra = {key: value for key, value in raw_config.items() if key not in names}
        return config

    def _validate(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.num_epochs is None and self.max_steps is None:
            raise ValueError("Set at least one of num_epochs or max_steps.")
        if not 0.0 <= self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be in [0, 1).")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative.")
        if self.checkpoint_every_steps < 0:
            raise ValueError("checkpoint_every_steps must be non-negative.")


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file with ``yaml.safe_load``."""
    with Path(path).open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise TypeError("The YAML training config must load to a mapping.")
    return loaded


def build_records_dataset(records_pattern: str, config: Mapping[str, Any]) -> IterableDataset:
    """Build the user-supplied iterable records dataset.

    Implement this project-specific hook to read ``records_pattern`` and yield
    samples shaped like ``(map_tensor, target_tensor)``.  ``map_tensor`` should
    be a part-sky Healpix map with shape similar to ``(channels, pixels)`` (or
    ``(pixels, channels)``, if that becomes the final convention).  The
    ``target_tensor`` should contain ``float32`` regression target data.
    """

    raise NotImplementedError("Provide a project-specific records dataset builder.")


class StreamSplitDataset(IterableDataset):
    """Deterministically keep either the training or validation part of a stream."""

    def __init__(self, dataset: Iterable, validation_fraction: float, seed: int, split: str) -> None:
        self.dataset = dataset
        self.validation_fraction = validation_fraction
        self.seed = seed
        self.split = split

    def __iter__(self):
        for index, sample in enumerate(self.dataset):
            is_validation = _index_goes_to_validation(index, self.validation_fraction, self.seed)
            if (self.split == "validation") == is_validation:
                yield sample


class WorkerShardDataset(IterableDataset):
    """Shard an iterable stream across DataLoader workers."""

    def __init__(self, dataset: Iterable) -> None:
        self.dataset = dataset

    def __iter__(self):
        worker = get_worker_info()
        if worker is None:
            yield from self.dataset
            return
        for index, sample in enumerate(self.dataset):
            if index % worker.num_workers == worker.id:
                yield sample


def _index_goes_to_validation(index: int, validation_fraction: float, seed: int) -> bool:
    if validation_fraction <= 0.0:
        return False
    digest = hashlib.blake2b(f"{seed}:{index}".encode(), digest_size=8).digest()
    unit_interval = int.from_bytes(digest, "big") / float(2**64)
    return unit_interval < validation_fraction


def split_iterable_dataset(
    dataset: IterableDataset,
    validation_fraction: float,
    seed: int = 0,
) -> tuple[IterableDataset, IterableDataset]:
    """Return disjoint train and validation iterable streams."""
    train_dataset = StreamSplitDataset(dataset, validation_fraction, seed, "train")
    validation_dataset = StreamSplitDataset(dataset, validation_fraction, seed, "validation")
    return train_dataset, validation_dataset


def make_dataloader(dataset: IterableDataset, config: TrainingConfig, *, drop_last: bool | None = None) -> DataLoader:
    """Create a worker-sharded DataLoader for an iterable dataset."""
    return DataLoader(
        WorkerShardDataset(dataset),
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        drop_last=config.drop_last if drop_last is None else drop_last,
    )


class SmallRegressionNet(nn.Module):
    """Minimal fallback regression model used when callers do not pass one."""

    def __init__(self, config: TrainingConfig) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Flatten(), nn.LazyLinear(config.hidden_channels), nn.ReLU()]
        for _ in range(max(config.num_blocks - 1, 0)):
            layers.extend([nn.Linear(config.hidden_channels, config.hidden_channels), nn.ReLU()])
            if config.dropout > 0.0:
                layers.append(nn.Dropout(config.dropout))
        layers.append(nn.Linear(config.hidden_channels, config.num_targets))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_one_step(
    model: nn.Module,
    batch: tuple[torch.Tensor, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device | str,
) -> float:
    """Run one optimization step and return the scalar loss."""
    model.train()
    maps, labels = batch
    maps = maps.to(device)
    labels = labels.to(device=device, dtype=torch.float32)

    optimizer.zero_grad(set_to_none=True)
    predictions = model(maps)
    loss = loss_fn(predictions, labels)
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu())


@torch.no_grad()
def evaluate(model: nn.Module, dataloader: DataLoader, loss_fn: nn.Module, device: torch.device | str) -> float | None:
    """Evaluate one full validation stream pass and return the mean MSE."""
    model.eval()
    losses: list[float] = []
    for maps, labels in dataloader:
        maps = maps.to(device)
        labels = labels.to(device=device, dtype=torch.float32)
        predictions = model(maps)
        losses.append(float(loss_fn(predictions, labels).detach().cpu()))
    if not losses:
        return None
    return sum(losses) / len(losses)


def init_wandb(config: TrainingConfig | Mapping[str, Any]):
    """Initialize a Weights & Biases run when enabled by configuration.

    Local training should not require network access, so runs default to
    ``offline`` mode unless ``wandb_mode`` explicitly requests another mode.
    Set ``wandb_mode: disabled`` or ``use_wandb: false`` to skip wandb entirely.
    """
    config_dict = asdict(config) if isinstance(config, TrainingConfig) else dict(config)
    use_wandb = bool(config_dict.get("use_wandb", True))
    wandb_mode = config_dict.get("wandb_mode")
    if not use_wandb or wandb_mode == "disabled":
        return None

    wandb_project = config_dict.get("wandb_project")
    if not wandb_project:
        return None

    return wandb.init(
        project=wandb_project,
        name=config_dict.get("wandb_run_name"),
        mode=wandb_mode or "offline",
        config=config_dict,
    )


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    config: TrainingConfig | Mapping[str, Any],
    train_losses: list[float],
    val_losses: list[float],
) -> None:
    """Save model, optimizer, config, and loss-history state."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
            "config": asdict(config) if isinstance(config, TrainingConfig) else dict(config),
            "train_losses": train_losses,
            "val_losses": val_losses,
        },
        path,
    )


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device | str,
) -> tuple[int, list[float], list[float]]:
    """Restore model/optimizer state and return step plus loss histories.

    Iterable dataset stream position is not exactly restored during resume.
    Until a future dataset implementation supports deterministic seeking,
    for now restarting from a checkpoint restores model state, optimizer
    state, and global-step/loss-history bookkeeping.
    """
    checkpoint = torch.load(Path(path), map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return (
        int(checkpoint["step"]),
        list(checkpoint.get("train_losses", [])),
        list(checkpoint.get("val_losses", checkpoint.get("validation_losses", []))),
    )


def train(
    config_or_path: str | Path | Mapping[str, Any] | TrainingConfig,
    model: nn.Module | None = None,
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    """Run a compact train/evaluate/checkpoint loop.

    Resuming from a checkpoint restores the model, optimizer, and global-step
    state before the main loop starts.  Iterable dataset stream position is not
    exactly restored unless a future dataset implementation supports
    deterministic seeking; for now, restarts resume model/optimizer/global-step
    state rather than seeking to the prior stream item.
    """
    config = _coerce_config(config_or_path)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # Data streams.
    dataset = build_records_dataset(config.records_pattern, config.extra | {"config": config})
    train_dataset, validation_dataset = split_iterable_dataset(dataset, config.validation_fraction, config.seed)
    train_loader = make_dataloader(train_dataset, config, drop_last=config.drop_last)
    validation_loader = make_dataloader(validation_dataset, config, drop_last=False)

    # Model and optimizer.
    model = model or SmallRegressionNet(config)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.MSELoss()
    step = 0
    train_losses: list[float] = []
    validation_losses: list[float] = []

    if config.resume_from_checkpoint:
        step, train_losses, validation_losses = load_checkpoint(
            config.resume_from_checkpoint,
            model,
            optimizer,
            device,
        )

    run = init_wandb(config)
    train_start_time = time.perf_counter()
    examples_seen = 0

    # Training loop.
    for _epoch in range(config.num_epochs or 10**12):
        for batch in train_loader:
            step += 1
            train_loss = train_one_step(model, batch, optimizer, loss_fn, device)
            train_losses.append(train_loss)
            maps, _labels = batch
            examples_seen += int(maps.shape[0]) if hasattr(maps, "shape") and maps.ndim > 0 else config.batch_size
            elapsed_seconds = max(time.perf_counter() - train_start_time, 1.0e-12)
            current_learning_rate = optimizer.param_groups[0]["lr"]
            if run is not None:
                wandb.log(
                    {
                        "train/loss": train_loss,
                        "step": step,
                        "learning_rate": current_learning_rate,
                        "runtime/examples_per_second": examples_seen / elapsed_seconds,
                    },
                    step=step,
                )

            if config.checkpoint_dir and config.checkpoint_every_steps and step % config.checkpoint_every_steps == 0:
                checkpoint_path = Path(config.checkpoint_dir) / f"checkpoint-step-{step}.pt"
                save_checkpoint(
                    checkpoint_path,
                    model,
                    optimizer,
                    step,
                    config,
                    train_losses,
                    validation_losses,
                )
                if run is not None:
                    wandb.log(
                        {"checkpoint/saved": 1, "checkpoint/path": str(checkpoint_path), "step": step},
                        step=step,
                    )

            if config.max_steps is not None and step >= config.max_steps:
                break

        validation_loss = evaluate(model, validation_loader, loss_fn, device)
        if validation_loss is not None:
            validation_losses.append(validation_loss)
            if run is not None:
                wandb.log(
                    {
                        "validation/loss": validation_loss,
                        "step": step,
                        "learning_rate": optimizer.param_groups[0]["lr"],
                    },
                    step=step,
                )

        if config.max_steps is not None and step >= config.max_steps:
            break

    if config.checkpoint_dir:
        final_checkpoint_path = Path(config.checkpoint_dir) / "checkpoint-final.pt"
        save_checkpoint(
            final_checkpoint_path,
            model,
            optimizer,
            step,
            config,
            train_losses,
            validation_losses,
        )
        if run is not None:
            wandb.log(
                {"checkpoint/saved": 1, "checkpoint/path": str(final_checkpoint_path), "step": step},
                step=step,
            )
    if run is not None:
        run.finish()

    return {"model": model, "step": step, "train_losses": train_losses, "validation_losses": validation_losses}


def train_from_config(
    config_path: str | Path,
    *,
    resume_from_checkpoint: str | None = None,
    checkpoint_dir: str | None = None,
    max_steps: int | None = None,
    device: torch.device | str | None = None,
    wandb_mode: str | None = None,
) -> dict[str, Any]:
    """Train from a YAML config file with optional CLI-style overrides."""
    raw_config = load_config(config_path)
    overrides = {
        "resume_from_checkpoint": resume_from_checkpoint,
        "checkpoint_dir": checkpoint_dir,
        "max_steps": max_steps,
        "wandb_mode": wandb_mode,
    }
    raw_config.update({key: value for key, value in overrides.items() if value is not None})
    return train(raw_config, device=device)


def _coerce_config(config_or_path: str | Path | Mapping[str, Any] | TrainingConfig) -> TrainingConfig:
    if isinstance(config_or_path, TrainingConfig):
        return config_or_path
    if isinstance(config_or_path, str | Path):
        return TrainingConfig.from_mapping(load_config(config_or_path))
    return TrainingConfig.from_mapping(config_or_path)
