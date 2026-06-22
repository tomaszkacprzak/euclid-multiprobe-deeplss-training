"""Small, readable training utilities for DeepLSS regression experiments."""

from __future__ import annotations

import itertools
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import torch
import wandb
from torch import nn
from torch.utils.data import DataLoader

from .utils.config import load_config, with_forward_model_config
from .utils.logger import get_logger
from .networks.builder import build_model
from .networks.smoothing import NestChannelDownsampler, NestDownsampler

from msfm.onthefly_physics.onthefly_linear import OntheflyPhysicsModelLinear
from msfm.onthefly_pipeline import OntheflyPipeline

LOGGER = get_logger(__file__)


@dataclass(slots=True)
class TrainingConfig:
    """Normalized training configuration loaded from a YAML file.

    The parser accepts flat YAML keys and also checks a nested ``model`` section
    for the model fields.  This keeps small paper-code experiments convenient
    while still allowing the configuration file to grow later.
    """
    
    records_pattern: str
    model_name: str = "nested_transformer"
    config_forward_model: str | None = None
    forward_model: dict[str, Any] = field(default_factory=dict)
    batch_size: int = 32
    num_epochs: int | None = 1
    max_steps: int | None = None
    validation_fraction: float = 0.1
    learning_rate: float = 1.0e-3
    num_workers: int = 1
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



# Re-export shared helpers for callers and tests that import them from this module.
_with_forward_model_config = with_forward_model_config





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
    LOGGER.debug(f'Maps shape={maps.shape} size={maps.numel()*maps.itemsize/1024**2:.2f} MB')
    LOGGER.debug(f'Labels shape={labels.shape}')
    maps = maps.to(device=device, dtype=torch.float32)
    labels = labels.to(device=device, dtype=torch.float32)

    optimizer.zero_grad(set_to_none=True)
    LOGGER.debug('Running forward pass')
    predictions = model(maps)
    LOGGER.debug('Running loss')
    loss = loss_fn(predictions, labels)
    if not loss.requires_grad:
        raise RuntimeError(
            "The training loss is detached from the model parameters; "
            "check the model forward pass for torch.no_grad(), detach(), or non-PyTorch conversions."
        )

    LOGGER.debug('Running backward pass')
    loss.backward()
    LOGGER.debug('Running optimizer step')
    optimizer.step()
    return float(loss.detach().cpu())


@torch.no_grad()
def evaluate(model: nn.Module, dataloader: DataLoader, loss_fn: nn.Module, device: torch.device | str) -> float | None:
    """Evaluate one full validation stream pass and return the mean MSE."""
    model.eval()
    losses: list[float] = []
    for maps, labels in dataloader:
        maps = maps.to(device=device, dtype=torch.float32)
        labels = labels.to(device=device, dtype=torch.float32)
        predictions = model(maps)
        losses.append(float(loss_fn(predictions, labels).detach().cpu()))
    if not losses:
        return None
    return sum(losses) / len(losses)


def _validate_gradient_flow(model: nn.Module) -> None:
    """Fail fast when backpropagation produced no usable trainable gradients."""
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise RuntimeError("The model has no trainable parameters with requires_grad=True.")

    gradients = [parameter.grad for parameter in trainable_parameters if parameter.grad is not None]
    if not gradients:
        raise RuntimeError(
            "No gradients were produced for any trainable model parameter; "
            "the loss is not connected to the model output."
        )

    finite_gradient_count = sum(int(torch.isfinite(gradient).all().item()) for gradient in gradients)
    nonzero_gradient_count = sum(int(torch.any(gradient != 0).item()) for gradient in gradients)
    if finite_gradient_count != len(gradients):
        raise RuntimeError("Non-finite gradients were produced during backpropagation.")
    if nonzero_gradient_count == 0:
        raise RuntimeError(
            "All produced gradients are exactly zero; the optimizer step cannot update the model."
        )


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


def _print_initial_model_summary(
    model: nn.Module,
    dataloader: Any,
    device: torch.device | str,
) -> Any:
    """Print architecture and trainable-parameter table before training.

    The summary needs one forward pass to collect layer input/output shapes.
    The sampled batch is chained back onto the returned iterator so the first
    training epoch still sees the same data item.
    """
    from .modelprofile import _print_model_specification_table, _register_model_specification_hooks

    iterator = iter(dataloader)
    try:
        first_batch = next(iterator)
    except StopIteration:
        LOGGER.warning("Skipping model parameter table because the training dataloader produced no batches.")
        _print_model_specification_table(model, [])
        return iter(())

    rows, hooks = _register_model_specification_hooks(model)
    was_training = model.training
    model.eval()
    try:
        maps, _labels = first_batch
        with torch.no_grad():
            model(maps.to(device))
    finally:
        for handle in hooks:
            handle.remove()
        model.train(was_training)

    _print_model_specification_table(model, rows)
    return itertools.chain([first_batch], iterator)

import torch
import wandb


def safe_name(name):
    """
    W&B metric names are easier to work with when they avoid
    dots and unusual characters.
    """
    return name.replace(".", "_").replace("/", "_")


def get_gradient_stats(model, log_per_parameter=False):
    """
    Collect gradient statistics after loss.backward()
    and before optimizer.step().

    Returns a dictionary suitable for wandb.log().
    """
    logs = {}

    total_sq_norm = 0.0
    max_abs_grad = 0.0
    mean_abs_grads = []

    num_params_with_grad = 0
    num_params_without_grad = 0
    num_nonfinite_grads = 0

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        if p.grad is None:
            num_params_without_grad += 1
            continue

        g = p.grad.detach()

        num_params_with_grad += 1

        finite = torch.isfinite(g).all()
        if not finite:
            num_nonfinite_grads += 1
            continue

        grad_norm = g.norm(2)
        grad_mean_abs = g.abs().mean()
        grad_max_abs = g.abs().max()

        total_sq_norm += grad_norm.item() ** 2
        max_abs_grad = max(max_abs_grad, grad_max_abs.item())
        mean_abs_grads.append(grad_mean_abs.item())

        if log_per_parameter:
            n = safe_name(name)

            logs[f"grad_norm_{n}"] = grad_norm.item()
            logs[f"grad_mean_abs_{n}"] = grad_mean_abs.item()
            logs[f"grad_max_abs_{n}"] = grad_max_abs.item()

    total_grad_norm = total_sq_norm ** 0.5

    logs["grad_norm_total"] = total_grad_norm
    logs["grad_max_abs_global"] = max_abs_grad

    if mean_abs_grads:
        logs["grad_mean_abs_average"] = sum(mean_abs_grads) / len(mean_abs_grads)
    else:
        logs["grad_mean_abs_average"] = 0.0

    logs["grad_num_params_with_grad"] = num_params_with_grad
    logs["grad_num_params_without_grad"] = num_params_without_grad
    logs["grad_num_nonfinite"] = num_nonfinite_grads

    return logs


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

    nside_training = 256

    LOGGER.info(f"Training on {device} with config: {config}")

    physics_model = OntheflyPhysicsModelLinear(config.forward_model, 
                        scalers=True,
                        device=device).to(device)

    # Downsample all maps to the same nside
    smoothing_model = NestDownsampler(nside=config.forward_model["analysis"]["n_side"], 
                            nside_base=config.forward_model["analysis"]["n_side_down"], 
                            nside_lower=nside_training, 
                            operator="mean").to(device)
    
    # Downsample each channel to a different nside
    # smoothing_model = NestChannelDownsampler(nside=config.forward_model["analysis"]["n_side"], 
    #                     nside_base=config.forward_model["analysis"]["n_side_down"], 
    #                     nside_lower=[nside_training]*24, 
    #                     operator="mean").to(device)
                            
    loader = OntheflyPipeline(config.records_pattern, 
                              physics_model, 
                              smoothing_model=smoothing_model,
                              batch_size=config.batch_size, 
                              num_workers=config.num_workers,
                              pin_memory=True,
                              device=device)

    # Model 
    model = build_model(config.model_name, 
                    num_channels=physics_model.num_channels,
                    num_targets=physics_model.num_targets,
                    num_pixels=loader.num_pixels,
                    nside=nside_training,
                    nside_down=int(config.forward_model["analysis"]["n_side_down"]),
                    )
    model.to(device)

    print()
    LOGGER.info(f'Model: {config.model_name}')
    print(model)
    print()


    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.MSELoss()
    step = 0
    train_losses: list[float] = []
    validation_losses: list[float] = []
    
    print()
    LOGGER.info(f'Optimizer:')
    print(optimizer)
    print()



    # Checkpoints
    if config.resume_from_checkpoint:
        step, train_losses, validation_losses = load_checkpoint(
            config.resume_from_checkpoint,
            model,
            optimizer,
            device,
        )

    training_batches = _print_initial_model_summary(model, loader, device)

    run = init_wandb(config)
    train_start_time = time.perf_counter()
    examples_seen = 0

    # Training loop.
    LOGGER.info(f'Training loop starting with num_epochs={config.num_epochs}')
    for _epoch in range(config.num_epochs or 10**12):
        epoch_batches = training_batches if _epoch == 0 else loader

        for batch in epoch_batches:

        # overfit on a single batch
        # batch = next(iter(epoch_batches))
        # for _ in range(10000):

            step += 1

            # Main magic - update model
            train_loss = train_one_step(model, batch, optimizer, loss_fn, device)

            if step < 10:
                _validate_gradient_flow(model)

            # logging
            if step % 10 == 0:
                LOGGER.info(f'Train loss epoch={_epoch:>3d} step={step:>5d} loss={train_loss: .8e}')

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

                if step % 100 == 0:
                    grad_logs = get_gradient_stats(model, log_per_parameter=False)
                    wandb.log(grad_logs, step=step)

            # checkpoint management
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

        validation_loss = evaluate(model, loader, loss_fn, device)
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
    config_path = Path(config_path)
    raw_config = with_forward_model_config(load_config(config_path), config_path.parent)
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
        config_path = Path(config_or_path)
        return TrainingConfig.from_mapping(with_forward_model_config(load_config(config_path), config_path.parent))
    return TrainingConfig.from_mapping(with_forward_model_config(config_or_path))
