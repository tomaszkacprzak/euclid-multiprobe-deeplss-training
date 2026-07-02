"""Small, readable training utilities for DeepLSS regression experiments."""

from __future__ import annotations

import itertools
import shutil
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any
import psutil
import torch
import torch.distributed as dist
import wandb
import yaml
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from .utils.config import load_config, with_forward_model_config
from .utils.logger import get_logger

LOGGER = get_logger(__file__)

OntheflyPhysicsModelLinear = None
OntheflyPipeline = None
build_loss = None
build_model = None
NestDownsampler = None


@dataclass(slots=True)
class TrainingConfig:
    """Normalized training configuration loaded from a YAML file.

    The parser accepts flat YAML keys and also checks a nested ``model`` section
    for the model fields.  This keeps small paper-code experiments convenient
    while still allowing the configuration file to grow later.
    """
    
    records_pattern: str
    model_name: str = "nested_transformer"
    model_args: dict[str, Any] = field(default_factory=dict)
    config_forward_model: str | None = None
    forward_model: dict[str, Any] = field(default_factory=dict)
    loss_function: str = "mse"
    batch_size: int = 32
    num_epochs: int | None = 1
    max_steps: int | None = None
    learning_rate: float = 1.0e-3
    num_workers: int = 1
    checkpoint_dir: str | None = None
    checkpoint_every_steps: int = 0
    evaluation_predictions_dir: str | None = None
    resume_from_checkpoint: str | None = None
    tag: str = "test-run"
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
    use_ddp: bool = True
    ddp_backend: str | None = None
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
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative.")
        if self.checkpoint_every_steps < 0:
            raise ValueError("checkpoint_every_steps must be non-negative.")
        if not isinstance(self.model_args, Mapping):
            raise TypeError("model_args must be a mapping.")




def _ddp_env_world_size() -> int:
    """Return torchrun world size from the environment, defaulting to one process."""
    import os

    return int(os.environ.get("WORLD_SIZE", "1"))


def _is_main_process() -> bool:
    """Return True for rank zero and for non-distributed runs."""
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def _unwrap_parallel_module(module: nn.Module) -> nn.Module:
    """Return the original module behind DistributedDataParallel wrappers."""
    return module.module if isinstance(module, DDP) else module


def _setup_ddp(config: TrainingConfig, requested_device: torch.device | str | None) -> tuple[bool, int, int, int, torch.device]:
    """Initialize DDP from torchrun environment variables and choose this rank's device."""
    import os

    world_size = _ddp_env_world_size()
    ddp_enabled = bool(config.use_ddp and world_size > 1)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))

    if ddp_enabled:
        backend = config.ddp_backend or ("nccl" if torch.cuda.is_available() else "gloo")
        if not dist.is_initialized():
            dist.init_process_group(backend=backend)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device(requested_device or "cpu")
    else:
        device = torch.device(requested_device or ("cuda" if torch.cuda.is_available() else "cpu"))

    return ddp_enabled, rank, world_size, local_rank, device


def _ddp_barrier() -> None:
    """Synchronize ranks when DDP is active."""
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


# Re-export shared helpers for callers and tests that import them from this module.
_with_forward_model_config = with_forward_model_config






@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device | str,
    num_examples: int = 10000,
    predictions_path: str | Path | None = None,
) -> float | None:
    """Evaluate one full validation stream pass and optionally save targets/predictions."""
    model.eval()
    losses: list[float] = []
    target_batches: list[torch.Tensor] = []
    prediction_batches: list[torch.Tensor] = []
    batch_size = dataloader.batch_size
    num_examples_seen = 0
    for maps, labels in dataloader:
        maps = maps.to(device=device, dtype=torch.float32)
        labels = labels.to(device=device, dtype=torch.float32)
        predictions = model(maps)
        losses.append(float(loss_fn(predictions, labels).detach().cpu()))
        if predictions_path is not None:
            target_batches.append(labels.detach().cpu())
            prediction_batches.append(predictions.detach().cpu())
        num_examples_seen += maps.shape[0]
        if num_examples_seen >= num_examples:
            break
    if not losses:
        return None
    if predictions_path is not None:
        _save_evaluation_predictions(predictions_path, target_batches, prediction_batches)
    return sum(losses) / len(losses)


def _save_evaluation_predictions(
    path: str | Path,
    target_batches: list[torch.Tensor],
    prediction_batches: list[torch.Tensor],
) -> None:
    """Save concatenated evaluation targets and predictions to an HDF5 file."""
    if len(target_batches) != len(prediction_batches):
        raise ValueError("Targets and predictions must have the same number of batches.")
    if not target_batches:
        return

    targets = torch.cat(target_batches, dim=0).numpy()
    predictions = torch.cat(prediction_batches, dim=0).numpy()
    if targets.shape != predictions.shape:
        raise ValueError(
            f"Targets and predictions must have the same shape, got {targets.shape} and {predictions.shape}."
        )

    import h5py

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("targets", data=targets)
        handle.create_dataset("predictions", data=predictions)


def _evaluation_predictions_path(config: TrainingConfig, step: int) -> Path | None:
    """Return the run-specific HDF5 path for per-step evaluation arrays, if enabled."""
    if config.checkpoint_dir is None:
        return None
    return Path(config.checkpoint_dir) / config.tag / f"evaluation-step-{step:06d}.h5"


def _latest_checkpoint_path(checkpoint_dir: Path) -> Path:
    """Return the rolling checkpoint path used for restarts and completed runs."""
    return checkpoint_dir / "checkpoint-latest.pt"


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


def _wandb_info_from_run(run: Any | None) -> dict[str, Any] | None:
    """Return checkpoint-safe metadata needed to resume a W&B run later."""
    if run is None:
        return None

    info = {
        "id": getattr(run, "id", None),
        "project": getattr(run, "project", None),
        "entity": getattr(run, "entity", None),
        "name": getattr(run, "name", None),
    }
    return {key: value for key, value in info.items() if value is not None}


def _wandb_info_from_checkpoint(path: str | Path | None) -> dict[str, Any] | None:
    """Load W&B resume metadata from a checkpoint without restoring training state."""
    if path is None:
        return None

    checkpoint = torch.load(Path(path), map_location="cpu")
    wandb_info = checkpoint.get("wandb")
    if isinstance(wandb_info, Mapping):
        return dict(wandb_info)
    return None


def init_wandb(config: TrainingConfig | Mapping[str, Any], wandb_info: Mapping[str, Any] | None = None):
    """Initialize a Weights & Biases run when enabled by configuration.

    Local training should not require network access, so runs default to
    ``offline`` mode unless ``wandb_mode`` explicitly requests another mode.
    Set ``wandb_mode: disabled`` or ``use_wandb: false`` to skip wandb entirely.
    When checkpoint metadata contains a W&B run id, reuse it so resumed
    training continues the same run.
    """
    config_dict = asdict(config) if isinstance(config, TrainingConfig) else dict(config)
    use_wandb = bool(config_dict.get("use_wandb", True))
    wandb_mode = config_dict.get("wandb_mode")
    if not use_wandb or wandb_mode == "disabled":
        return None

    wandb_info_dict = dict(wandb_info or {})
    wandb_project = wandb_info_dict.get("project") or config_dict.get("wandb_project")
    if not wandb_project:
        return None

    init_kwargs = {
        "project": wandb_project,
        "name": config_dict.get("wandb_run_name") or config_dict.get("tag"),
        "mode": wandb_mode or "offline",
        "config": config_dict,
    }
    if wandb_info_dict.get("entity") is not None:
        init_kwargs["entity"] = wandb_info_dict["entity"]
    if wandb_info_dict.get("id") is not None:
        init_kwargs["id"] = wandb_info_dict["id"]
        init_kwargs["resume"] = "allow"
        if wandb_info_dict.get("name") is not None and config_dict.get("wandb_run_name") is None:
            init_kwargs["name"] = wandb_info_dict["name"]

    return wandb.init(**init_kwargs)


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    config: TrainingConfig | Mapping[str, Any],
    train_losses: list[float],
    val_losses: list[float],
    loss_fn: nn.Module | None = None,
    wandb_info: Mapping[str, Any] | None = None,
) -> None:
    """Save model, loss function, optimizer, config, loss history, and W&B resume metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": _unwrap_parallel_module(model).state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "config": asdict(config) if isinstance(config, TrainingConfig) else dict(config),
        "train_losses": train_losses,
        "val_losses": val_losses,
    }
    if wandb_info is not None:
        checkpoint["wandb"] = dict(wandb_info)
    if loss_fn is not None:
        checkpoint["loss_state_dict"] = _unwrap_parallel_module(loss_fn).state_dict()
    torch.save(checkpoint, path)


def _prepare_checkpoint_dir(config: TrainingConfig, *, clear_existing: bool | None = None) -> Path | None:
    """Return the run-specific checkpoint directory, preserving contents when resuming."""
    if not config.checkpoint_dir:
        return None

    if clear_existing is None:
        clear_existing = not bool(config.resume_from_checkpoint)

    checkpoint_dir = Path(config.checkpoint_dir) / config.tag
    if checkpoint_dir.exists():
        if not checkpoint_dir.is_dir():
            raise NotADirectoryError(f"Checkpoint path exists and is not a directory: {checkpoint_dir}")
        if clear_existing:
            for child in checkpoint_dir.iterdir():
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir


def _write_reproducibility_config(checkpoint_dir: Path | None, config: TrainingConfig) -> Path | None:
    """Write the resolved training configuration into the run checkpoint directory."""
    if checkpoint_dir is None:
        return None

    config_path = checkpoint_dir / "config.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(asdict(config), handle, sort_keys=False)
    return config_path


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device | str,
    loss_fn: nn.Module | None = None,
) -> tuple[int, list[float], list[float]]:
    """Restore model/loss/optimizer state and return step plus loss histories.

    Iterable dataset stream position is not exactly restored during resume.
    Until a future dataset implementation supports deterministic seeking,
    for now restarting from a checkpoint restores model state, loss-function
    state, optimizer state, and global-step/loss-history bookkeeping.
    """
    checkpoint = torch.load(Path(path), map_location=device)
    _unwrap_parallel_module(model).load_state_dict(checkpoint["model_state_dict"])
    if loss_fn is not None:
        loss_state_dict = checkpoint.get("loss_state_dict")
        if loss_state_dict is None:
            LOGGER.warning("Checkpoint does not contain loss_state_dict; using the initialized loss function state.")
        else:
            _unwrap_parallel_module(loss_fn).load_state_dict(loss_state_dict)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    LOGGER.info(f"Loaded checkpoint from {path} with step {checkpoint['step']}")
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

    def _print_rank_zero(model, rows):
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        global_rank = int(os.environ.get("RANK", "0"))
        if local_rank == 0 and global_rank == 0:
            _print_model_specification_table(model, rows)
            print()

    iterator = iter(dataloader)
    try:
        first_batch = next(iterator)
    except StopIteration:
        LOGGER.warning("Skipping model parameter table because the training dataloader produced no batches.")
        _print_rank_zero(model, [])
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

    _print_rank_zero(model, rows)
    return itertools.chain([first_batch], iterator)



def safe_name(name):
    """
    W&B metric names are easier to work with when they avoid
    dots and unusual characters.
    """
    return name.replace(".", "_").replace("/", "_")


def log_selected_gradient_histograms(model):

    logs = {}

    for name, p in model.named_parameters():
        if p.grad is None:
            continue

        # Example: only log attention and head gradients
        if "attn" not in name and "head" not in name:
            continue

        g = p.grad.detach().float().cpu().flatten()

        if torch.isfinite(g).all():
            logs[f"grad_hist_{safe_name(name)}"] = wandb.Histogram(g.numpy())

    return logs

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

    LOGGER.debug('Clipping gradients')
    clip_grad_norm_(itertools.chain(model.parameters(), loss_fn.parameters()), 1.0)

    LOGGER.debug('Running optimizer step')
    optimizer.step()
    return float(loss.detach().cpu())


def train(
    config_or_path: str | Path | Mapping[str, Any] | TrainingConfig,
    model: nn.Module | None = None,
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    """Run a compact train/evaluate/checkpoint loop.

    Resuming from a checkpoint restores the model, loss function, optimizer,
    and global-step state before the main loop starts.  Iterable dataset stream
    position is not exactly restored unless a future dataset implementation
    supports deterministic seeking; for now, restarts resume model/loss/
    optimizer/global-step state rather than seeking to the prior stream item.
    """
    global NestDownsampler, OntheflyPhysicsModelLinear, OntheflyPipeline, build_loss, build_model

    if OntheflyPhysicsModelLinear is None:
        from msfm.onthefly_physics.onthefly_linear import OntheflyPhysicsModelLinear as _OntheflyPhysicsModelLinear

        OntheflyPhysicsModelLinear = _OntheflyPhysicsModelLinear
    if OntheflyPipeline is None:
        from msfm.onthefly_pipeline import OntheflyPipeline as _OntheflyPipeline

        OntheflyPipeline = _OntheflyPipeline
    if build_loss is None:
        from .losses.builder import build_loss as _build_loss

        build_loss = _build_loss
    if build_model is None:
        from .networks.builder import build_model as _build_model

        build_model = _build_model
    if NestDownsampler is None:
        from .networks.smoothing import NestDownsampler as _NestDownsampler

        NestDownsampler = _NestDownsampler

    config = _coerce_config(config_or_path)
    ddp_enabled, rank, world_size, local_rank, device = _setup_ddp(config, device)
    LOGGER.info(f"CUDA available: {torch.cuda.is_available()}")
    LOGGER.info(f"CUDA device count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        LOGGER.info(f"Device {i}: {torch.cuda.get_device_name(i)}")
    
    nside_training = 1024

    LOGGER.info(f"\n\nTag: {config.tag}\n")
    LOGGER.info(f"Training on {device} with config: {config}")
    if ddp_enabled:
        LOGGER.info(f"DDP enabled: rank={rank} local_rank={local_rank} world_size={world_size}")

    # 
    # Data loaders
    #

    # use non-reproducible seed, TODO: fix to reproducible
    seed = int(time.time())
    physics_model = OntheflyPhysicsModelLinear(config.forward_model, 
                        scalers=True,
                        device=device,
                        seed=seed).to(device)

    # Downsample all maps to the same nside
    # smoothing_model = NestDownsampler(nside=config.forward_model["analysis"]["n_side"], 
    #                         nside_base=config.forward_model["analysis"]["n_side_down"], 
    #                         nside_lower=nside_training, 
    #                         operator="mean").to(device)
    
    # Downsample each channel to a different nside
    # smoothing_model = NestChannelDownsampler(nside=config.forward_model["analysis"]["n_side"], 
    #                     nside_base=config.forward_model["analysis"]["n_side_down"], 
    #                     nside_lower=[nside_training]*24, 
    #                     operator="mean").to(device)

    def get_loaders(**kwargs):
        loader_training = OntheflyPipeline(**kwargs)
        loader_validation = OntheflyPipeline(**kwargs, validation=True)
        return loader_training, loader_validation

                            
    loader_training, loader_validation = get_loaders(webds_pattern=config.records_pattern, 
                                                     batch_size=config.batch_size, 
                                                     physics_model=physics_model, 
                                                     smoothing_model=None, 
                                                     num_workers=config.num_workers, 
                                                     prefetch_factor=1, 
                                                     device=device)

    # 
    # Build neural network
    #
 
    model = build_model(config.model_name, 
                    num_channels=physics_model.num_channels,
                    num_targets=physics_model.num_targets,
                    num_pixels=loader_training.num_pixels,
                    nside=nside_training,
                    nside_down=int(config.forward_model["analysis"]["n_side_down"]),
                    model_args=config.model_args,
                    )
    model.to(device)
    if ddp_enabled:
        ddp_kwargs = {"device_ids": [local_rank], "output_device": local_rank} if device.type == "cuda" else {}
        model = DDP(model, **ddp_kwargs)

    LOGGER.info(f'Model: {config.model_name}\n' + str(_unwrap_parallel_module(model)) + '\n')    

    # Loss function 
    loss_fn = build_loss(config.loss_function, num_targets=physics_model.num_targets)
    loss_fn = loss_fn.to(device)
    if ddp_enabled and any(parameter.requires_grad for parameter in loss_fn.parameters()):
        ddp_kwargs = {"device_ids": [local_rank], "output_device": local_rank} if device.type == "cuda" else {}
        loss_fn = DDP(loss_fn, **ddp_kwargs)
    LOGGER.info(f'Loss function: {_unwrap_parallel_module(loss_fn)}')

    # 
    # Build optimizer
    #

    # Optimizer
    trainable_parameters = itertools.chain(model.parameters(), loss_fn.parameters())
    optimizer = torch.optim.Adam(trainable_parameters, lr=config.learning_rate)
    step = 0
    session_step = 0
    train_losses: list[float] = []
    validation_losses: list[float] = []
    LOGGER.info('Optimizer:\n' + str(optimizer) + '\n')

    # Checkpoints
    checkpoint_wandb_info = None
    if config.resume_from_checkpoint:
        
        # if checkpoint does not exist, set resume_from_checkpoint to None and start the run from scratch
        if not Path(config.resume_from_checkpoint).exists():
            LOGGER.warning(f"Checkpoint file {config.resume_from_checkpoint} does not exist")
            config.resume_from_checkpoint = None
        
        else:
            step, train_losses, validation_losses = load_checkpoint(
                config.resume_from_checkpoint,
                model,
                optimizer,
                device,
                loss_fn,
            )
            checkpoint_wandb_info = _wandb_info_from_checkpoint(config.resume_from_checkpoint)

    checkpoint_dir = _prepare_checkpoint_dir(config) if _is_main_process() else None
    _write_reproducibility_config(checkpoint_dir, config)
    _ddp_barrier()

    # 
    # Training loop
    #

    # Housekeeping
    # loader_prefetcher = CUDAPrefetcher(loader, device=device)
    training_batches = _print_initial_model_summary(model, loader_training, device)
    run = init_wandb(config, checkpoint_wandb_info) if _is_main_process() else None
    active_wandb_info = _wandb_info_from_run(run) or checkpoint_wandb_info
    train_examples_seen = 0
    train_timer = Timer()
    # grad_scaler = torch.amp.GradScaler("cuda")

    # Training loop.
    LOGGER.info(f'Training loop starting with num_epochs={config.num_epochs}')
    for _epoch in range(config.num_epochs or 10**12):

        epoch_batches = training_batches if _epoch == 0 else loader_training
        # with DeviceTraceMode(only_cpu=True):
        with torch.profiler.record_function("training_loop"):

            LOGGER.timer.start("10steps")
            train_timer.start()   
            for batch in epoch_batches:

                

            

            # overfit on a single batch
            # batch = next(iter(epoch_batches))
            # for _ in range(10000):

                step += 1
                session_step += 1
                LOGGER.debug(f"====================================== step {step}")


                # LOGGER.warning('continuing training loop')
                # continue


                prev_t = time.perf_counter()
                prev_read, prev_write = tree_io_counters()

                #
                # Main magic - update model
                #

                model.train()
                maps, labels = batch
                LOGGER.debug(f'Maps shape={maps.shape} size={maps.numel()*maps.itemsize/1024**2:.2f} MB')
                LOGGER.debug(f'Labels shape={labels.shape}')
                maps = maps.to(device=device, dtype=torch.float32)
                labels = labels.to(device=device, dtype=torch.float32)
                optimizer.zero_grad(set_to_none=True)
                LOGGER.debug('Running forward pass')
                # with torch.autocast("cuda", dtype=torch.bfloat16):
                predictions = model(maps)
                LOGGER.debug('Running loss')
                train_loss = loss_fn(predictions, labels)
                # train_loss = grad_scaler.scale(train_loss)

                if not train_loss.requires_grad:
                    raise RuntimeError(
                        "The training loss is detached from the model parameters; "
                        "check the model forward pass for torch.no_grad(), detach(), or non-PyTorch conversions."
                    )

                LOGGER.debug('Running backward pass')
                train_loss.backward()

                LOGGER.debug('Clipping gradients')
                clip_grad_norm_(itertools.chain(model.parameters(), loss_fn.parameters()), 1.0)

                LOGGER.debug('Running optimizer step')
                # grad_scaler.step(optimizer)
                # grad_scaler.update()
                optimizer.step()

                global_train_loss = reduce_mean(train_loss)
                

                # LOGGER.warning('Skipping step housekeeping due to DeviceTraceMode')
                # continue
                
                train_loss = train_loss.detach().cpu()

                #
                # Step housekeeping
                # 

                # every step housekeeping
                LOGGER.debug('Running housekeeping')
                train_losses.append(train_loss)
                maps, _labels = batch
                train_examples_seen += int(maps.shape[0]) if hasattr(maps, "shape") and maps.ndim > 0 else config.batch_size
                train_timer.stop()
                
                current_learning_rate = optimizer.param_groups[0]["lr"]
                if run is not None:

                    # IO statistics    
                    # This should be at the end of the step to not dilute the timing/rates, but the difference should be negligible.
                    now_t = time.perf_counter()
                    now_read, now_write = tree_io_counters()
                    dt = now_t - prev_t
                    d_read = now_read - prev_read
                    d_write = now_write - prev_write

                    # adjust for DDP

                    wandb.log(
                        {
                            "Train/loss": global_train_loss.item(),
                            "step": step,
                            "learning_rate": current_learning_rate,
                            **get_examples_stats(train_examples_seen, train_timer.elapsed()),
                            **get_io_stats(d_read, d_write, dt),
                            **get_tensor_stats(maps, "maps"),
                            **get_tensor_stats(labels, "labels"),
                        },
                        step=step,
                    )

                # warm-up checks
                if step < 10:

                    _validate_gradient_flow(model)

                # frequent metrics
                if step % 10 == 0:

                    LOGGER.info(
                        f'Train loss epoch={_epoch:>3d} step={step:>5d} '
                        f'loss={train_loss: .8e} time_elapsed={LOGGER.timer.elapsed("10steps")}')
                    LOGGER.timer.reset("10steps")
                    if run is not None:
                        train_loss_components = loss_fn.loss_components(predictions, labels) if hasattr(loss_fn, "loss_components") else {}
                        for key, value in train_loss_components.items():
                            value = float(value.detach().cpu())
                            wandb.log({f"Train/loss_component/{key}": value}, step=step)
                
                # infrequent metrics
                if step % 100 == 0:

                    if run is not None:
                        LOGGER.debug('Running gradient logging')
                        grad_logs = get_gradient_stats(model, log_per_parameter=False)
                        grad_hist = log_selected_gradient_histograms(model)
                        wandb.log({**grad_logs, **grad_hist}, step=step)

                # very infrequent, checkpoint management
                if _is_main_process() and config.checkpoint_dir and config.checkpoint_every_steps and step % config.checkpoint_every_steps == 0:
                    checkpoint_path = _latest_checkpoint_path(checkpoint_dir)
                    save_checkpoint(
                        checkpoint_path,
                        model,
                        optimizer,
                        step,
                        config,
                        train_losses,
                        validation_losses,
                        loss_fn,
                        active_wandb_info,
                    )

                    if run is not None:
                        wandb.log(
                            {"Checkpoint/saved": 1, "Checkpoint/path": str(checkpoint_path), "step": step},
                            step=step,
                        )

                if config.max_steps is not None and session_step >= config.max_steps:
                    LOGGER.debug('Breaking training loop due to max steps')
                    break

                train_timer.start()
                LOGGER.debug('End of step')


            #
            # Validatio after each epoch
            #

            validation_predictions_path = _evaluation_predictions_path(config, _epoch) if _is_main_process() else None
            validation_loss = evaluate(model, loader_validation, loss_fn, device, num_examples=2000, predictions_path=validation_predictions_path)
            if validation_loss is not None:
                validation_losses.append(validation_loss)
                if run is not None:
                    wandb.log(
                        {
                            "Validation/loss": validation_loss,
                            "step": step,
                            "learning_rate": optimizer.param_groups[0]["lr"],
                        },
                        step=step,
                    )

            if config.max_steps is not None and session_step >= config.max_steps:
                break

        LOGGER.info(f'Epoch {_epoch} completed')

    LOGGER.info(f'Training completed with {step} steps')

    # save the latest checkpoint after all epochs/steps complete
    if _is_main_process() and checkpoint_dir:
        latest_checkpoint_path = _latest_checkpoint_path(checkpoint_dir)
        save_checkpoint(
            latest_checkpoint_path,
            model,
            optimizer,
            step,
            config,
            train_losses,
            validation_losses,
            loss_fn,
            active_wandb_info,
        )
        if run is not None:
            wandb.log(
                {"Checkpoint/saved": 1, "Checkpoint/path": str(latest_checkpoint_path), "step": step},
                step=step,
            )

    # finish wandb run
    if run is not None:
        run.finish()
    _ddp_barrier()

    # destroy process group
    if ddp_enabled and dist.is_initialized():
        dist.destroy_process_group()

    return {"model": model, "step": step, "train_losses": train_losses, "validation_losses": validation_losses}


def train_from_config(
    config_path: str | Path,
    *,
    resume_from_checkpoint: str | None = None,
    checkpoint_dir: str | None = None,
    max_steps: int | None = None,
    device: torch.device | str | None = None,
    wandb_mode: str | None = None,
    tag: str | None = None,
) -> dict[str, Any]:
    """Train from a YAML config file with optional CLI-style overrides."""
    config_path = Path(config_path)
    raw_config = with_forward_model_config(load_config(config_path), config_path.parent)
    overrides = {
        "resume_from_checkpoint": resume_from_checkpoint,
        "checkpoint_dir": checkpoint_dir,
        "max_steps": max_steps,
        "wandb_mode": wandb_mode,
        "tag": tag,
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


#
# Tracing of tensor placement
#

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from collections.abc import Mapping, Sequence

def tree_tensors(x):
    if torch.is_tensor(x):
        yield x
    elif isinstance(x, Mapping):
        for v in x.values():
            yield from tree_tensors(v)
    elif isinstance(x, Sequence) and not isinstance(x, (str, bytes)):
        for v in x:
            yield from tree_tensors(v)

def tensor_sig(t):
    return f"{tuple(t.shape)} {t.dtype} {t.device}"

class DeviceTraceMode(TorchDispatchMode):
    def __init__(self, only_cpu=True, max_lines=5000):
        super().__init__()
        self.only_cpu = only_cpu
        self.max_lines = max_lines
        self.lines = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        in_tensors = list(tree_tensors((args, kwargs)))
        result = func(*args, **kwargs)
        out_tensors = list(tree_tensors(result))

        devices = {str(t.device) for t in in_tensors + out_tensors}

        should_log = True
        if self.only_cpu:
            should_log = any(d == "cpu" for d in devices)

        if should_log and self.lines < self.max_lines:
            print(f"\n{func}")
            if in_tensors:
                print("  in :", [tensor_sig(t) for t in in_tensors])
            if out_tensors:
                print("  out:", [tensor_sig(t) for t in out_tensors])
            self.lines += 1

        return result


#
# Prefetcher data loader
#

import torch
from collections.abc import Mapping, Sequence


def iter_tensors(x):
    """Yield all tensors inside a nested batch structure."""
    if torch.is_tensor(x):
        yield x
    elif isinstance(x, Mapping):
        for v in x.values():
            yield from iter_tensors(v)
    elif isinstance(x, Sequence) and not isinstance(x, (str, bytes)):
        for v in x:
            yield from iter_tensors(v)


def move_to_device(x, device):
    """Recursively move a nested batch structure to device."""
    if torch.is_tensor(x):
        return x.to(device, non_blocking=True)
    elif isinstance(x, Mapping):
        return type(x)({k: move_to_device(v, device) for k, v in x.items()})
    elif isinstance(x, tuple) and hasattr(x, "_fields"):  # namedtuple
        return type(x)(*(move_to_device(v, device) for v in x))
    elif isinstance(x, tuple):
        return tuple(move_to_device(v, device) for v in x)
    elif isinstance(x, list):
        return [move_to_device(v, device) for v in x]
    else:
        return x


class CUDAPrefetcher:
    """
    Wraps a DataLoader and asynchronously preloads the next batch onto CUDA.

    Usage:
        loader = DataLoader(..., pin_memory=True)
        loader = CUDAPrefetcher(loader, device="cuda:0")

        for batch in loader:
            loss = train_step(batch)
    """

    def __init__(self, loader, device="cuda"):
        self.loader = loader
        self.device = torch.device(device)

        if self.device.type != "cuda":
            raise ValueError(f"CUDAPrefetcher requires a CUDA device, got {self.device}")

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        return _CUDAPrefetcherIterator(self.loader, self.device)


class _CUDAPrefetcherIterator:
    def __init__(self, loader, device):
        self.loader_iter = iter(loader)
        self.device = device
        self.stream = torch.cuda.Stream(device=device)
        self.next_batch = None
        self._done = False

        self._preload()

    def __iter__(self):
        return self

    def __next__(self):
        if self._done and self.next_batch is None:
            raise StopIteration

        # Wait until the side-stream H2D copy for next_batch is complete.
        torch.cuda.current_stream(self.device).wait_stream(self.stream)

        batch = self.next_batch

        if batch is None:
            raise StopIteration

        # Tell the caching allocator that these tensors are used on the
        # current stream too, not only on the prefetch stream.
        for t in iter_tensors(batch):
            if t.device.type == "cuda":
                t.record_stream(torch.cuda.current_stream(self.device))

        # Start copying the following batch while the caller computes on this one.
        self._preload()

        return batch

    def _preload(self):
        try:
            batch = next(self.loader_iter)
        except StopIteration:
            self.next_batch = None
            self._done = True
            return

        with torch.cuda.stream(self.stream):
            self.next_batch = move_to_device(batch, self.device)

#
# IO helpers
#

import os
import time
import psutil

def tree_io_counters(root_pid=None):
    root = psutil.Process(root_pid or os.getpid())
    procs = [root] + root.children(recursive=True)

    read_bytes = 0
    write_bytes = 0

    for p in procs:
        try:
            io = p.io_counters()
            read_bytes += io.read_bytes
            write_bytes += io.write_bytes
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    return read_bytes, write_bytes


#
# Logging helpers
#

def get_tensor_stats(x, name: str):

    return {
        f"Batch/{name}/mean": x.mean().detach().cpu(),
        f"Batch/{name}/std": x.std().detach().cpu(),
        f"Batch/{name}/min": x.min().detach().cpu(),
        f"Batch/{name}/max": x.max().detach().cpu(),
    }

def get_examples_stats(examples_seen, dt):
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    return {
        "Runtime/examples_per_second": examples_seen / dt * world_size,
    }

def get_io_stats(d_read, d_write, dt):

    world_size = dist.get_world_size() if dist.is_initialized() else 1
    return {
        "Proc_tree_io/read_MB_s": d_read / dt / 1e6 * world_size,
        "Proc_tree_io/write_MB_s": d_write / dt / 1e6 * world_size,
        "Proc_tree_io/read_MB": d_read / 1e6 * world_size,
        "Proc_tree_io/write_MB": d_write / 1e6 * world_size,
    }

#
# Timer
# 

class Timer:

    def __init__(self):
        self.start_time = 0
        self.elapsed_time = 0
        self.running = False

    def start(self):
        if self.running:
            pass
        else:
            self.running = True
            self.start_time = time.perf_counter()

    def stop(self):
        if not self.running:
            pass
        else:
            self.elapsed_time += time.perf_counter() - self.start_time
            self.running = False
        
    def elapsed(self):
        if self.running:
            self.elapsed_time += time.perf_counter() - self.start_time
        return self.elapsed_time

    def __str__(self):
        return f'Timer(elapsed={self.elapsed_time:.2f}s)'


#
# DDP helpers
# 

def reduce_mean(x: torch.Tensor) -> torch.Tensor:

    x = x.detach().clone()
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    x /= dist.get_world_size()

    return x
