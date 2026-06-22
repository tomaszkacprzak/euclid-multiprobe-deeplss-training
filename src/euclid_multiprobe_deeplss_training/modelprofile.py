"""Transformer forward-pass profiling command for DeepLSS training records."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any
import healpy as hp

import torch
from msfm.onthefly_physics.onthefly_linear import OntheflyPhysicsModelLinear
from msfm.onthefly_pipeline import OntheflyPipeline

from euclid_multiprobe_deeplss_training.networks.builder import build_model
from euclid_multiprobe_deeplss_training.networks.smoothing import HealpyDownsampling

from .datastats import print_profiler_stats
from .training import TrainingConfig
from .utils.config import load_config, with_forward_model_config
from .utils.logger import get_logger

LOGGER = get_logger(__file__)


def modelprofile(config_or_path: str | Path | Mapping[str, Any] | TrainingConfig) -> list[torch.Tensor]:
    """Profile untrained nested-transformer forward passes over pipeline batches."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    LOGGER.info(f"Using device: {device}")
    config = _coerce_config(config_or_path)

    physics_model = OntheflyPhysicsModelLinear(config.forward_model, device=device).to(device)
    smoothing_model = HealpyDownsampling(
        nside=config.forward_model["analysis"]["n_side"],
        nside_base=config.forward_model["analysis"]["n_side_down"],
        nside_lower=[512] * 24,
        operator="mean",
    ).to(device)

    loader = OntheflyPipeline(
        config.records_pattern,
        physics_model,
        smoothing_model,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=True,
        device=device,
    )

    model = build_model(
        "nested_transformer",
        num_channels=config.in_channels,
        num_targets=config.num_targets,
    ).to(device)
    model.eval()
    LOGGER.info(f"Profiling model: {model.__class__.__name__}")

    return _profile_loader_forward_passes(loader, model, config=config)


def modelprofile_from_config(config_path: str | Path) -> list[torch.Tensor]:
    """Run modelprofile from a YAML config file."""
    config_path = Path(config_path)
    raw_config = with_forward_model_config(load_config(config_path), config_path.parent)
    return modelprofile(raw_config)


def _profile_loader_forward_passes(
    dataloader: Iterable,
    model: torch.nn.Module,
    *,
    config: TrainingConfig,
) -> list[torch.Tensor]:
    from torch.profiler import ProfilerActivity, profile, schedule, tensorboard_trace_handler

    outputs: list[torch.Tensor] = []
    prof_schedule = schedule(wait=2, warmup=2, active=5, repeat=1)

    profiler_activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        profiler_activities.append(ProfilerActivity.CUDA)

    with profile(
        activities=profiler_activities,
        schedule=prof_schedule,
        on_trace_ready=tensorboard_trace_handler("./torch_profiler_logs"),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        with torch.no_grad():
            for batch_count, (maps, _labels) in enumerate(dataloader):
                transformer_batch = _prepare_transformer_batch(maps, config=config)
                LOGGER.debug(
                    "Batch %s transformer input shape=%s size=%.2f MB",
                    batch_count,
                    tuple(transformer_batch.shape),
                    transformer_batch.numel() * transformer_batch.itemsize / 1024**2,
                )
                output = model(transformer_batch)
                outputs.append(output.detach().cpu())

                if transformer_batch.is_cuda:
                    torch.cuda.synchronize()
                prof.step()

                if batch_count + 1 == 10:
                    break

    print_profiler_stats(prof)
    return outputs


def _prepare_transformer_batch(maps: torch.Tensor, *, config: TrainingConfig) -> torch.Tensor:
    """Convert pipeline maps shaped ``(B, P, C)`` to nested transformer input."""
    if maps.ndim != 3:
        raise ValueError(f"Expected maps with shape (batch, pixels, channels), got {tuple(maps.shape)}")

    batch_size, num_pixels, num_channels = maps.shape
    print(f"maps.shape: {maps.shape}")
    if num_channels != config.in_channels:
        raise ValueError(f"Expected {config.in_channels} input channels, got {num_channels}")

    num_top_level_tokens = _calculate_num_top_level_tokens(config, num_pixels)
    if num_pixels % num_top_level_tokens != 0:
        raise ValueError(f"Cannot split {num_pixels} pixels into {num_top_level_tokens} top-level tokens")

    nested_factor = num_pixels // num_top_level_tokens
    nested_levels = 0
    while nested_factor > 1 and nested_factor % 4 == 0:
        nested_levels += 1
        nested_factor //= 4

    if nested_factor != 1:
        raise ValueError(f"Pixel ratio {num_pixels // num_top_level_tokens} is not a power of 4")

    nested_shape = (4,) * nested_levels
    return maps.movedim(2, 1).contiguous().reshape(batch_size, num_channels, num_top_level_tokens, *nested_shape)


def _calculate_num_top_level_tokens(config: TrainingConfig, num_pixels: int) -> int:
    """Calculate the number of top-level tokens for the nested transformer."""
    """
    Calculate the number of top-level tokens for the nested transformer.

    Args:
        config: The training configuration.
        num_pixels: The number of pixels in the map.

    Returns:
        The number of top-level tokens.
    """

    analysis_config = config.forward_model.get("analysis", {}) if isinstance(config.forward_model, Mapping) else {}
    nside_down = analysis_config.get("n_side_down")
    nside = analysis_config.get("n_side")
    assert nside_down is not None
    assert nside is not None

    num_pixels_nside = hp.nside2npix(nside)
    num_pixels_nside_down = hp.nside2npix(nside_down)

    print("nside_down:", nside_down, "num_pixels:", num_pixels)
   

    num_pixels_per_top_level_token = num_pixels_nside //num_pixels_nside_down
    print("num_pixels_per_top_level_token:", num_pixels_per_top_level_token)

    num_top_level_tokens = num_pixels // num_pixels_per_top_level_token
    print("num_top_level_tokens:", num_top_level_tokens)
      

    return num_top_level_tokens


def _coerce_config(config_or_path: str | Path | Mapping[str, Any] | TrainingConfig) -> TrainingConfig:
    if isinstance(config_or_path, TrainingConfig):
        return config_or_path
    if isinstance(config_or_path, str | Path):
        config_path = Path(config_or_path)
        return TrainingConfig.from_mapping(with_forward_model_config(load_config(config_path), config_path.parent))
    return TrainingConfig.from_mapping(with_forward_model_config(config_or_path))
