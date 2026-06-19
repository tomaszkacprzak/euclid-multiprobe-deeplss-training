"""Dataset statistics command for DeepLSS training records."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .training import TrainingConfig
from .utils.config import load_config, with_forward_model_config
from .utils.data import build_records_dataset, make_dataloader, split_iterable_dataset, make_physics_dataloader
from .utils.logger import get_logger

LOGGER = get_logger(__file__)

@dataclass(slots=True)
class BatchChannelStats:
    """Per-channel summary values for one input batch."""

    split: str
    count: torch.Tensor
    minimum: torch.Tensor
    maximum: torch.Tensor
    total: torch.Tensor
    sum_squares: torch.Tensor


def datastats(config_or_path: str | Path | Mapping[str, Any] | TrainingConfig) -> list[BatchChannelStats]:
    """Print per-channel input-map statistics for one full train/validation epoch."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = _coerce_config(config_or_path)
    dataset = build_records_dataset(config.records_pattern, config.extra | {"config": config, "forward_model": config.forward_model})
    training_dataset, validation_dataset = split_iterable_dataset(dataset, config.validation_fraction, config.seed)
    training_loader = make_dataloader(training_dataset, config, drop_last=config.drop_last)
    validation_loader = make_dataloader(validation_dataset, config, drop_last=False)
    training_physics_loader = make_physics_dataloader(training_loader, config, device=device)
    validation_physics_loader = make_physics_dataloader(validation_loader, config, device=device)

    batch_stats: list[BatchChannelStats] = []
    batch_stats.extend(_collect_loader_stats(training_physics_loader, split="train"))
    batch_stats.extend(_collect_loader_stats(validation_physics_loader, split="validation"))

    _print_channel_stats(_combine_batch_stats(batch_stats))
    return batch_stats


def datastats_from_config(config_path: str | Path) -> list[BatchChannelStats]:
    """Run datastats from a YAML config file."""
    config_path = Path(config_path)
    raw_config = with_forward_model_config(load_config(config_path), config_path.parent)
    return datastats(raw_config)


def _collect_loader_stats(dataloader: Iterable, *, split: str) -> list[BatchChannelStats]:
    stats = []
    batch_count = 0
    for maps, _labels in dataloader:

        LOGGER.debug(f'Batch {batch_count} maps shape={maps.shape} size={maps.numel()*maps.itemsize/1024**2:.2f} MB')

        stats.append(_summarize_maps(maps, split=split))
        batch_count += 1
    return stats


def _summarize_maps(maps: torch.Tensor, *, split: str) -> BatchChannelStats:
    maps = torch.as_tensor(maps, dtype=torch.float64).detach().cpu()
    channel_values = _flatten_by_channel(maps)
    return BatchChannelStats(
        split=split,
        count=torch.full((channel_values.shape[0],), channel_values.shape[1], dtype=torch.float64),
        minimum=channel_values.min(dim=1).values,
        maximum=channel_values.max(dim=1).values,
        total=channel_values.sum(dim=1),
        sum_squares=(channel_values * channel_values).sum(dim=1),
    )


def _flatten_by_channel(maps: torch.Tensor) -> torch.Tensor:
    if maps.ndim == 0:
        return maps.reshape(1, 1)
    if maps.ndim == 1:
        return maps.reshape(1, -1)
    if maps.ndim == 2:
        return maps.reshape(1, -1)

    channels_first = maps.movedim(1, 0)
    return channels_first.reshape(channels_first.shape[0], -1)


def _combine_batch_stats(batch_stats: list[BatchChannelStats]) -> BatchChannelStats | None:
    if not batch_stats:
        return None
    return BatchChannelStats(
        split="all",
        count=torch.stack([item.count for item in batch_stats]).sum(dim=0),
        minimum=torch.stack([item.minimum for item in batch_stats]).min(dim=0).values,
        maximum=torch.stack([item.maximum for item in batch_stats]).max(dim=0).values,
        total=torch.stack([item.total for item in batch_stats]).sum(dim=0),
        sum_squares=torch.stack([item.sum_squares for item in batch_stats]).sum(dim=0),
    )


def _print_channel_stats(stats: BatchChannelStats | None) -> None:
    if stats is None:
        print("No input batches found.")
        return

    mean = stats.total / stats.count
    variance = (stats.sum_squares / stats.count) - (mean * mean)
    std = torch.sqrt(torch.clamp(variance, min=0.0))

    print("channel,min,max,mean,std")
    for channel in range(stats.count.numel()):
        print(
            f"{channel},"
            f"{stats.minimum[channel].item():.10g},"
            f"{stats.maximum[channel].item():.10g},"
            f"{mean[channel].item():.10g},"
            f"{std[channel].item():.10g}"
        )


def _coerce_config(config_or_path: str | Path | Mapping[str, Any] | TrainingConfig) -> TrainingConfig:
    if isinstance(config_or_path, TrainingConfig):
        return config_or_path
    if isinstance(config_or_path, str | Path):
        config_path = Path(config_or_path)
        return TrainingConfig.from_mapping(with_forward_model_config(load_config(config_path), config_path.parent))
    return TrainingConfig.from_mapping(with_forward_model_config(config_or_path))
