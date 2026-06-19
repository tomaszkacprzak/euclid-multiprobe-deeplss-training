"""Dataset statistics command for DeepLSS training records."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import numpy as np

from .training import TrainingConfig
from .utils.config import load_config, with_forward_model_config
# from .utils.data import build_records_dataset, make_dataloader, split_iterable_dataset, make_physics_dataloader
from .utils.logger import get_logger

from msfm.onthefly_physics.onthefly_linear import OntheflyPhysicsModelLinear
from msfm.onthefly_pipeline import OntheflyPipeline


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
    LOGGER.info(f'Using device: {device}')
    config = _coerce_config(config_or_path)
    physics_model = OntheflyPhysicsModelLinear(config.forward_model, device=device).to(device)
    
    loader = OntheflyPipeline(config.records_pattern, 
                              physics_model, 
                              batch_size=config.batch_size, 
                              num_workers=config.num_workers,
                              pin_memory=True,
                              device=device)

    batch_stats: list[BatchChannelStats] = []
    LOGGER.info(f'Collecting stats for training loader')
    batch_stats.extend(_collect_loader_stats(loader, split="train"))
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

    from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler

    for maps, _labels in dataloader:

        LOGGER.debug(f'Batch {batch_count} maps shape={maps.shape} size={maps.numel()*maps.itemsize/1024**2:.2f} MB')

        stats.append(_summarize_maps(maps, split=split))
        batch_count += 1

        print('maps.device =', maps.device)
        print('_labels.device =', _labels.device)

        fname = 'maps_batch_1.npy'
        np.save(fname, maps.detach().cpu().numpy())
        LOGGER.info(f'Saved maps to {fname}')

        break



    prof_schedule = schedule(
        wait=2,
        warmup=2,
        active=5,
        repeat=1,
    )

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=prof_schedule,
        on_trace_ready=tensorboard_trace_handler("./torch_profiler_logs"),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:

        for maps, _labels in dataloader:

            LOGGER.debug(f'Batch {batch_count} maps shape={maps.shape} size={maps.numel()*maps.itemsize/1024**2:.2f} MB')

            stats.append(_summarize_maps(maps, split=split))
            batch_count += 1

            if batch_count == 10:
                break

            torch.cuda.synchronize()
            prof.step()

    print(prof.key_averages().table(sort_by='self_cuda_time_total', row_limit=100, max_name_column_width=200, max_src_column_width=30))
    return stats

    


def _summarize_maps(maps: torch.Tensor, *, split: str) -> BatchChannelStats:

    channel_values = _flatten_by_channel(maps)
    channel_min = channel_values.min(dim=1).values.detach().cpu()
    channel_max = channel_values.max(dim=1).values.detach().cpu()
    channel_total = channel_values.sum(dim=1).detach().cpu()
    channel_sum_squares = torch.sum(channel_values * channel_values, dim=1).detach().cpu()
    channel_count = torch.full((channel_values.shape[0],), channel_values.shape[1], dtype=torch.float32)

    return BatchChannelStats(
        split=split,
        count=channel_count,
        minimum=channel_min,
        maximum=channel_max,
        total=channel_total,
        sum_squares=channel_sum_squares,
    )


def _flatten_by_channel(maps: torch.Tensor) -> torch.Tensor:

    assert maps.ndim == 3, f'Unsupported number of dimensions: {maps.ndim}'
    channels_first = maps.movedim(2, 0)
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

    for channel in range(stats.count.numel()):
        print(
            f"channel={channel:>3d}, "
            f"min ={stats.minimum[channel].item(): 10.10e}, "
            f"max ={stats.maximum[channel].item(): 10.10e}, "
            f"mean={mean[channel].item(): 10.10e}, "
            f"std ={std[channel].item(): 10.10e}"
        )


def _coerce_config(config_or_path: str | Path | Mapping[str, Any] | TrainingConfig) -> TrainingConfig:
    if isinstance(config_or_path, TrainingConfig):
        return config_or_path
    if isinstance(config_or_path, str | Path):
        config_path = Path(config_or_path)
        return TrainingConfig.from_mapping(with_forward_model_config(load_config(config_path), config_path.parent))
    return TrainingConfig.from_mapping(with_forward_model_config(config_or_path))
