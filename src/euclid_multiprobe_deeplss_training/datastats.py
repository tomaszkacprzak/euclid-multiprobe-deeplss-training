"""Dataset statistics command for DeepLSS training records."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import time
import numpy as np
import torch
from msfm.onthefly_physics.onthefly_linear import OntheflyPhysicsModelLinear
from msfm.onthefly_pipeline import OntheflyPipeline

from euclid_multiprobe_deeplss_training.networks.smoothing import NestChannelDownsampler

from .training import TrainingConfig
from .utils.config import load_config, with_forward_model_config
from .utils.logger import get_logger

LOGGER = get_logger(__file__)

@dataclass(slots=True)
class BatchChannelStats:
    """Per-feature summary values for one input batch."""

    split: str
    count: torch.Tensor
    minimum: torch.Tensor
    maximum: torch.Tensor
    total: torch.Tensor
    sum_squares: torch.Tensor


def datastats(config_or_path: str | Path | Mapping[str, Any] | TrainingConfig) -> list[BatchChannelStats]:
    """Print per-channel input-map and per-label statistics for one full training epoch."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    LOGGER.info(f'Using device: {device}')
    config = _coerce_config(config_or_path)
    
    physics_model = OntheflyPhysicsModelLinear(config.forward_model, 
                        scalers=True,
                        device=device).to(device)

    smoothing_model = NestChannelDownsampler(nside=config.forward_model["analysis"]["n_side"], 
                        nside_base=config.forward_model["analysis"]["n_side_down"], 
                        nside_lower=[512]*24, 
                        operator="mean").to(device)
    
    loader = OntheflyPipeline(config.records_pattern, 
                              config.batch_size, 
                              physics_model, 
                              smoothing_model,
                              num_workers=config.num_workers,
                              device=device)

    batch_stats: list[BatchChannelStats] = []
    batch_label_stats: list[BatchChannelStats] = []
    LOGGER.info('Collecting stats for training loader')
    map_stats, label_stats = _collect_loader_stats(loader, split="train")
    batch_stats.extend(map_stats)
    batch_label_stats.extend(label_stats)
    print("Map channel statistics:")
    _print_feature_stats(_combine_batch_stats(batch_stats), feature_name="channel")
    print("Label statistics:")
    _print_feature_stats(_combine_batch_stats(batch_label_stats), feature_name="label")

    return batch_stats


def datastats_from_config(config_path: str | Path) -> list[BatchChannelStats]:
    """Run datastats from a YAML config file."""
    config_path = Path(config_path)
    raw_config = with_forward_model_config(load_config(config_path), config_path.parent)
    return datastats(raw_config)


def _collect_loader_stats(dataloader: Iterable, *, split: str) -> tuple[list[BatchChannelStats], list[BatchChannelStats]]:
    map_stats = []
    label_stats = []
    batch_count = 0

    from torch.profiler import ProfilerActivity, profile, schedule, tensorboard_trace_handler

    data_iter = iter(dataloader)
    try:
        maps, labels = next(data_iter)
    except StopIteration:
        return map_stats, label_stats

    LOGGER.debug(f'Batch {batch_count} maps shape={maps.shape} size={maps.numel()*maps.itemsize/1024**2:.2f} MB')
    LOGGER.debug(f'Batch {batch_count} labels shape={labels.shape} size={labels.numel()*labels.itemsize/1024**2:.2f} MB')

    map_stats.append(_summarize_maps(maps, split=split))
    label_stats.append(_summarize_labels(labels, split=split))
    batch_count += 1

    print('maps.device =', maps.device)
    print('labels.device =', labels.device)

    fname = 'maps_batch_1.npy'
    np.save(fname, maps.detach().cpu().numpy())
    LOGGER.info(f'Saved maps to {fname}')

    prof_schedule = schedule(
        wait=2,
        warmup=2,
        active=5,
        repeat=1,
    )

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

        time_start = time.time()
        for maps, labels in data_iter:

            LOGGER.debug(f'Batch {batch_count:>4d} maps shape={maps.shape} size={maps.numel()*maps.itemsize/1024**2:.2f} MB')
            LOGGER.debug(f'Batch {batch_count:>4d} labels shape={labels.shape} size={labels.numel()*labels.itemsize/1024**2:.2f} MB')

            if batch_count % 10 == 0:
                
                time_diff = time_end = time.time() - time_start
                LOGGER.info(f'Batch {batch_count:>4d}, num_examples_per_second: {batch_count * dataloader.batch_size / time_diff:.2f}')

            map_stats.append(_summarize_maps(maps, split=split))
            label_stats.append(_summarize_labels(labels, split=split))
            batch_count += 1

            if maps.is_cuda:
                torch.cuda.synchronize()
            prof.step()

    print_profiler_stats(prof)
    
    return map_stats, label_stats

    
def print_profiler_stats(prof: Any, num_rows=20):

    events = prof.key_averages()

    def ms(us):
        return us / 1000.0


    def event_attr(evt, name, default=0):
        return getattr(evt, name, default)

    events = list(prof.key_averages())

    events = sorted(
        events,
        key=lambda e: event_attr(e, "self_device_time_total", 0),
        reverse=True,
    )

    total_self_device = sum(
        event_attr(e, "self_device_time_total", 0)
        for e in events
    )

    for i, evt in enumerate(events[:num_rows], start=1):
        self_device = event_attr(evt, "self_device_time_total", 0)
        device_total = event_attr(evt, "device_time_total", 0)

        pct = (
            100.0 * self_device / total_self_device
            if total_self_device > 0
            else 0.0
        )

        

        print(f"\n================================================================ Event #{i}")
        print(f"Name: {event_attr(evt, 'key', '<unknown>')}")
        print()
        print(f"Self device: {ms(self_device):.3f} ms")
        print(f"Self device %: {pct:.2f}%")
        print(f"Device total: {ms(device_total):.3f} ms")
        print(f"CPU total: {ms(event_attr(evt, 'cpu_time_total', 0)):.3f} ms")
        print(f"Self CPU: {ms(event_attr(evt, 'self_cpu_time_total', 0)):.3f} ms")
        print(f"Calls: {event_attr(evt, 'count', 0)}")
        print(f"CPU mem: {event_attr(evt, 'cpu_memory_usage', 0) / 1024**2:.2f} MB")
        print(f"Self CPU mem: {event_attr(evt, 'self_cpu_memory_usage', 0) / 1024**2:.2f} MB")
        print(f"Device mem: {event_attr(evt, 'device_memory_usage', 0) / 1024**2:.2f} MB")
        print(f"Self device mem: {event_attr(evt, 'self_device_memory_usage', 0) / 1024**2:.2f} MB")

    print()
    

def _summarize_maps(maps: torch.Tensor, *, split: str) -> BatchChannelStats:

    channel_values = _flatten_by_channel(maps)
    return _summarize_feature_values(channel_values, split=split)


def _summarize_labels(labels: torch.Tensor, *, split: str) -> BatchChannelStats:

    label_values = _flatten_by_final_dimension(labels)
    return _summarize_feature_values(label_values, split=split)


def _summarize_feature_values(feature_values: torch.Tensor, *, split: str) -> BatchChannelStats:
    if feature_values.shape[1] == 0:
        raise ValueError("Cannot calculate statistics for empty feature vectors.")

    values = feature_values.to(torch.float64)
    feature_min = values.min(dim=1).values.detach().cpu()
    feature_max = values.max(dim=1).values.detach().cpu()
    feature_total = values.sum(dim=1).detach().cpu()
    feature_sum_squares = torch.sum(values * values, dim=1).detach().cpu()
    feature_count = torch.full((values.shape[0],), values.shape[1], dtype=torch.float64)

    return BatchChannelStats(
        split=split,
        count=feature_count,
        minimum=feature_min,
        maximum=feature_max,
        total=feature_total,
        sum_squares=feature_sum_squares,
    )


def _flatten_by_final_dimension(values: torch.Tensor) -> torch.Tensor:

    assert values.ndim >= 1, f'Unsupported number of dimensions: {values.ndim}'
    features_first = values.movedim(-1, 0)
    return features_first.reshape(features_first.shape[0], -1)


def _flatten_by_channel(maps: torch.Tensor) -> torch.Tensor:

    assert maps.ndim == 3, f'Unsupported number of dimensions: {maps.ndim}'
    return _flatten_by_final_dimension(maps)


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


def _print_feature_stats(stats: BatchChannelStats | None, *, feature_name: str) -> None:
    if stats is None:
        print("No input batches found.")
        return

    mean = stats.total / stats.count
    variance = (stats.sum_squares / stats.count) - (mean * mean)
    std = torch.sqrt(torch.clamp(variance, min=0.0))

    for feature in range(stats.count.numel()):
        print(
            f"{feature_name}={feature:>3d}, "
            f"min ={stats.minimum[feature].item(): 10.10e}, "
            f"max ={stats.maximum[feature].item(): 10.10e}, "
            f"mean={mean[feature].item(): 10.10e}, "
            f"std ={std[feature].item(): 10.10e}"
        )


def _print_channel_stats(stats: BatchChannelStats | None) -> None:
    _print_feature_stats(stats, feature_name="channel")


def _coerce_config(config_or_path: str | Path | Mapping[str, Any] | TrainingConfig) -> TrainingConfig:
    if isinstance(config_or_path, TrainingConfig):
        return config_or_path
    if isinstance(config_or_path, str | Path):
        config_path = Path(config_or_path)
        return TrainingConfig.from_mapping(with_forward_model_config(load_config(config_path), config_path.parent))
    return TrainingConfig.from_mapping(with_forward_model_config(config_or_path))
