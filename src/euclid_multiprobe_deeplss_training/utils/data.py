"""Shared dataset-loading helpers for CLI drivers."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from typing import Any

from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from msfm.onthefly_pipeline import OntheflyPipeline
from msfm.onthefly_physics.onthefly_base import OntheflyPhysicsModel
from msfm.onthefly_physics.onthefly_linear import OntheflyPhysicsModelLinear




def build_records_dataset(records_pattern: str, config: Mapping[str, Any]) -> IterableDataset:
    """Build the user-supplied iterable records dataset.

    Implement this project-specific hook to read ``records_pattern`` and yield
    samples shaped like ``(map_tensor, target_tensor)``.  ``map_tensor`` should
    be a part-sky Healpix map with shape similar to ``(channels, pixels)`` (or
    ``(pixels, channels)``, if that becomes the final convention).  The
    ``target_tensor`` should contain ``float32`` regression target data.
    """
    dataset = OntheflyPipeline(records_pattern).get_dataset()
    return dataset


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
    training_dataset = StreamSplitDataset(dataset, validation_fraction, seed, "train")
    validation_dataset = StreamSplitDataset(dataset, validation_fraction, seed, "validation")
    return training_dataset, validation_dataset


def make_dataloader(dataset: IterableDataset, config, *, drop_last: bool | None = None) -> DataLoader:
    """Create a worker-sharded DataLoader for an iterable dataset."""
    return DataLoader(
        WorkerShardDataset(dataset),
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        drop_last=config.drop_last if drop_last is None else drop_last,
    )

def make_physics_dataloader(loader: IterableDataset, config: Mapping[str, Any], seed_offset: int = 0, device: torch.device | str | None = None) -> OntheflyPhysicsModel:
    """Build the physics loader from the given loader and config."""

    model = OntheflyPhysicsModelLinear(loader, config.forward_model, seed_offset=seed_offset, device=device)
    return model
