"""Configuration objects and shared configuration loading helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ConfigPath = str | Path
ConfigPaths = ConfigPath | Sequence[ConfigPath]


@dataclass(slots=True)
class Config:
    """The two sections of the application configuration file.

    Defaults belong in ``configs/example.yaml`` rather than in this Python
    representation.  Consequently, consumers read training options from the
    ``training`` mapping exactly as they were loaded from YAML.
    """

    forward_model: dict[str, Any]
    training: dict[str, Any]

    @classmethod
    def from_mapping(cls, raw_config: Mapping[str, Any]) -> Config:
        """Create a config from the two required top-level mappings."""
        sections: dict[str, dict[str, Any]] = {}
        for name in ("forward_model", "training"):
            value = raw_config.get(name)
            if not isinstance(value, Mapping):
                raise TypeError(f"The '{name}' configuration section must be a mapping.")
            sections[name] = dict(value)

        config = cls(**sections)
        config._validate()
        return config

    def _validate(self) -> None:
        training = self.training
        if training["batch_size"] <= 0:
            raise ValueError("batch_size must be positive.")
        if training.get("num_epochs") is None and training.get("max_steps") is None:
            raise ValueError("Set at least one of num_epochs or max_steps.")
        if training["learning_rate"] <= 0.0:
            raise ValueError("learning_rate must be positive.")
        if training["num_workers"] < 0:
            raise ValueError("num_workers must be non-negative.")
        if training["checkpoint_every_steps"] < 0:
            raise ValueError("checkpoint_every_steps must be non-negative.")
        if not isinstance(training["encoder_args"], Mapping):
            raise TypeError("encoder_args must be a mapping.")


def _merge_mappings(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base`` without mutating inputs."""
    merged = dict(base)
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            merged[key] = _merge_mappings(dict(existing), value)
        else:
            merged[key] = value
    return merged


def config_paths(paths: ConfigPaths) -> list[Path]:
    """Normalize one config path or a non-empty sequence of config paths."""
    if isinstance(paths, str | Path):
        return [Path(paths)]
    normalized = [Path(path) for path in paths]
    if not normalized:
        raise ValueError("At least one configuration file is required.")
    return normalized


def load_config(paths: ConfigPaths) -> dict[str, Any]:
    """Load and recursively merge one or more YAML configuration files.

    Files are applied from left to right; values from later files override
    earlier values while preserving unrelated keys in nested mappings.
    """
    merged: dict[str, Any] = {}
    for path in config_paths(paths):
        with path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        if not isinstance(loaded, dict):
            raise TypeError(f"The YAML config {path} must load to a mapping.")
        merged = _merge_mappings(merged, loaded)
    return merged


def load_pixel_indices(conf: dict):
    """Load the survey pixel indices from the configured HDF5 file."""
    import h5py

    with h5py.File(conf["files"]["pixels"], "r") as f:
        return f["data_vec"][:]
