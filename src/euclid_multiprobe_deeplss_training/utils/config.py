"""Configuration objects and shared configuration loading helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
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

    forward_model: dict[str, Any] = field(default_factory=dict)
    training: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, raw_config: Mapping[str, Any]) -> Config:
        """Create a config from the two top-level mappings.

        Empty sections are allowed for lightweight and legacy programmatic
        callers; individual workflows validate the settings they require.
        """
        training = raw_config.get("training", {})
        if not isinstance(training, Mapping):
            raise TypeError("The 'training' configuration section must be a mapping.")
        forward_model = raw_config.get("forward_model", {})
        if not isinstance(forward_model, Mapping):
            raise TypeError("The 'forward_model' configuration section must be a mapping.")

        # Keep accepting the original flat training mapping for programmatic
        # callers. Values in the merged ``training`` section are authoritative.
        legacy = {
            key: value
            for key, value in raw_config.items()
            if key not in {"forward_model", "training"}
        }
        sections = {
            "forward_model": dict(forward_model),
            "training": _merge_mappings(legacy, training),
        }

        config = cls(**sections)
        config._validate()
        return config

    def _validate(self) -> None:
        training = self.training
        if "batch_size" in training and training["batch_size"] <= 0:
            raise ValueError("batch_size must be positive.")
        if {"num_epochs", "max_steps"} <= training.keys() and training["num_epochs"] is None and training["max_steps"] is None:
            raise ValueError("Set at least one of num_epochs or max_steps.")
        if "learning_rate" in training and training["learning_rate"] <= 0.0:
            raise ValueError("learning_rate must be positive.")
        if "num_workers" in training and training["num_workers"] < 0:
            raise ValueError("num_workers must be non-negative.")
        if "checkpoint_every_steps" in training and training["checkpoint_every_steps"] < 0:
            raise ValueError("checkpoint_every_steps must be non-negative.")
        if "encoder_args" in training and not isinstance(training["encoder_args"], Mapping):
            raise TypeError("encoder_args must be a mapping.")

    def __getattr__(self, name: str) -> Any:
        """Provide read-only compatibility with the former flat interface."""
        if name in self.training:
            return self.training[name]
        model = self.training.get("model", {})
        if isinstance(model, Mapping) and name in model:
            return model[name]
        raise AttributeError(name)


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
