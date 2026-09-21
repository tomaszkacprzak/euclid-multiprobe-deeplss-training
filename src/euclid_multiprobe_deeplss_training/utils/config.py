"""Configuration objects and shared configuration loading helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

ConfigPath = str | Path
ConfigPaths = ConfigPath | Sequence[ConfigPath]


@dataclass(slots=True)
class Config:
    """Normalized application configuration.

    Training options are read from the master configuration's ``training``
    section. Flat keys remain supported for programmatic callers, while keys
    in ``training`` take precedence when both forms are present. The optional
    ``model`` and ``physics_model_args`` subsections are also supported. The
    top-level ``forward_model`` section is retained as part of the normalized
    configuration.
    """

    records_pattern: str = ""
    model_name: str = "nested_transformer"
    model_args: dict[str, Any] = field(default_factory=dict)
    sweep: list[Any] = field(default_factory=list)
    encoder_name: str = "nested_transformer"
    encoder_args: dict[str, Any] = field(default_factory=dict)
    embed_dim: int = 64
    forward_model: dict[str, Any] = field(default_factory=dict)
    physics_model: str = "onthefly_linear"
    physics_model_args: dict[str, Any] = field(default_factory=dict)
    loss_function: str = "mse"
    loss_args: dict[str, Any] = field(default_factory=dict)
    batch_size: int = 32
    num_epochs: int | None = 1
    max_steps: int | None = None
    learning_rate: float = 1.0e-3
    grad_clip_max_norm: float = 1.0
    num_workers: int = 1
    checkpoint_dir: str | None = None
    checkpoint_every_steps: int = 0
    validation_every_steps: int | None = None
    num_validation_examples: int = 1000
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
    def from_mapping(cls, raw_config: Mapping[str, Any]) -> Config:
        """Create a validated config object from merged YAML data."""
        training_config = raw_config.get("training", {})
        if training_config is None:
            training_config = {}
        if not isinstance(training_config, Mapping):
            raise TypeError("The 'training' configuration section must be a mapping.")

        # Keep accepting flat mappings for direct Python callers while making
        # the master configuration's training section authoritative for YAML.
        normalized = dict(raw_config)
        normalized.update(training_config)
        model_config = normalized.get("model", {}) or {}
        physics_config = normalized.get("physics_model_args", {}) or {}
        if not isinstance(model_config, Mapping):
            raise TypeError("The optional 'model' configuration section must be a mapping.")

        names = {item.name for item in fields(cls) if item.name != "extra"}
        values = {
            name: normalized[name] if name in normalized else model_config[name]
            for name in names
            if name in normalized or name in model_config
        }
        for name in names - values.keys():
            if name in physics_config:
                values[name] = physics_config[name]

        config = cls(**values)
        config._validate()
        config.extra = {
            key: value for key, value in normalized.items() if key not in names and key != "training"
        }
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
        if not isinstance(self.encoder_args, Mapping):
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
    """Loads the .h5 file that contains the pixel indices associated with the survey like the different patches. That
    file is generated in notebooks/survey_file_gen/pixel_file.ipynb. If the conf argument is not passed, the default
    within the directory where this file resides is used.

    Args:
        conf dict: A dictionary with msfm config.

    Returns:
        data_vec_pix: data vector pixels including padding in NEST ordering (non-tomographic).
    """

    import h5py
    with h5py.File(conf["files"]["pixels"], "r") as f:
        # pixel indices of padded data vector
        data_vec_pix = f["data_vec"][:]


    return data_vec_pix
