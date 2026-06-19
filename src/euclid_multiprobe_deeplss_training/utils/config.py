"""Shared configuration loading helpers."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file with ``yaml.safe_load``."""
    with Path(path).open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise TypeError("The YAML training config must load to a mapping.")
    return loaded


def load_forward_model_config(path: str | Path) -> dict[str, Any]:
    """Load the forward-model YAML configuration file."""
    return load_config(path)


def with_forward_model_config(raw_config: Mapping[str, Any], base_dir: Path | None = None) -> dict[str, Any]:
    """Return a copy of ``raw_config`` with ``forward_model`` loaded when configured."""
    config = dict(raw_config)
    forward_model_path = config.get("config_forward_model")
    if forward_model_path is None:
        return config

    path = Path(forward_model_path)
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    config["forward_model"] = load_forward_model_config(path)
    return config
