"""Configuration objects and shared configuration loading helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from euclid_multiprobe_deeplss_training.utils.logger import get_logger

LOGGER = get_logger(__file__)

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
    likelihood: dict[str, Any] = field(default_factory=dict)

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
        likelihood = raw_config.get("likelihood", {})
        if not isinstance(likelihood, Mapping):
            raise TypeError("The 'likelihood' configuration section must be a mapping.")

        # Keep accepting the original flat training mapping for programmatic
        # callers. Values in the merged ``training`` section are authoritative.
        legacy = {
            key: value
            for key, value in raw_config.items()
            if key not in {"forward_model", "training", "likelihood"}
        }
        sections = {
            "forward_model": dict(forward_model),
            "training": _merge_mappings(legacy, training),
            "likelihood": dict(likelihood),
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

def load_pixel_file(conf):
    """Loads the .h5 file that contains the pixel indices associated with the survey like the different patches. That
    file is generated in notebooks/survey_file_gen/pixel_file.ipynb. If the conf argument is not passed, the default
    within the directory where this file resides is used.

    Args:
        conf (str, dict, optional): Can be either a string (a config.yaml is read in), a dictionary (the config is
            passed through) or None (the default config is loaded). The relative paths are stored here. Defaults to
            None.

    Returns:
        data_vec_pix: data vector pixels including padding in NEST ordering (non-tomographic).
        patches_pix_dict: For "WL" (tomographic) and "GC" (non-tomographic), four patch indices in RING
            ordering to cut out from the full sky maps.
        corresponding_pix_dict: For "WL" (tomographic) and "GC" (non-tomographic), needed to convert the
            pixels in RING ordering to NEST inside the datavector.
        gamma2_signs: Signs for gamma2 that come from mirroring the survey patch, needed for WL only.
    """

    import os

    import h5py

    if os.path.isabs(conf["files"]["pixels"]):
        pixel_file = conf["files"]["pixels"]
    else:
        file_dir = os.path.dirname(__file__)
        repo_dir = os.path.abspath(os.path.join(file_dir, "../.."))
        pixel_file = os.path.join(repo_dir, conf["files"]["pixels"])
    LOGGER.debug(f"Loading the pixel file from {pixel_file}")

    with h5py.File(pixel_file, "r") as f:
        # pixel indices of padded data vector
        data_vec_pix = f["data_vec"][:]

        # WL sample: weak lensing
        wl_tomo_patches_pix = []
        wl_tomo_corresponding_pix = []
        for z_bin in conf["survey"]["WL"]["z_bins"]:

            # shape (n_bins, pix_in_bin)
            dset = f"WL/patches/{z_bin}"

            assert dset in f.keys(), f"Dataset {dset} not found in {pixel_file}"

            patches_pix = f[dset][:]
            # shape (pix_in_bin,)

            dset = f"WL/patch_to_data_vec/{z_bin}"
            assert dset in f.keys(), f"Dataset {dset} not found in {pixel_file}"
            corresponding_pix = f[dset][:]

            wl_tomo_patches_pix.append(patches_pix)
            wl_tomo_corresponding_pix.append(corresponding_pix)

        # to correct the shear for patch cut outs that have been mirrored
        gamma2_signs = f["WL/gamma_2_sign"][:]

        # GC sample: galaxy clustering
        gc_tomo_patches_pix = []
        gc_tomo_corresponding_pix = []
        for z_bin in conf["survey"]["GC"]["z_bins"]:

            dset = f"GC/patches/{z_bin}"
            assert dset in f.keys(), f"Dataset {dset} not found in {pixel_file}"
            patches_pix = f[dset][:]

            dset = f"GC/patch_to_data_vec/{z_bin}"
            assert dset in f.keys(), f"Dataset {dset} not found in {pixel_file}"
            corresponding_pix = f[dset][:]

            gc_tomo_patches_pix.append(patches_pix)
            gc_tomo_corresponding_pix.append(corresponding_pix)

    LOGGER.info(f"Loaded the pixel file {pixel_file}")

    # package into dictionaries
    patches_pix_dict = {}
    patches_pix_dict["WL"] = wl_tomo_patches_pix
    patches_pix_dict["GC"] = gc_tomo_patches_pix

    corresponding_pix_dict = {}
    corresponding_pix_dict["WL"] = wl_tomo_corresponding_pix
    corresponding_pix_dict["GC"] = gc_tomo_corresponding_pix

    return data_vec_pix, patches_pix_dict, corresponding_pix_dict, gamma2_signs
