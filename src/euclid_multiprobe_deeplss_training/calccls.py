"""Calculate auto power spectra for generated DeepLSS training maps."""

from __future__ import annotations

import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import h5py
import torch

from .training import TrainingConfig, load_physics_model_class
from .utils.cls_cuhpx import PartSkyAutoCls
from .utils.config import load_config, load_pixel_indices, with_forward_model_config
from .utils.logger import get_logger

LOGGER = get_logger(__file__)


def calccls(
    config_or_path: str | Path | Mapping[str, Any] | TrainingConfig,
    *,
    output_path: str | Path = "cls.h5",
    device: torch.device | str | None = None,
) -> Path:
    """Calculate auto spectra for every batch in one training-set epoch.

    The output contains one extensible dataset per map probe, named
    ``cls_0``, ``cls_1``, and so on. Each batch is appended immediately so
    neither CPU nor GPU memory use grows over the epoch.
    """
    from msfm.onthefly_pipeline import OntheflyPipeline

    config, raw_config = _coerce_config(config_or_path)
    requested_device = device or raw_config.get("device")
    run_device = torch.device(requested_device or ("cuda" if torch.cuda.is_available() else "cpu"))

    indices = load_pixel_indices(config.forward_model)
    analysis = config.forward_model["analysis"]
    nside = int(analysis["n_side"])
    cls_config = raw_config.get("calccls", {}) or {}
    if not isinstance(cls_config, Mapping):
        raise TypeError("The optional 'calccls' configuration section must be a mapping.")
    lmax = int(cls_config.get("lmax", raw_config.get("lmax", analysis.get("l_max", 3 * nside))))

    physics_model_class = load_physics_model_class(config.physics_model)
    physics_model = physics_model_class(
        config.forward_model,
        scalers=True,
        device=run_device,
        seed=int(time.time()),
        nside=nside,
    ).to(run_device)
    loader = OntheflyPipeline(
        webds_pattern=config.records_pattern,
        batch_size=config.batch_size,
        physics_model=physics_model,
        downsampler=None,
        smoother=None,
        num_workers=config.num_workers,
    )
    cls_calculator = PartSkyAutoCls(
        indices=torch.as_tensor(indices, dtype=torch.long, device=run_device),
        nside=nside,
        lmax=lmax,
    ).to(run_device)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Calculating auto power spectra for one training epoch")
    # Opening in write mode creates an empty file before loader iteration and
    # deliberately replaces an earlier result at the requested path.
    j = 0
    with h5py.File(output_path, "w") as output_file, torch.no_grad():
        for maps, _labels in loader:
            j += 1
            maps = maps.to(device=run_device, dtype=torch.float32)
            maps_list = physics_model.unstack_batch_channels(maps)
            batch_spectra_list = cls_calculator.forward(*maps_list)
            _append_spectra(output_file, batch_spectra_list)
            output_file.flush()
            batch_spectra = torch.stack(batch_spectra_list, dim=-1)
            LOGGER.debug(f"Batch {j: 5d}: input maps shape: {maps.shape}, output spectra shape: {batch_spectra.shape}")

    return output_path


def calccls_from_config(config_path: str | Path, *, output_path: str | Path = "cls.h5") -> Path:
    """Load a YAML configuration file and calculate its training spectra."""
    path = Path(config_path)
    raw_config = with_forward_model_config(load_config(path), path.parent)
    return calccls(raw_config, output_path=output_path)


def _append_spectra(output_file: h5py.File, spectra: tuple[torch.Tensor, ...]) -> None:
    """Append one spectra batch to an open HDF5 output file."""
    if len(output_file) and len(output_file) != len(spectra):
        raise ValueError(f"Expected {len(output_file)} spectra tensors, got {len(spectra)}.")

    for probe_index, spectrum in enumerate(spectra):
        values = spectrum.detach().cpu().numpy()
        dataset_name = f"cls_{probe_index}"
        if dataset_name not in output_file:
            output_file.create_dataset(
                dataset_name,
                data=values,
                maxshape=(None, *values.shape[1:]),
                chunks=True,
            )
            continue

        dataset = output_file[dataset_name]
        old_size = dataset.shape[0]
        dataset.resize(old_size + values.shape[0], axis=0)
        dataset[old_size:] = values


def _coerce_config(
    config_or_path: str | Path | Mapping[str, Any] | TrainingConfig,
) -> tuple[TrainingConfig, dict[str, Any]]:
    """Normalize config input while retaining calccls-specific settings."""
    if isinstance(config_or_path, TrainingConfig):
        raw_config = {**config_or_path.extra}
        for field_name in config_or_path.__dataclass_fields__:
            if field_name != "extra":
                raw_config[field_name] = getattr(config_or_path, field_name)
        return config_or_path, raw_config
    if isinstance(config_or_path, str | Path):
        path = Path(config_or_path)
        raw_config = with_forward_model_config(load_config(path), path.parent)
    else:
        raw_config = dict(config_or_path)
    return TrainingConfig.from_mapping(raw_config), raw_config
