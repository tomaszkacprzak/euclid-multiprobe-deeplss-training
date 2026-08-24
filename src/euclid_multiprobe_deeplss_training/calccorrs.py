"""Calculate TreeCorr two-point correlations for generated training maps."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import healpy as hp
import numpy as np
import torch
import webdataset as wds

from .training import TrainingConfig, load_physics_model_class
from .utils.config import load_config, load_pixel_indices, with_forward_model_config
from .utils.logger import get_logger

LOGGER = get_logger(__file__)


def calccorrs(
    config_or_path: str | Path | Mapping[str, Any] | TrainingConfig,
    *,
    output_path: str | Path = "corrs-%06d.tar",
    num_examples: int = 100,
    num_batches_per_file: int = 10,
    device: torch.device | str | None = None,
) -> list[Path]:
    """Calculate all auto/cross map correlations and write batched WebDataset shards.

    Each WebDataset sample is one input batch.  ``xi_p.pth`` and ``xi_m.pth``
    contain all shear--shear pairs, while ``xi.pth`` contains scalar--scalar and
    scalar--shear pairs.  The corresponding pair-index tensors identify the
    input probes for every correlation.  Labels and source indices retain the
    same batch dimension as the correlation tensors.
    """
    if num_examples <= 0:
        raise ValueError("num_examples must be positive.")
    if num_batches_per_file <= 0:
        raise ValueError("num_batches_per_file must be positive.")

    from msfm.onthefly_pipeline import OntheflyPipeline

    config, raw_config = _coerce_config(config_or_path)
    requested_device = device or raw_config.get("device")
    run_device = torch.device(requested_device or ("cuda" if torch.cuda.is_available() else "cpu"))
    indices = np.asarray(load_pixel_indices(config.forward_model), dtype=np.int64)
    analysis = config.forward_model["analysis"]
    nside = int(analysis["n_side"])
    corr_config = _correlation_config(raw_config, nside)
    coordinates = _pixel_coordinates(indices, nside)

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

    written_paths: list[Path] = []
    writer: wds.TarWriter | None = None
    examples_written = 0
    try:
        with torch.no_grad():
            for batch_index, (maps, labels, inds) in enumerate(loader):
                if batch_index % num_batches_per_file == 0:
                    if writer is not None:
                        writer.close()
                    shard_path = _shard_path(output_path, len(written_paths))
                    shard_path.parent.mkdir(parents=True, exist_ok=True)
                    writer = wds.TarWriter(str(shard_path))
                    written_paths.append(shard_path)

                maps = maps.to(device=run_device, dtype=torch.float32)
                map_list = physics_model.unstack_batch_channels(maps)
                correlations = calculate_batch_correlations(map_list, coordinates=coordinates, treecorr_config=corr_config)
                sample = {
                    "__key__": f"batch-{batch_index:06d}",
                    "xi_p.pth": correlations["xi_p"],
                    "xi_m.pth": correlations["xi_m"],
                    "xi.pth": correlations["xi"],
                    "shear_pair_indices.pth": correlations["shear_pair_indices"],
                    "correlation_pair_indices.pth": correlations["correlation_pair_indices"],
                    "labels.pth": labels.detach().cpu(),
                    "inds.pth": inds.detach().cpu(),
                }
                assert writer is not None
                writer.write(sample)
                examples_written += len(maps)
                LOGGER.debug(
                    "Batch %5d: input maps %s, xi %s, xi+ %s",
                    batch_index + 1,
                    maps.shape,
                    correlations["xi"].shape,
                    correlations["xi_p"].shape,
                )
                if examples_written >= num_examples:
                    break
    finally:
        if writer is not None:
            writer.close()

    LOGGER.info("Wrote correlations for %d examples to %d WebDataset shard(s)", examples_written, len(written_paths))
    return written_paths


def calculate_batch_correlations(
    maps: Sequence[torch.Tensor],
    *,
    coordinates: tuple[np.ndarray, np.ndarray],
    treecorr_config: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    """Return TreeCorr correlations for a batch of scalar/complex shear maps."""
    import treecorr

    if not maps:
        raise ValueError("At least one map probe is required.")
    batch_size, pixel_count = maps[0].shape
    if any(value.ndim != 2 or value.shape != (batch_size, pixel_count) for value in maps):
        raise ValueError("Every map must have the same (batch, pixel) shape.")
    ra, dec = coordinates
    if ra.shape != (pixel_count,) or dec.shape != (pixel_count,):
        raise ValueError("Coordinate arrays must contain one position per map pixel.")

    shear_pairs: list[tuple[int, int]] = []
    other_pairs: list[tuple[int, int]] = []
    xip_batches: list[list[np.ndarray]] = [[] for _ in range(batch_size)]
    xim_batches: list[list[np.ndarray]] = [[] for _ in range(batch_size)]
    xi_batches: list[list[np.ndarray]] = [[] for _ in range(batch_size)]
    cpu_maps = [value.detach().cpu().numpy() for value in maps]

    for first in range(len(maps)):
        for second in range(first, len(maps)):
            first_shear, second_shear = maps[first].is_complex(), maps[second].is_complex()
            target_pairs = shear_pairs if first_shear and second_shear else other_pairs
            target_pairs.append((first, second))
            for example in range(batch_size):
                catalog1 = _catalog(treecorr, ra, dec, cpu_maps[first][example], first_shear)
                catalog2 = catalog1 if first == second else _catalog(treecorr, ra, dec, cpu_maps[second][example], second_shear)
                if first_shear and second_shear:
                    correlation = treecorr.GGCorrelation(dict(treecorr_config))
                    correlation.process(catalog1, catalog2)
                    xip_batches[example].append(np.asarray(correlation.xip, dtype=np.float32))
                    xim_batches[example].append(np.asarray(correlation.xim, dtype=np.float32))
                elif not first_shear and not second_shear:
                    correlation = treecorr.KKCorrelation(dict(treecorr_config))
                    correlation.process(catalog1, catalog2)
                    xi_batches[example].append(np.asarray(correlation.xi, dtype=np.float32))
                else:
                    # KGCorrelation requires the scalar catalog first.
                    scalar_catalog, shear_catalog = (catalog2, catalog1) if first_shear else (catalog1, catalog2)
                    correlation = treecorr.KGCorrelation(dict(treecorr_config))
                    correlation.process(scalar_catalog, shear_catalog)
                    xi_batches[example].append(np.asarray(correlation.xi, dtype=np.float32))

    nbins = int(treecorr_config["nbins"])
    return {
        "xi_p": _stack_correlations(xip_batches, batch_size, len(shear_pairs), nbins),
        "xi_m": _stack_correlations(xim_batches, batch_size, len(shear_pairs), nbins),
        "xi": _stack_correlations(xi_batches, batch_size, len(other_pairs), nbins),
        "shear_pair_indices": torch.tensor(shear_pairs, dtype=torch.long).reshape(-1, 2),
        "correlation_pair_indices": torch.tensor(other_pairs, dtype=torch.long).reshape(-1, 2),
    }


def _catalog(treecorr: Any, ra: np.ndarray, dec: np.ndarray, values: np.ndarray, shear: bool) -> Any:
    common = {"ra": ra, "dec": dec, "ra_units": "rad", "dec_units": "rad"}
    if shear:
        return treecorr.Catalog(**common, g1=np.real(values), g2=np.imag(values))
    return treecorr.Catalog(**common, k=np.asarray(values))


def _stack_correlations(values: list[list[np.ndarray]], batch_size: int, pair_count: int, nbins: int) -> torch.Tensor:
    if pair_count == 0:
        return torch.empty((batch_size, 0, nbins), dtype=torch.float32)
    return torch.from_numpy(np.asarray(values, dtype=np.float32))


def _pixel_coordinates(indices: np.ndarray, nside: int) -> tuple[np.ndarray, np.ndarray]:
    theta, phi = hp.pix2ang(nside, indices, nest=True)
    return np.asarray(phi), np.asarray(np.pi / 2 - theta)


def _correlation_config(raw_config: Mapping[str, Any], nside: int) -> dict[str, Any]:
    settings = raw_config.get("calccorrs", {}) or {}
    if not isinstance(settings, Mapping):
        raise TypeError("The optional 'calccorrs' configuration section must be a mapping.")
    pixel_scale = float(hp.nside2resol(nside))
    return {
        "nbins": int(settings.get("nbins", 20)),
        "min_sep": float(settings.get("min_sep", pixel_scale)),
        "max_sep": float(settings.get("max_sep", np.pi)),
        "sep_units": str(settings.get("sep_units", "rad")),
        "bin_slop": float(settings.get("bin_slop", 0.1)),
    }


def _shard_path(output_path: str | Path, shard_index: int) -> Path:
    pattern = str(output_path)
    if "%" in pattern:
        try:
            return Path(pattern % shard_index)
        except (TypeError, ValueError) as error:
            raise ValueError("output_path must contain a valid integer printf placeholder") from error
    path = Path(pattern)
    if shard_index == 0:
        return path
    return path.with_name(f"{path.stem}-{shard_index:06d}{path.suffix}")


def calccorrs_from_config(
    config_path: str | Path,
    *,
    output_path: str | Path = "corrs-%06d.tar",
    num_examples: int = 100,
    num_batches_per_file: int = 10,
) -> list[Path]:
    """Load a YAML configuration and calculate its training correlations."""
    path = Path(config_path)
    raw_config = with_forward_model_config(load_config(path), path.parent)
    return calccorrs(raw_config, output_path=output_path, num_examples=num_examples, num_batches_per_file=num_batches_per_file)


def _coerce_config(config_or_path: str | Path | Mapping[str, Any] | TrainingConfig) -> tuple[TrainingConfig, dict[str, Any]]:
    """Normalize config input while retaining calccorrs-specific settings."""
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
