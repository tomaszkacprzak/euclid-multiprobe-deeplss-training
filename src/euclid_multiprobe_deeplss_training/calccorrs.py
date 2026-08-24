"""Calculate TreeCorr two-point correlations for generated training maps."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from pathlib import Path
import os
from typing import Any

import healpy as hp
import numpy as np
import torch
import webdataset as wds
import h5py
import treecorr

from .training import TrainingConfig, load_physics_model_class
from .utils.config import load_config, load_pixel_indices, with_forward_model_config
from .utils.logger import get_logger

LOGGER = get_logger(__file__)


def calccorrs(
    config_or_path: str | Path | Mapping[str, Any] | TrainingConfig,
    *,
    output_dir: Path,
    num_batches_per_file: int = 10,
    file_index: int = 0,
    dataset_split: str = 'training',
) -> list[Path]:
    """Calculate all auto/cross map correlations and write batched WebDataset shards.

    Each WebDataset sample is one input batch.  ``xi_p.pth`` and ``xi_m.pth``
    contain all shear--shear pairs, while ``xi.pth`` contains scalar--scalar and
    scalar--shear pairs.  The corresponding pair-index tensors identify the
    input probes for every correlation.  Labels and source indices retain the
    same batch dimension as the correlation tensors.
    """

    LOGGER.info(f"Calculating correlations {file_index}")
    if num_batches_per_file <= 0:
        raise ValueError("num_batches_per_file must be positive.")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        LOGGER.info(f"Created output directory {output_dir}")
    else:
        LOGGER.info(f"Using existing output directory {output_dir}")

    from msfm.onthefly_pipeline import OntheflyPipeline

    config, raw_config = _coerce_config(config_or_path)
    requested_device = 'cpu'
    run_device = torch.device(requested_device or ("cuda" if torch.cuda.is_available() else "cpu"))
    LOGGER.info(f"Running on {run_device}")
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
        seed=file_index * 1001,
        nside=nside,
    ).to(run_device)
    loader = OntheflyPipeline(
        webds_pattern=config.records_pattern,
        batch_size=config.batch_size,
        physics_model=physics_model,
        downsampler=None,
        smoother=None,
        num_workers=config.num_workers,
        device=run_device,
        seed=file_index * 1000,
        validation=dataset_split == 'validation',
    )

    written_paths: list[Path] = []
    writer: wds.TarWriter | None = None
    examples_written = 0
    ncpus = len(os.sched_getaffinity(0))
    LOGGER.info(f"Using {ncpus} CPUs")

    with torch.no_grad():
        
        shard_path = os.path.join(output_dir, f"corrs_{dataset_split}_{file_index:06d}.tar")
        LOGGER.info(f"Writing correlations to {shard_path}, starting {num_batches_per_file} batches per file with {config.batch_size} examples per batch")
        with wds.TarWriter(str(shard_path)) as writer:

            for batch_index, (maps, labels, inds) in enumerate(loader):

                maps = maps.to(device=run_device, dtype=torch.float32)
                map_list = physics_model.unstack_batch_channels(maps)
                weight_list = physics_model.prepare_weights(map_list)
                map_list = physics_model.preprocess_for_correlations(map_list)

                if batch_index == 0:
                    for map_index, m in enumerate(map_list):
                        LOGGER.info(f"map {map_index:>4d} shape={m.shape} dtype={m.dtype}")

                LOGGER.info(f"batch {batch_index:>6d} / {num_batches_per_file:>6d}")

                # Main magic - calculate the correlations
                correlations, separations = calculate_batch_correlations(map_list, weight_list, ncpus=ncpus, coordinates=coordinates, treecorr_config=corr_config)

                for example in range(config.batch_size):
                    sample = {
                        "__key__": f"example-{example:0d}",
                        "corr.pth": correlations[example],
                        "labels.pth": labels[example],
                        "inds.pth": inds[example]
                    }

                    writer.write(sample)
    
                LOGGER.debug(
                    f'Batch {batch_index + 1}: maps.shape={maps.shape} correlations.shape={correlations.shape}, separations={separations.shape}',
                )
                
                # write the separations and example correlations to a separate file
                if batch_index == 0:
                    example_batches_path = os.path.join(output_dir, f"example-batches-{file_index:06d}.h5")
                    with h5py.File(example_batches_path, "w") as f:
                        f.create_dataset("separations", data=separations, compression="lzf", shuffle=True)
                        f.create_dataset("correlations", data=correlations, compression="lzf", shuffle=True)
                        f.create_dataset("labels", data=labels, compression="lzf", shuffle=True)
                        f.create_dataset("inds", data=inds, compression="lzf", shuffle=True)

                examples_written += config.batch_size

    LOGGER.info(f"Wrote correlations for {examples_written} examples")
    return written_paths


def calculate_batch_correlations(
    maps: Sequence[torch.Tensor],
    weight_maps: Sequence[torch.Tensor|None],
    *,
    ncpus: int = 1,
    coordinates: tuple[np.ndarray, np.ndarray],
    treecorr_config: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    """Return TreeCorr correlations for a batch of scalar/complex shear maps."""
    

    if not maps:
        raise ValueError("At least one map probe is required.")
    batch_size, pixel_count = maps[0].shape
    if any(value.ndim != 2 or value.shape != (batch_size, pixel_count) for value in maps):
        raise ValueError("Every map must have the same (batch, pixel) shape.")
    ra, dec = coordinates
    if ra.shape != (pixel_count,) or dec.shape != (pixel_count,):
        raise ValueError("Coordinate arrays must contain one position per map pixel.")

    cpu_maps = [value.detach().cpu().numpy() for value in maps]
    cpu_weight_maps = [value.detach().cpu().numpy() if value is not None else None for value in weight_maps]
    
    examples_corr = []
    for example in LOGGER.progressbar(range(batch_size), desc="Computing correlations"):

        example_corr = []
        example_bins = []

        for i1 in range(len(maps)):
            for i2 in range(i1, len(maps)):

                map1 =  cpu_maps[i1][example]
                map2 =  cpu_maps[i2][example]
                weight1 = weight_maps[i1][example] if weight_maps[i1] is not None else None
                weight2 = weight_maps[i2][example] if weight_maps[i2] is not None else None

                catalog1 = _catalog(treecorr, ra, dec, map1, weight1)
                catalog2 = _catalog(treecorr, ra, dec, map2, weight2)
                corr, bins = _correlation(catalog1, catalog2, treecorr_config, ncpus=ncpus)

                LOGGER.debug(f"example {example:>6d} correlation {i1:>4d} {i2:>4d} shape={corr.shape} bins={bins.shape} weight1={True if weight1 is not None else False} weight2={True if weight2 is not None else False}")

                example_corr.append(corr)
                example_bins.append(bins)

        examples_corr.append(np.concatenate(example_corr))
        example_bins = np.concatenate(example_bins)

    examples_corr = np.array(examples_corr)
   
    return examples_corr, example_bins



def _correlation(catalog1, catalog2, treecorr_config, ncpus: int):

    if catalog1.type == "shear" and catalog2.type == "shear":
        correlation = treecorr.GGCorrelation(dict(treecorr_config))
        correlation.process(catalog1, catalog2, num_threads=ncpus)
        corr =  np.concatenate([correlation.xip, correlation.xim])
    elif catalog1.type == "scalar" and catalog2.type == "scalar":
        correlation = treecorr.KKCorrelation(dict(treecorr_config))
        correlation.process(catalog1, catalog2, num_threads=ncpus)
        corr = np.array(correlation.xi)
    elif catalog1.type == "scalar" and catalog2.type == "shear":
        correlation = treecorr.KGCorrelation(dict(treecorr_config))
        correlation.process(catalog1, catalog2, num_threads=ncpus)
        corr = np.array(correlation.xi)
    else:
        raise ValueError("Invalid catalog types.")

    sep = np.array(correlation.rnom)
    return corr, sep


def _catalog(treecorr: Any, ra: np.ndarray, dec: np.ndarray, values: np.ndarray, weight: np.ndarray|None = None) -> Any:
    
    common = {"ra": ra, "dec": dec, "ra_units": "rad", "dec_units": "rad"}

    if np.iscomplexobj(values):
        catalog = treecorr.Catalog(**common, g1=np.real(values), g2=np.imag(values), w=weight)
        catalog.type = "shear"
    else:
        catalog = treecorr.Catalog(**common, k=np.asarray(values), w=weight)
        catalog.type = "scalar"
    
    return catalog


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
        "nbins": int(settings.get("nbins", 50)),
        "min_sep": float(settings.get("min_sep", pixel_scale)),
        "max_sep": float(settings.get("max_sep", np.pi)),
        "sep_units": str(settings.get("sep_units", "rad")),
        # "bin_slop": float(settings.get("bin_slop", 0.1)),
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
    output_dir: str | Path = "corrs",
    file_index: int = 0,
    num_batches_per_file: int = 10,
    dataset_split: str = 'training',
) -> list[Path]:
    """Load a YAML configuration and calculate its training correlations."""
    path = Path(config_path)
    raw_config = with_forward_model_config(load_config(path), path.parent)
    return calccorrs(raw_config, output_dir=output_dir, file_index=file_index, num_batches_per_file=num_batches_per_file, dataset_split=dataset_split)


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
