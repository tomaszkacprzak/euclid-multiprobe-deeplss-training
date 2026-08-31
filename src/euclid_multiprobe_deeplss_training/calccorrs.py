"""Calculate correlations for generated training maps."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from pathlib import Path
import os
from typing import Any

import healpy as hp
import numpy as np
import torch
import h5py


from .training import TrainingConfig, load_physics_model_class
from .utils.config import load_config, load_pixel_indices, with_forward_model_config
from .utils.logger import get_logger

LOGGER = get_logger(__file__)


def calccorrs(
    config_or_path: str | Path | Mapping[str, Any] | TrainingConfig,
    *,
    output_dir: Path,
    num_batches_per_file: int = 100,
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
    requested_device = 'cuda'
    run_device = torch.device(requested_device or ("cuda" if torch.cuda.is_available() else "cpu"))
    LOGGER.info(f"Running on {run_device}")
    indices = np.asarray(load_pixel_indices(config.forward_model), dtype=np.int64)
    analysis = config.forward_model["analysis"]
    nside = int(analysis["n_side"])
    nside_down = int(analysis["n_side_down"])
    L = hp.nside2order(nside)
    R = hp.nside2order(nside_down)
    footprint_indices = get_footprint_indices(indices, L, R)
    LOGGER.info(f"indices min={indices.min()}, max={indices.max()}")
    LOGGER.info(f"footprint_indices min={footprint_indices.min()}, max={footprint_indices.max()}")
    LOGGER.info(f"L={L}, R={R}, footprint_indices {len(footprint_indices)}/{hp.nside2npix(nside_down)}")

    OntheflyPhysicsModel = load_physics_model_class(config.physics_model)
    physics_model = OntheflyPhysicsModel(
        config.forward_model,
        scalers=False,
        device=run_device,
        seed=file_index * 1001,
        nside=nside,
        **config.physics_model_args if hasattr(config, "physics_model_args") else {},
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

    examples_written = 0
    compression_args = {"compression": "lzf", "shuffle": True}

    from pyracorr import PyracorrFastFootprint
    device = torch.device("cuda")
    LOGGER.info(f"Using PyracorrFast with L={L}")
    spins = [0]*physics_model.num_channels
    correlator = PyracorrFastFootprint(L, 
                    spins, 
                    R_footprint=R,
                    matmul_precision="high",
                    footprint_indices=footprint_indices,
                    recompute_pairs=False, 
                    doublesets=True,
                    pairs_filename=f'pairs_L{L:02d}.h5').to(device)

    time_start = time.time()
    time_corr = 0
    with torch.no_grad():

        shard_path = os.path.join(output_dir, f"corrs_{dataset_split}_{file_index:06d}.h5")
        with h5py.File(shard_path, "w") as f:
            
            LOGGER.info(f"Writing correlations to {shard_path}, starting {num_batches_per_file} batches per file with {config.batch_size} examples per batch")

            for batch_index, (maps, labels, inds) in enumerate(loader):

                # raw batch of shape (batch_size, num_pixels, num_channels)
                maps = maps.to(device=run_device, dtype=torch.float32)

                # calculate wegiths used for correlations
                # weights of shape (batch_size, num_pixels, num_channels)
                weights = physics_model.get_weight_maps(maps)

                # convert the raw counts to density contrast, subtract the mean from shear/kappa maps
                # (batch_size, num_pixels, num_channels)
                maps = physics_model.preprocess_for_correlations(maps)

                if batch_index == 0:
                    LOGGER.info(f"maps    shape={maps.shape}    dtype={maps.dtype}")
                    LOGGER.info(f"weights shape={weights.shape} dtype={weights.dtype}")

                # Main magic - calculate the correlations
                maps = torch.movedim(maps, -1, 1) # -> (batch_size, num_channels, num_pixels)
                weights = torch.movedim(weights, -1, 1) # -> (batch_size, num_channels, num_pixels)
                time_start_corr = time.time()

                # main magic - calculate the correlations
                # num_correlationis 2*(L+1) for pyracorr with doublesets
                # correlations of shape (batch_size, num_channels, num_channels, num_correlations)
                correlations = correlator(maps, weights)

                time_corr += time.time() - time_start_corr
                separations = correlator.theta
                LOGGER.info(f'Batch {batch_index + 1}: maps.shape={maps.shape} correlations.shape={correlations.shape}, separations={separations.shape}')

                                
                # write the separations and example correlations to a separate file

                f.create_dataset(f"batch{batch_index:04d}/separations",  data=separations.cpu().numpy(),  **compression_args)
                f.create_dataset(f"batch{batch_index:04d}/correlations", data=correlations.cpu().numpy(), **compression_args)
                f.create_dataset(f"batch{batch_index:04d}/labels",       data=labels.cpu().numpy(),       **compression_args)
                f.create_dataset(f"batch{batch_index:04d}/inds",         data=inds.cpu().numpy(),         **compression_args)

                examples_written += config.batch_size

                if batch_index == num_batches_per_file - 1:
                    break

    LOGGER.info(f"Wrote correlations in {shard_path} for {examples_written} examples")
    LOGGER.info(f"Time total: {time.time() - time_start:.2f}s, time corr: {time_corr:.2f}s")

def get_footprint_indices(indices, L, R):
    assert np.all(np.diff(indices) >= 0), "indices must be sorted"
    return np.unique(np.asarray(indices) //  ((hp.order2nside(L) // hp.order2nside(R)) ** 2))

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
