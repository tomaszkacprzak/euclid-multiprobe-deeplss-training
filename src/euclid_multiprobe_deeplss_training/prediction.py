"""Checkpoint prediction over the complete validation dataset."""

from __future__ import annotations

import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from .networks.builder import build_encoder, build_loss
from .training import load_physics_model_class
from .utils.config import Config, ConfigPaths, config_paths, load_config, load_pixel_indices
from .utils.logger import get_logger

LOGGER = get_logger(__file__)


def _extra_mapping(config: Config, name: str) -> dict[str, Any]:
    value = config.training.get(name, {})
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    return dict(value)


@torch.no_grad()
def predict(
    config: Config,
    *,
    checkpoint: str | Path,
    output_file: str | Path,
    batch_size: int | None = None,
    num_examples: int | None = None,
    device: torch.device | str | None = None,
) -> Path:
    """Predict every example from the validation split and write an HDF5 file."""
    from msfm.onthefly_pipeline import OntheflyPipeline

    evaluation_batch_size = config.training['batch_size'] if batch_size is None else batch_size
    if evaluation_batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    indices = load_pixel_indices(config.forward_model)
    nside = config.forward_model["analysis"]["n_side"]
    physics_model_class = load_physics_model_class(config.training['physics_model'])
    physics_model = physics_model_class(
        config.forward_model,
        scalers=True,
        device=device,
        seed=int(time.time()),
        nside=nside,
        **config.training['physics_model_args'],
    ).to(device)
    validation_loader = OntheflyPipeline(
        webds_pattern=config.training['records_pattern'],
        batch_size=evaluation_batch_size,
        physics_model=physics_model,
        downsampler=None,
        smoother=None,
        num_workers=config.training['num_workers'],
        validation=True,
    )

    encoder = build_encoder(
        config.training['encoder_name'],
        num_channels=physics_model.num_channels,
        embed_dim=config.training['embed_dim'],
        num_pixels=validation_loader.num_pixels,
        nside=nside,
        nside_down=int(config.forward_model["analysis"]["n_side_down"]),
        encoder_args=config.training['encoder_args'],
        batch_size=evaluation_batch_size,
        indices=indices,
        physics_model=physics_model,
        device=device,
    ).to(device)
    model = build_loss(
        config.training['loss_function'],
        encoder=encoder,
        num_targets=physics_model.num_targets,
        embed_dim=config.training['embed_dim'],
        batch_size=evaluation_batch_size,
        loss_args=_extra_mapping(config, "loss_args"),
    ).to(device)

    checkpoint_data = torch.load(Path(checkpoint), map_location=device)
    model.load_state_dict(checkpoint_data["model_state_dict"])
    model.eval()

    label_batches: list[torch.Tensor] = []
    prediction_batches: list[torch.Tensor] = []
    inds_batches: list[torch.Tensor] = []
    j = 0
    i = 0
    while i < num_examples:
        for batch in validation_loader:
            inputs, labels, inds = batch
            inputs = inputs.to(device=device, dtype=torch.float32)
            predictions = model.predict(inputs)
            label_batches.append(labels.detach().cpu())
            inds_batches.append(inds.detach().cpu())
            prediction_batches.append(predictions.detach().cpu())
            LOGGER.debug(
                "Batch %5d: input maps shape: %s, labels shape: %s, predictions shape: %s, indices shape: %s",
                j,
                inputs.shape,
                labels.shape,
                predictions.shape,
                inds.shape,
            )
            j += 1
            i += evaluation_batch_size
            if i >= num_examples:
                break
            if j % 10 == 0:
                LOGGER.info(f"Predicted {i:>10d} examples out of {num_examples:>10d} [{i / num_examples * 100:.2f}%]")

    if not label_batches:
        raise ValueError("The validation set did not produce any examples.")
    labels = torch.cat(label_batches).numpy()
    predictions = torch.cat(prediction_batches).numpy()
    inds = torch.cat(inds_batches).numpy()
    if labels.shape != predictions.shape:
        raise ValueError(f"Labels and predictions have different shapes: {labels.shape} and {predictions.shape}.")
    if len(inds) != len(labels):
        raise ValueError(f"Indices and labels have different lengths: {len(inds)} and {len(labels)}.")
    import h5py

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as handle:
        handle.create_dataset("grid/cosmos/test", data=labels)
        handle.create_dataset("grid/preds/test", data=predictions)
        handle.create_dataset("grid/indices/test", data=inds)
    LOGGER.info("Wrote %d validation predictions to %s", len(labels), output_path)
    return output_path


def predict_from_config(
    config_path: ConfigPaths,
    *,
    checkpoint: str | Path,
    output_file: str | Path,
    batch_size: int | None = None,
    num_examples: int | None = None,
    device: torch.device | str | None = None,
) -> Path:
    """Load a training config and predict its complete validation set."""
    paths = config_paths(config_path)
    raw_config = load_config(paths)
    return predict(
        Config.from_mapping(raw_config),
        checkpoint=checkpoint,
        output_file=output_file,
        batch_size=batch_size,
        num_examples=num_examples,
        device=device,
    )
