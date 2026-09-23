"""Configuration and file interface for conditional likelihood training."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from ..utils.config import ConfigPaths, config_paths, load_config
from ..utils.logger import get_logger
from .likelihood_base import LikelihoodBase
from .likelihood_cnf import ConditionalNormalizingFlowFM
from .likelihood_mdn import GaussianMixtureMDN

LOGGER = get_logger(__file__)

STANDARDIZED_PARAMETER_MIN = 0.0
STANDARDIZED_PARAMETER_MAX = 1.0


def _validate_standardized_labels(labels: torch.Tensor) -> None:
    """Ensure conditioning parameters use the forward-model scale unchanged."""
    if not torch.isfinite(labels).all():
        raise ValueError("labels must contain only finite standardized parameter values.")
    if torch.any(labels < STANDARDIZED_PARAMETER_MIN) or torch.any(labels > STANDARDIZED_PARAMETER_MAX):
        raise ValueError("labels must be standardized parameter values in the inclusive range [0, 1].")


def plot_likelihood_fit(
    model: LikelihoodBase,
    predictions: torch.Tensor,
    labels: torch.Tensor,
):
    """Plot the samples and fitted log-likelihood surface for each parameter.

    Each surface varies one prediction over its observed range and one label
    over the full standardized prior [0, 1], while holding all other dimensions
    at their sample means. This produces a useful two-dimensional slice through
    a multivariate conditional density.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    LikelihoodBase._validate_pairs(predictions, labels, "plot")
    _validate_standardized_labels(labels)
    device = next(model.parameters()).device
    model.eval()
    predictions_device = predictions.float().to(device)
    labels_device = labels.float().to(device)
    with torch.no_grad():
        log_likelihood = model.log_likelihood(predictions_device, labels_device).detach().cpu().numpy()

    predictions_array = predictions.detach().cpu().numpy()
    labels_array = labels.detach().cpu().numpy()

    # sort ascending by log likelihood
    sorting = np.argsort(log_likelihood)
    predictions_array = predictions_array[sorting]
    labels_array = labels_array[sorting]
    log_likelihood = log_likelihood[sorting]

    num_parameters = labels.shape[1]
    fig, axes = plt.subplots(2, num_parameters, figsize=(5 * num_parameters, 8), squeeze=False)
    for index, axis in enumerate(axes[0]):
        axis.scatter(
            labels_array[:, index],
            predictions_array[:, index],
            c=log_likelihood,
            marker="o",
            cmap="Spectral_r",
        )
        axis.set_xlabel(f"Label {index}")
        axis.set_ylabel(f"Prediction {index}")

    mean_predictions = predictions_device.mean(dim=0)
    mean_labels = labels_device.mean(dim=0)
    surface_data = []
    for index in range(num_parameters):
        label_values = torch.linspace(
            STANDARDIZED_PARAMETER_MIN,
            STANDARDIZED_PARAMETER_MAX,
            100,
            device=device,
        )
        prediction_values = torch.linspace(predictions_device[:, index].min(), predictions_device[:, index].max(), 100, device=device)
        label_grid, prediction_grid = torch.meshgrid(label_values, prediction_values, indexing="xy")
        grid_labels = mean_labels.repeat(label_grid.numel(), 1)
        grid_predictions = mean_predictions.repeat(prediction_grid.numel(), 1)
        grid_labels[:, index] = label_grid.ravel()
        grid_predictions[:, index] = prediction_grid.ravel()
        with torch.no_grad():
            grid_log_likelihood = model.log_likelihood(grid_predictions, grid_labels).reshape(label_grid.shape).cpu().numpy()

        surface_data.append((label_values.cpu().numpy(), prediction_values.cpu().numpy(), grid_log_likelihood))

    for index, (axis, (label_values, prediction_values, grid_log_likelihood)) in enumerate(zip(axes[1], surface_data, strict=True)):
        likelihood = np.exp(grid_log_likelihood - np.max(grid_log_likelihood))
        norm = likelihood.sum(axis=1, keepdims=True)
        likelihood = likelihood / norm

        axis.pcolormesh(
            label_values,
            prediction_values,
            likelihood,
            # shading="auto",
            # vmin=surface_min,
            # vmax=surface_max,
            cmap="Spectral_r",
        )
        axis.set_xlabel(f"Label {index}")
        axis.set_ylabel(f"Prediction {index}")

    # Validation above guarantees at least one parameter, and therefore a scatter.
    # assert scatter is not None
    # fig.colorbar(scatter, ax=axes.ravel().tolist(), label="Log likelihood", orientation="horizontal", location="bottom", pad=0.15)
    # assert surface is not None
    # fig.colorbar(
    #     surface,
    #     ax=axes[1].tolist(),
    #     label="Predicted log likelihood",
    #     orientation="horizontal",
    #     location="bottom",
    #     pad=0.15,
    # )

    fig.subplots_adjust(bottom=0.12, right=0.9, hspace=0.45, wspace=0.3)
    return fig


def build_likelihood(num_parameters: int, settings: Mapping[str, Any]) -> LikelihoodBase:
    """Build the likelihood implementation selected by ``model_type``."""
    model_type = settings.get("model_type")
    model_args = settings.get("model_args", {})
    if not isinstance(model_args, Mapping):
        raise TypeError("likelihood.model_args must be a mapping.")
    if model_type == "mdn":
        return GaussianMixtureMDN(num_parameters, **dict(model_args))
    if model_type == "cnffm":
        return ConditionalNormalizingFlowFM(num_parameters, **dict(model_args))
    raise ValueError(f"Unknown likelihood model_type: {model_type!r}.")


def train_likelihood(
    settings: Mapping[str, Any],
    *,
    input_file: str | Path,
    output_file: str | Path,
    device: torch.device | str | None = None,
) -> tuple[LikelihoodBase, dict[str, list[float]]]:
    """Train from a prediction HDF5 file containing ``predictions`` and ``labels``.

    Predictions are theta_obs and labels are theta_true. Both datasets must
    have shape ``(N, M)`` and are split along N into training and validation.
    Labels are consumed directly on the standardized scale produced by the
    forward model and must lie in [0, 1]; no prior-range rescaling is applied.
    """
    import h5py

    with h5py.File(input_file, "r") as handle:
        theta_obs = torch.as_tensor(handle["predictions"][:], dtype=torch.float32)
        theta_true = torch.as_tensor(handle["labels"][:], dtype=torch.float32)
    LikelihoodBase._validate_pairs(theta_obs, theta_true, "input")
    _validate_standardized_labels(theta_true)
    LOGGER.info(f"Loaded observations {theta_obs.shape}")
    LOGGER.info(f"Loaded labels {theta_true.shape}")

    validation_fraction = float(settings.get("validation_fraction", 0.2))
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("likelihood.validation_fraction must be between zero and one.")
    validation_size = max(1, round(len(theta_obs) * validation_fraction))
    if validation_size >= len(theta_obs):
        raise ValueError("The input file must contain at least two samples.")

    # A seeded permutation gives reproducible, disjoint splits along the N dimension.
    generator = torch.Generator().manual_seed(int(settings.get("seed", 42)))
    order = torch.randperm(len(theta_obs), generator=generator)
    validation_indices, training_indices = order[:validation_size], order[validation_size:]
    model = build_likelihood(theta_obs.shape[1], settings)

    LOGGER.info(f"Training likelihood model {settings.get('model_type')}")

    history = model.fit(
        theta_obs[training_indices],
        theta_true[training_indices],
        theta_obs[validation_indices],
        theta_true[validation_indices],
        num_epochs=int(settings.get("num_epochs", 100)),
        batch_size=int(settings.get("batch_size", 128)),
        learning_rate=float(settings.get("learning_rate", 1e-3)),
        device=device or settings.get("device"),
    )
    model.save(output_file)

    # plot the likelihood fit
    figure = plot_likelihood_fit(model, theta_obs, theta_true)
    plot_file = Path(output_file).with_suffix(".png")
    figure.savefig(plot_file, bbox_inches="tight")
    LOGGER.info(f"Saved likelihood fit plot to {plot_file}")
    import matplotlib.pyplot as plt

    plt.close(figure)

    return model, history


def train_likelihood_from_config(
    config_path: ConfigPaths,
    *,
    input_file: str | Path,
    output_file: str | Path,
    device: torch.device | str | None = None,
) -> tuple[LikelihoodBase, dict[str, list[float]]]:
    """Load and merge YAML configuration files, then train a likelihood model."""

    if isinstance(config_path, str):
        config_path = [path.strip() for path in config_path.split(",") if path.strip()]
    paths = config_paths(config_path)
    raw_config = load_config(paths)

    settings = raw_config.get("likelihood")
    if settings is None:
        raise ValueError("The 'likelihood' configuration section is missing.")
    if not isinstance(settings, Mapping):
        raise TypeError("The 'likelihood' configuration section must be a mapping.")
    return train_likelihood(
        settings,
        input_file=input_file,
        output_file=output_file,
        device=device,
    )
