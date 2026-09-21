"""Configuration and file interface for conditional likelihood training."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from ..utils.config import ConfigPaths, config_paths, load_config
from .likelihood_base import LikelihoodBase
from .likelihood_mdn import GaussianMixtureMDN


def plot_likelihood_fit(
    model: LikelihoodBase,
    predictions: torch.Tensor,
    labels: torch.Tensor,
):
    """Plot labels against predictions, coloured by their fitted log likelihood."""
    import matplotlib.pyplot as plt

    LikelihoodBase._validate_pairs(predictions, labels, "plot")
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        log_likelihood = model(
            predictions.float().to(device), labels.float().to(device)
        ).detach().cpu().numpy()

    predictions_array = predictions.detach().cpu().numpy()
    labels_array = labels.detach().cpu().numpy()
    num_parameters = labels.shape[1]
    fig, axes = plt.subplots(
        1, num_parameters, figsize=(5 * num_parameters, 4), squeeze=False
    )
    scatter = None
    for index, axis in enumerate(axes[0]):
        scatter = axis.scatter(
            labels_array[:, index],
            predictions_array[:, index],
            c=log_likelihood,
            marker="o",
        )
        axis.set_xlabel(f"Label {index}")
        axis.set_ylabel(f"Prediction {index}")

    # Validation above guarantees at least one parameter, and therefore a scatter.
    assert scatter is not None
    fig.colorbar(scatter, ax=axes.ravel().tolist(), label="Log likelihood")
    fig.subplots_adjust(bottom=0.15, right=0.9, wspace=0.3)
    return fig


def build_likelihood(num_parameters: int, settings: Mapping[str, Any]) -> LikelihoodBase:
    """Build the likelihood implementation selected by ``model_type``."""
    model_type = settings.get("model_type")
    model_args = settings.get("model_args", {})
    if not isinstance(model_args, Mapping):
        raise TypeError("likelihood.model_args must be a mapping.")
    if model_type == "mdn":
        return GaussianMixtureMDN(num_parameters, **dict(model_args))
    raise ValueError(f"Unknown likelihood model_type: {model_type!r}.")


def train_likelihood(
    settings: Mapping[str, Any],
    *,
    input_file: str | Path,
    output_file: str | Path,
    device: torch.device | str | None = None,
) -> tuple[LikelihoodBase, dict[str, list[float]]]:
    """Train from a prediction HDF5 file containing ``predictions`` and ``labels``.

    Predictions are theta_obs and labels are theta_true.  Both datasets must
    have shape ``(N, M)`` and are split along N into training and validation.
    """
    import h5py

    with h5py.File(input_file, "r") as handle:
        theta_obs = torch.as_tensor(handle["predictions"][:], dtype=torch.float32)
        theta_true = torch.as_tensor(handle["labels"][:], dtype=torch.float32)
    LikelihoodBase._validate_pairs(theta_obs, theta_true, "input")

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
    figure = plot_likelihood_fit(model, theta_obs, theta_true)
    plot_file = Path(output_file).with_suffix(".png")
    figure.savefig(plot_file, bbox_inches="tight")
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
    return train_likelihood(settings, input_file=input_file, output_file=output_file, device=device)
