"""Plotting helpers for training diagnostics."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .utils.logger import get_logger

LOGGER = get_logger(__file__)

def parameter_names_from_physics_model(physics_model: Any) -> list[str]:
    """Return target parameter names from ``OntheflyPhysicsModelLinear.params``."""
    return [str(name) for name in physics_model.params]


def plot_targets_vs_predictions(
    targets: np.ndarray,
    predictions: np.ndarray,
    parameter_names: Sequence[str] | None = None,
):
    """Create a square-grid 2D histogram figure of targets versus predictions."""
    if targets.shape != predictions.shape:
        raise ValueError(f"targets and predictions must have the same shape, got {targets.shape} and {predictions.shape}.")
    if targets.ndim != 2:
        raise ValueError(f"targets and predictions must be 2D arrays, got {targets.ndim} dimensions.")

    import matplotlib.pyplot as plt

    names = list(parameter_names or [])
    num_targets = targets.shape[1]
    ncols = math.ceil(np.sqrt(num_targets))
    nrows = math.ceil(num_targets / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).reshape((nrows, ncols))

    for i in range(num_targets):
        row, col = divmod(i, ncols)
        ax = axes[row, col]
        x = targets[:, i]
        y = predictions[:, i]
        # ax.hist2d(x, y, bins=32, cmap="turbo", cmin=1)
        bins_x_edges = np.linspace(x.min(), x.max(), 50)
        bins_y_edges = np.linspace(y.min(), y.max(), 50)
        bins_x_centers = (bins_x_edges[:-1] + bins_x_edges[1:]) / 2
        bins_y_centers = (bins_y_edges[:-1] + bins_y_edges[1:]) / 2
        h = np.histogram2d(x, y, bins=(bins_x_edges, bins_y_edges))[0]
 
        LOGGER.debug(f"Target     {i:>2d}: min={x.min():.6e}, max={x.max():.6e}, mean={x.mean():.6e}, std={x.std():.6e} h_sum={h.sum()}")
        LOGGER.debug(f"Prediction {i:>2d}: min={y.min():.6e}, max={y.max():.6e}, mean={y.mean():.6e}, std={y.std():.6e} h_sum={h.sum()}")

        ax.pcolormesh(bins_x_centers, bins_y_centers, h.T, cmap="turbo", vmin=1)
        name = names[i] if i < len(names) else f"Target {i}"
        ax.set_xlabel(f"Target: {name}")
        ax.set_ylabel(f"Prediction: {name}")
        ax.plot([x.min(), x.max()], [x.min(), x.max()], "r--", lw=1)
        ax.set_ylim(y.min(), y.max())
        ax.set_xlim(x.min(), x.max())


    for i in range(num_targets, nrows * ncols):
        row, col = divmod(i, ncols)
        axes[row, col].axis("off")

    fig.tight_layout()
    return fig


def plot_evaluation_file(path: str | Path, parameter_names: Sequence[str] | None = None):
    """Create a targets-vs-predictions figure from an evaluation HDF5 file."""
    import h5py

    with h5py.File(path, "r") as handle:
        targets = handle["targets"][:]
        predictions = handle["predictions"][:]
    return plot_targets_vs_predictions(targets, predictions, parameter_names)
