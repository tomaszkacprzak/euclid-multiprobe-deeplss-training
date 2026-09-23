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


def sample_posterior_metropolis_hastings(
    model: LikelihoodBase,
    theta_obs: torch.Tensor,
    theta_init: torch.Tensor,
    *,
    num_samples: int = 1000,
    burn_in: int = 200,
    proposal_scale: float = 0.05,
    seed: int = 42,
) -> torch.Tensor:
    """Backward-compatible convenience wrapper for Metropolis--Hastings."""
    from ..samplers import MetropolisHastingsBatchSampler

    return MetropolisHastingsBatchSampler(
        model,
        num_samples,
        burn_in=burn_in,
        proposal_scale=proposal_scale,
        seed=seed,
    ).sample(theta_obs, theta_init)


def plot_posterior_samples(samples: torch.Tensor, theta_true: torch.Tensor):
    """Backward-compatible wrapper for the sampler histogram helper."""
    from ..samplers import BaseBatchSampler

    return BaseBatchSampler.plot_posterior_samples(samples, theta_true)


def plot_likelihood_fit(
    model: LikelihoodBase,
    predictions: torch.Tensor,
    labels: torch.Tensor,
):
    """Plot the samples and fitted log-likelihood surface for each parameter.

    Each surface varies one label/prediction pair over its observed range while
    holding all other dimensions at their sample means.  This produces a useful
    two-dimensional slice through a multivariate conditional density.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    LikelihoodBase._validate_pairs(predictions, labels, "plot")
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
        label_values = torch.linspace(labels_device[:, index].min(), labels_device[:, index].max(), 100, device=device)
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

    Predictions are theta_obs and labels are theta_true.  Both datasets must
    have shape ``(N, M)`` and are split along N into training and validation.
    """
    import h5py

    with h5py.File(input_file, "r") as handle:
        theta_obs = torch.as_tensor(handle["predictions"][:], dtype=torch.float32)
        theta_true = torch.as_tensor(handle["labels"][:], dtype=torch.float32)
    LikelihoodBase._validate_pairs(theta_obs, theta_true, "input")
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

    # run MCMC for the selected observation
    num_observations = settings.get("mcmc_num_observations", 4)
    num_observations_plot = 4
    if "mcmc_num_samples" in settings:
        
        # find an observation that is closest to the mean
        theta_obs_select = theta_obs[:num_observations]
        theta_true_select = theta_true[:num_observations]

        # Run the configured sampler. Metropolis--Hastings remains the default
        # so existing configuration files retain their previous behaviour.
        from ..samplers import HamiltonianMonteCarloBatchSampler, MetropolisHastingsBatchSampler

        sampler_name = str(settings.get("mcmc_sampler", "metropolis_hastings")).lower().replace("-", "_")
        common_arguments = {
            "burn_in": int(settings.get("mcmc_burn_in", 1000)),
            "seed": int(settings.get("seed", 42)),
        }
        if sampler_name in {"metropolis", "metropolis_hastings", "mh"}:
            sampler = MetropolisHastingsBatchSampler(
                model,
                int(settings.get("mcmc_num_samples", 100000)),
                proposal_scale=float(settings.get("mcmc_proposal_scale", 0.01)),
                **common_arguments,
            )
        elif sampler_name in {"hamiltonian_monte_carlo", "hamiltonian", "hmc"}:
            sampler = HamiltonianMonteCarloBatchSampler(
                model,
                int(settings.get("mcmc_num_samples", 100000)),
                step_size=float(settings.get("mcmc_step_size", 5e-2)),
                num_leapfrog_steps=int(settings.get("mcmc_num_leapfrog_steps", 10)),
                **common_arguments,
            )
        else:
            raise ValueError(f"Unknown mcmc_sampler: {sampler_name!r}.")

        samples = sampler.sample(theta_obs_select, theta_true_select)
        samples_file = Path(output_file).with_name(f"{Path(output_file).stem}_samples.h5")
        # sampler.save_chains(samples_file, samples, theta_obs_select)

        posterior_figure = sampler.plot_likelihood_samples(samples[:num_observations_plot], theta_true_select[:num_observations_plot])
        posterior_plot_file = Path(output_file).with_name(f"{Path(output_file).stem}_{sampler_name}_samples.png")
        posterior_figure.savefig(posterior_plot_file, bbox_inches="tight")
        plt.close(posterior_figure)
        LOGGER.info(f"Saved posterior samples plot to {posterior_plot_file}")

        chain_figure = sampler.plot_chain(samples[0])
        chain_plot_file = Path(output_file).with_name(f"{Path(output_file).stem}_{sampler_name}_chain.png")
        chain_figure.savefig(chain_plot_file, bbox_inches="tight")
        plt.close(chain_figure)
        LOGGER.info(f"Saved chain plot to {chain_plot_file}")

    return model, history


def plot_chain(samples: torch.Tensor):
    """Backward-compatible wrapper for the sampler chain-plot helper."""
    from ..samplers import BaseBatchSampler

    return BaseBatchSampler.plot_chain(samples)


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
