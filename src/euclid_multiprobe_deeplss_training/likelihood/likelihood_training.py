"""Configuration and file interface for conditional likelihood training."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any
from tqdm import tqdm
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
    prior_bounds: torch.Tensor,
    *,
    num_samples: int = 1000,
    burn_in: int = 200,
    proposal_scale: float = 0.05,
    seed: int = 42,
) -> torch.Tensor:
    """Draw one posterior chain in min-max transformed parameter space.

    ``theta_obs`` and the sampled ``theta_true`` values are scaled to ``[0, 1]``
    using the physical box-prior bounds before evaluating the likelihood. The
    returned samples are transformed back to the original parameter ranges.
    """
    theta_obs = torch.as_tensor(theta_obs)
    prior_bounds = torch.as_tensor(prior_bounds)
    if theta_obs.ndim != 1 or prior_bounds.shape != (theta_obs.numel(), 2):
        raise ValueError("theta_obs must have shape (M,) and prior_bounds must have shape (M, 2).")
    if not torch.all(prior_bounds[:, 0] < prior_bounds[:, 1]):
        raise ValueError("Every prior lower bound must be smaller than its upper bound.")
    if num_samples <= 0 or burn_in < 0:
        raise ValueError("num_samples must be positive and burn_in must be non-negative.")
    if proposal_scale <= 0:
        raise ValueError("proposal_scale must be positive.")

    parameter = next(model.parameters())
    device, dtype = parameter.device, parameter.dtype
    bounds = prior_bounds.to(device=device, dtype=dtype)
    lower, upper = bounds.unbind(dim=1)
    width = upper - lower
    observation = ((theta_obs.to(device=device, dtype=dtype) - lower) / width).unsqueeze(0)
    proposal_std = torch.full_like(lower, proposal_scale)
    current = ((theta_init.to(device=device, dtype=dtype) - lower) / width).unsqueeze(0)
    generator = torch.Generator(device=device).manual_seed(seed)

    model.eval()
    with torch.no_grad():
        current_log_probability = model.log_likelihood(observation, current)[0]
        chain = []
        accepted = 0
        for step in tqdm(range(burn_in + num_samples)):
            proposal = current + proposal_std * torch.randn(current.shape, device=device, dtype=dtype, generator=generator)
            if torch.all((proposal >= 0) & (proposal <= 1)):
                proposal_log_probability = model.log_likelihood(observation, proposal.unsqueeze(0))[0]
                log_acceptance = proposal_log_probability - current_log_probability
                if torch.log(torch.rand((), device=device, dtype=dtype, generator=generator)) < log_acceptance:
                    current = proposal
                    current_log_probability = proposal_log_probability
                    accepted += 1
                    if step >= burn_in:
                        chain.append(current.clone())

    LOGGER.info(f"Metropolis-Hastings acceptance rate: {accepted / (burn_in + num_samples):.3f}")
    transformed_samples = torch.stack(chain)
    return (lower + width * transformed_samples).cpu()


def plot_posterior_samples(samples: torch.Tensor, prior_bounds: torch.Tensor, theta_true: torch.Tensor):
    """Plot one marginal posterior histogram for every parameter dimension."""
    import matplotlib.pyplot as plt

    if samples.ndim != 2 or prior_bounds.shape != (samples.shape[1], 2):
        raise ValueError("samples must have shape (N, M) and prior_bounds must have shape (M, 2).")
    figure, axes = plt.subplots(1, samples.shape[1], figsize=(4 * samples.shape[1], 3), squeeze=False)
    for index, axis in enumerate(axes[0]):
        bounds = prior_bounds[index].detach().cpu().tolist()
        axis.hist(samples[:, index].numpy(), bins=40, range=bounds)
        axis.set_xlabel(f"Parameter {index}")
        axis.set_ylabel("Samples")
        axis.axvline(theta_true[index], color='red', linestyle='--')
    figure.tight_layout()
    return figure

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
    prior_bounds: torch.Tensor | None = None,
) -> tuple[LikelihoodBase, dict[str, list[float]]]:
    """Train from a prediction HDF5 file containing ``predictions`` and ``labels``.

    Predictions are theta_obs and labels are theta_true.  Both datasets must
    have shape ``(N, M)`` and are split along N into training and validation.
    """
    import h5py, numpy as np

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
    if prior_bounds is not None:

        # find an observation that is closest to the mean
        ind_obs = np.argmin(np.linalg.norm(theta_obs - theta_obs.mean(dim=0, keepdim=True), axis=1))
        theta_obs_select = theta_obs[ind_obs]
        theta_true_select = theta_true[ind_obs]
        LOGGER.info(f"Selected observation {theta_obs_select} true {theta_true_select}")

        print(f'theta_obs_select {theta_obs_select} type {type(theta_obs_select)} dtype {theta_obs_select.dtype} shape {theta_obs_select.shape}')
        print(f'theta_true_select {theta_true_select} type {type(theta_true_select)} dtype {theta_true_select.dtype} shape {theta_true_select.shape}')

        # run MCMC for the selected observation
        samples = sample_posterior_metropolis_hastings(
            model,
            theta_obs_select.to(device),
            theta_true_select.to(device),
            prior_bounds,
            num_samples=int(settings.get("mcmc_num_samples", 100000)),
            burn_in=int(settings.get("mcmc_burn_in", 1000)),
            proposal_scale=float(settings.get("mcmc_proposal_scale", 0.05)),
            seed=int(settings.get("seed", 42)),
        )
        samples_file = Path(output_file).with_name(f"{Path(output_file).stem}_samples.h5")
        with h5py.File(samples_file, "w") as handle:
            handle.create_dataset("samples", data=samples.numpy())
            handle.create_dataset("theta_obs", data=theta_obs.mean(dim=0).numpy())
        LOGGER.info(f"Saved posterior samples to {samples_file}")
        
        posterior_figure = plot_posterior_samples(samples, prior_bounds, theta_true_select)
        posterior_plot_file = Path(output_file).with_name(f"{Path(output_file).stem}_samples.png")
        posterior_figure.savefig(posterior_plot_file, bbox_inches="tight")
        plt.close(posterior_figure)
        LOGGER.info(f"Saved posterior samples plot to {posterior_plot_file}")

        chain_figure = plot_chain(samples)
        chain_plot_file = Path(output_file).with_name(f"{Path(output_file).stem}_chain.png")
        chain_figure.savefig(chain_plot_file, bbox_inches="tight")
        plt.close(chain_figure)
        LOGGER.info(f"Saved chain plot to {chain_plot_file}")



    return model, history

def plot_chain(samples):
    import matplotlib.pyplot as plt
    num_parameters = samples.shape[1]
    figure, axes = plt.subplots(1, num_parameters, figsize=(4 * num_parameters, 3), squeeze=False)
    for index, axis in enumerate(axes[0]):
        axis.plot(samples[:, index].numpy(), 'o-', label=f"Parameter {index}")
        axis.legend()
        axis.set_xlabel(f"Step")
        axis.set_ylabel(f"Parameter {index}")
    figure.tight_layout()
    return figure

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

    training_settings = raw_config.get("training")
    forward_model = raw_config.get("forward_model")
    if not isinstance(training_settings, Mapping) or not isinstance(forward_model, Mapping):
        raise ValueError("Posterior sampling requires the 'training' and 'forward_model' configuration sections.")
    from ..training import load_physics_model_class

    physics_model_class = load_physics_model_class(str(training_settings["physics_model"]))
    physics_args = dict(training_settings.get("physics_model_args", {}))
    physics_args.setdefault("num_samples_prior", 1)
    physics_args.setdefault("nside", forward_model.get("analysis", {}).get("n_side"))
    physics_model = physics_model_class(
        forward_model,
        scalers=False,
        seed=int(settings.get("seed", 42)),
        device=device or settings.get("device"),
        **physics_args,
    )
    prior_bounds = torch.tensor([physics_model.priors[name] for name in physics_model.params], dtype=torch.float32)
    return train_likelihood(
        settings,
        input_file=input_file,
        output_file=output_file,
        device=device,
        prior_bounds=prior_bounds,
    )
