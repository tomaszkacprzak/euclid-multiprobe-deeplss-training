"""Configuration and file interface for conditional likelihood training."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torchebm.core import BaseModel
from torchebm.samplers import HamiltonianMonteCarlo

from ..utils.config import ConfigPaths, config_paths, load_config
from ..utils.logger import get_logger
from .likelihood_base import LikelihoodBase
from .likelihood_cnf import ConditionalNormalizingFlowFM
from .likelihood_mdn import GaussianMixtureMDN

LOGGER = get_logger(__file__)


class _BatchedPosteriorEnergy(BaseModel):
    """Unconstrained posterior energy for a batch of conditioned observations."""

    def __init__(
        self,
        likelihood: LikelihoodBase,
        observations: torch.Tensor,
        prior_bounds: torch.Tensor,
    ) -> None:
        super().__init__(dtype=observations.dtype, device=observations.device)
        self.likelihood = likelihood
        self.register_buffer("observations", observations)
        self.register_buffer("lower", prior_bounds[:, 0])
        self.register_buffer("width", prior_bounds[:, 1] - prior_bounds[:, 0])

    def forward(self, unconstrained: torch.Tensor) -> torch.Tensor:
        # Sampling in logit space gives HMC a smooth energy at the edges of the
        # uniform prior.  The Jacobian makes the transformed density exact.
        theta = self.lower + self.width * torch.sigmoid(unconstrained)
        log_jacobian = (torch.log(self.width) + torch.nn.functional.logsigmoid(unconstrained) + torch.nn.functional.logsigmoid(-unconstrained)).sum(
            dim=-1
        )
        return -self.likelihood(self.observations, theta) - log_jacobian

    def to_parameters(self, unconstrained: torch.Tensor) -> torch.Tensor:
        """Map unconstrained HMC states back inside the uniform-prior bounds."""
        return self.lower + self.width * torch.sigmoid(unconstrained)


def sample_posteriors(
    model: LikelihoodBase,
    observations: torch.Tensor,
    prior_bounds: torch.Tensor,
    *,
    num_steps: int = 1000,
    burn_in: int = 200,
    step_size: float = 0.01,
    num_leapfrog_steps: int = 10,
    seed: int = 42,
) -> list[torch.Tensor]:
    """Sample ``p(theta_true | theta_obs)`` with one HMC chain per observation.

    The independent chains are advanced together in a single PyTorch batch. A
    logit transform enforces the uniform ``prior_bounds`` without introducing
    a discontinuous energy at the boundary.
    """
    if observations.ndim != 2 or prior_bounds.shape != (observations.shape[1], 2):
        raise ValueError("observations must have shape (N, M) and prior_bounds must have shape (M, 2).")
    if not torch.all(prior_bounds[:, 0] < prior_bounds[:, 1]):
        raise ValueError("Every prior lower bound must be smaller than its upper bound.")
    if not 0 <= burn_in < num_steps:
        raise ValueError("mcmc_burn_in must be non-negative and smaller than mcmc_num_steps.")
    if step_size <= 0:
        raise ValueError("mcmc_step_size must be positive.")
    if num_leapfrog_steps <= 0:
        raise ValueError("mcmc_num_leapfrog_steps must be positive.")

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    observations = observations.to(device=device, dtype=dtype)
    prior_bounds = prior_bounds.to(device=device, dtype=dtype)
    energy = _BatchedPosteriorEnergy(model, observations, prior_bounds)

    generator = torch.Generator(device=device).manual_seed(seed)
    initial_unit = torch.rand(observations.shape, device=device, dtype=dtype, generator=generator)
    epsilon = torch.finfo(dtype).eps
    initial_state = torch.logit(initial_unit.clamp(min=epsilon, max=1.0 - epsilon))

    model.eval()
    LOGGER.info(f"Initializing HMC sampler with step size {step_size} and {num_leapfrog_steps} leapfrog steps")
    sampler = HamiltonianMonteCarlo(
        model=energy,
        step_size=step_size,
        n_leapfrog_steps=num_leapfrog_steps,
        dtype=dtype,
        device=device,
    )
    LOGGER.info(f"Sampling {len(observations)} posteriors with {num_steps} steps on {device}")
    trajectory, diagnostics = sampler.sample(
        x=initial_state,
        n_steps=num_steps,
        return_trajectory=True,
        generator=generator,
        return_diagnostics=True,
    )

    a = diagnostics["acceptance_rate"]
    LOGGER.info("acceptance rate: mean = %f, min = %f, max = %f", a.mean().item(), a.min().item(), a.max().item())

    # return parameters
    parameter_samples = energy.to_parameters(trajectory[:, burn_in:])
    return [samples.cpu() for samples in parameter_samples]


def plot_posterior_samples(samples: list[torch.Tensor], labels: torch.Tensor, prior_bounds: torch.Tensor):
    """Plot marginal posterior histograms using bins spanning each prior."""
    import matplotlib.pyplot as plt

    if not samples:
        raise ValueError("At least one posterior sample set is required.")
    if labels.ndim != 2:
        raise ValueError("labels must have shape (N, M).")
    rows = min(4, len(samples))
    num_parameters = labels.shape[1]
    if prior_bounds.shape != (num_parameters, 2):
        raise ValueError("prior_bounds must have shape (M, 2).")
    if not torch.all(prior_bounds[:, 0] < prior_bounds[:, 1]):
        raise ValueError("Every prior lower bound must be smaller than its upper bound.")

    # Reuse a single set of edges for every observation in a parameter column,
    # rather than allowing matplotlib to infer different edges from each sample.
    bin_edges = [
        np.linspace(float(prior_bounds[column, 0]), float(prior_bounds[column, 1]), 41) for column in range(num_parameters)
    ]
    figure, axes = plt.subplots(rows, num_parameters, figsize=(4 * num_parameters, 3 * rows), squeeze=False)
    for row in range(rows):
        for column in range(num_parameters):
            s = samples[row][:, column].numpy()
            axis = axes[row, column]
            axis.hist(s, bins=bin_edges[column], density=False, label=f"num samples: {len(s)}")
            # axis.axvline(float(labels[row, column]), color="tab:red", linewidth=2, label="True value")
            axis.set_xlabel(f"Parameter {column}")
            axis.set_ylabel("Density")
            axis.legend(loc="upper right")
            # axis.set_xlim(prior_bounds[column, 0], prior_bounds[column, 1])
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
        log_likelihood = model(predictions_device, labels_device).detach().cpu().numpy()

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
            grid_log_likelihood = model(grid_predictions, grid_labels).reshape(label_grid.shape).cpu().numpy()

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
    num_observations: int = 0,
    prior_bounds: torch.Tensor | None = None,
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

    if num_observations < 0 or num_observations > len(theta_obs):
        raise ValueError("num_observations must be between zero and the size of the predictions dataset.")
    if num_observations:
        if prior_bounds is None:
            raise ValueError("prior_bounds are required when posterior sampling is enabled.")

        LOGGER.info(f"Sampling posteriors with {num_observations} observations")

        selected = order[:num_observations]
        observations = theta_obs[selected]
        selected_labels = theta_true[selected]
        samples = sample_posteriors(
            model,
            observations,
            prior_bounds,
            num_steps=int(settings.get("mcmc_num_steps", 10000)),
            burn_in=int(settings.get("mcmc_burn_in", 200)),
            step_size=float(settings.get("mcmc_step_size", 0.01)),
            num_leapfrog_steps=int(settings.get("mcmc_num_leapfrog_steps", 10)),
            seed=int(settings.get("seed", 42)),
        )
        samples_file = Path(settings.get("samples_file", Path(output_file).with_name(f"{Path(output_file).stem}_samples.h5")))
        samples_file.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(samples_file, "w") as handle:
            for index, posterior_samples in enumerate(samples):
                handle.create_dataset(f"samples{index:04d}", data=posterior_samples.numpy())
        LOGGER.info(f"Saved posterior samples to {samples_file}")

        posterior_figure = plot_posterior_samples(samples, selected_labels, prior_bounds)
        posterior_plot_file = Path(settings.get("samples_plot_file", Path(output_file).with_name(f"{Path(output_file).stem}_samples.png")))
        posterior_plot_file.parent.mkdir(parents=True, exist_ok=True)
        posterior_figure.savefig(posterior_plot_file, bbox_inches="tight")
        plt.close(posterior_figure)
        LOGGER.info(f"Saved posterior sampling plot to {posterior_plot_file}")
    return model, history


def train_likelihood_from_config(
    config_path: ConfigPaths,
    *,
    input_file: str | Path,
    output_file: str | Path,
    device: torch.device | str | None = None,
    num_observations: int | None = None,
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
    sample_count = int(settings.get("num_observations", 0) if num_observations is None else num_observations)
    prior_bounds = None
    if sample_count:
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
        num_observations=sample_count,
        prior_bounds=prior_bounds,
    )
