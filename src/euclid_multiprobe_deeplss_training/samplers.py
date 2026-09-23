"""Batch posterior samplers and their common output helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torchebm.core import BaseModel
from tqdm import tqdm

from .likelihood.likelihood_base import LikelihoodBase
from .utils.logger import get_logger

LOGGER = get_logger(__file__)


class ConditionalLikelihoodEnergy(BaseModel):
    """TorchEBM energy for a likelihood with a uniform unit-box prior.

    HMC evolves an unconstrained variable and this model maps it into the unit
    box with a sigmoid.  The negative log-Jacobian is part of the energy, which
    both preserves the intended density under the change of variables and
    increasingly penalizes trajectories that move towards either boundary.
    """

    def __init__(self, likelihood_model: LikelihoodBase, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.likelihood_model = likelihood_model

    @staticmethod
    def to_parameter_space(unconstrained_parameters: torch.Tensor) -> torch.Tensor:
        """Map unconstrained HMC coordinates into the open unit box."""
        return torch.sigmoid(unconstrained_parameters)

    @staticmethod
    def to_unconstrained_space(parameters: torch.Tensor) -> torch.Tensor:
        """Map unit-box parameters to finite unconstrained coordinates."""
        epsilon = torch.finfo(parameters.dtype).eps
        return torch.logit(parameters.clamp(min=epsilon, max=1 - epsilon))

    def forward(self, unconstrained_parameters: torch.Tensor, *, theta_obs: torch.Tensor) -> torch.Tensor:
        """Return a finite transformed-space negative log posterior."""
        parameters = self.to_parameter_space(unconstrained_parameters)
        log_likelihood = self.likelihood_model.log_likelihood(theta_obs, parameters)
        log_jacobian = (F.logsigmoid(unconstrained_parameters) + F.logsigmoid(-unconstrained_parameters)).sum(dim=-1)
        energy = -log_likelihood - log_jacobian
        finite_limit = min(1e10, torch.finfo(energy.dtype).max)
        return torch.nan_to_num(energy, nan=finite_limit, posinf=finite_limit, neginf=-finite_limit)


class BaseBatchSampler(ABC):
    """Common interface for sampling a batch of conditional posteriors.

    Parameters are standardized to the unit box by the surrounding pipeline,
    so every sampler applies an implicit uniform prior on ``[0, 1]``.
    """

    def __init__(self, likelihood_model: LikelihoodBase, num_steps: int) -> None:
        if num_steps <= 0:
            raise ValueError("num_steps must be positive.")
        self.likelihood_model = likelihood_model
        self.num_steps = num_steps

    @abstractmethod
    def sample(self, theta_obs: torch.Tensor, theta_init: torch.Tensor) -> torch.Tensor:
        """Return chains shaped ``(num_observations, num_steps, num_parameters)``."""

    def _prepare_inputs(self, theta_obs: torch.Tensor, theta_init: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        theta_obs = torch.as_tensor(theta_obs)
        theta_init = torch.as_tensor(theta_init)
        if theta_obs.ndim != 2 or theta_init.shape != theta_obs.shape:
            raise ValueError("theta_obs and theta_init must have shape (num_observations, num_parameters).")
        if theta_obs.shape[0] == 0:
            raise ValueError("theta_obs and theta_init must contain at least one observation.")
        if not torch.all((theta_init >= 0) & (theta_init <= 1)):
            raise ValueError("theta_init must be inside the standardized [0, 1] prior.")

        parameter = next(self.likelihood_model.parameters())
        return (
            theta_obs.to(device=parameter.device, dtype=parameter.dtype),
            theta_init.to(device=parameter.device, dtype=parameter.dtype),
        )

    @staticmethod
    def plot_likelihood_samples(samples: torch.Tensor, theta_true: torch.Tensor):
        """Plot one marginal posterior histogram for every parameter dimension."""
        import matplotlib.pyplot as plt

        samples = torch.as_tensor(samples).detach().cpu()
        theta_true = torch.as_tensor(theta_true).detach().cpu()
        if samples.ndim == 2:
            samples = samples.unsqueeze(0)
        if theta_true.ndim == 1:
            theta_true = theta_true.unsqueeze(0)
        if samples.ndim != 3 or theta_true.shape != (samples.shape[0], samples.shape[2]):
            raise ValueError("samples and theta_true must have shapes (N, S, M) and (N, M).")

        num_observations, _, num_parameters = samples.shape
        figure, axes = plt.subplots(
            num_observations,
            num_parameters,
            figsize=(4 * num_parameters, 3 * num_observations),
            squeeze=False,
        )
        for i in range(num_observations):
            for j in range(num_parameters):
                axes[i, j].hist(samples[i, :, j].numpy(), bins=40, range=(0, 1))
                axes[i, j].set_xlabel(f"Parameter {j}")
                axes[i, j].set_ylabel("Samples")
                axes[i, j].axvline(theta_true[i, j], color="red", linestyle="--")
        figure.subplots_adjust(bottom=0.12, right=0.9, hspace=0.45, wspace=0.3)
        return figure

    # Keep the old terminology available to callers while putting the routine
    # on the sampler interface.
    plot_posterior_samples = plot_likelihood_samples

    @staticmethod
    def plot_triangle_chain(
        samples: torch.Tensor,
        parameter_names: Sequence[str],
        priors: Mapping[str, Sequence[float]],
    ) -> tuple[Any, Any]:
        """Plot one unit-box chain in the physical coordinates of its priors."""
        import numpy as np
        from trianglechain import TriangleChain

        chain = torch.as_tensor(samples).detach().cpu().numpy().astype(float, copy=False)
        labels = [str(name) for name in parameter_names]
        if chain.ndim != 2:
            raise ValueError("samples must have shape (num_steps, num_parameters).")
        if len(labels) != chain.shape[1]:
            raise ValueError("There must be one parameter name per column of samples.")
        if not np.isfinite(chain).all():
            raise ValueError("samples contain NaN or infinite values.")

        try:
            bounds = np.asarray([priors[name] for name in labels], dtype=float)
        except KeyError as error:
            raise ValueError(f"No prior is defined for parameter {error.args[0]!r}.") from error
        if bounds.shape != (chain.shape[1], 2):
            raise ValueError("Each parameter prior must contain exactly a lower and upper bound.")
        if not np.isfinite(bounds).all() or np.any(bounds[:, 1] <= bounds[:, 0]):
            raise ValueError("Parameter priors must have finite, increasing bounds.")

        physical_chain = bounds[:, 0] + chain * (bounds[:, 1] - bounds[:, 0])
        triangle = TriangleChain(labels=labels, fill=True, de_kwargs={"levels": [0.68, 0.95]})
        return triangle.contour_cl(
            physical_chain,
            show_values=True,
            bestfit_method="median",
            levels_method="percentile",
            credible_interval=0.68,
        )

    @staticmethod
    def plot_chain(samples: torch.Tensor):
        """Plot each parameter of a single chain against sampling step."""
        import matplotlib.pyplot as plt

        samples = torch.as_tensor(samples).detach().cpu()
        if samples.ndim != 2:
            raise ValueError("samples must have shape (num_steps, num_parameters).")
        num_parameters = samples.shape[1]
        figure, axes = plt.subplots(1, num_parameters, figsize=(4 * num_parameters, 3), squeeze=False)
        for index, axis in enumerate(axes[0]):
            axis.plot(samples[:, index].numpy(), "o-", label=f"Parameter {index}")
            axis.legend()
            axis.set_xlabel("Step")
            axis.set_ylabel(f"Parameter {index}")
        figure.tight_layout()
        return figure

    @staticmethod
    def save_chains(path: str | Path, samples: torch.Tensor, theta_obs: torch.Tensor) -> None:
        """Save posterior chains and their conditioning observations as HDF5."""
        import h5py

        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(output, "w") as handle:
            handle.create_dataset("samples", data=torch.as_tensor(samples).detach().cpu().numpy())
            handle.create_dataset("theta_obs", data=torch.as_tensor(theta_obs).detach().cpu().numpy())
        LOGGER.info(f"Saved posterior samples to {output}")

    save_samples = save_chains


class MetropolisHastingsBatchSampler(BaseBatchSampler):
    """Random-walk Metropolis--Hastings sampler for independent observations."""

    def __init__(
        self,
        likelihood_model: LikelihoodBase,
        num_steps: int,
        *,
        burn_in: int = 200,
        proposal_scale: float = 0.05,
        seed: int = 42,
    ) -> None:
        super().__init__(likelihood_model, num_steps)
        if burn_in < 0:
            raise ValueError("burn_in must be non-negative.")
        if proposal_scale <= 0:
            raise ValueError("proposal_scale must be positive.")
        self.burn_in = burn_in
        self.proposal_scale = proposal_scale
        self.seed = seed

    def sample(self, theta_obs: torch.Tensor, theta_init: torch.Tensor) -> torch.Tensor:
        observations, initial_parameters = self._prepare_inputs(theta_obs, theta_init)
        device, dtype = initial_parameters.device, initial_parameters.dtype
        generator = torch.Generator(device=device).manual_seed(self.seed)

        self.likelihood_model.eval()
        with torch.no_grad():
            chains = []
            LOGGER.info(f"running MCMC for {len(observations)} observations")
            for observation_index, (observation, current) in enumerate(zip(observations, initial_parameters, strict=True)):
                observation = observation.unsqueeze(0)
                current = current.unsqueeze(0)
                current_log_probability = self.likelihood_model.log_likelihood(observation, current)[0]
                chain = []
                accepted = 0
                total_steps = self.burn_in + self.num_steps
                for step in tqdm(range(total_steps), desc=f"Observation {observation_index + 1}/{len(observations)}"):
                    proposal = current + self.proposal_scale * torch.randn(
                        current.shape, device=device, dtype=dtype, generator=generator
                    )
                    if torch.all((proposal >= 0) & (proposal <= 1)):
                        proposal_log_probability = self.likelihood_model.log_likelihood(observation, proposal)[0]
                        if torch.log(torch.rand((), device=device, dtype=dtype, generator=generator)) < (
                            proposal_log_probability - current_log_probability
                        ):
                            current = proposal
                            current_log_probability = proposal_log_probability
                            accepted += 1
                            if step >= self.burn_in:
                                chain.append(current.squeeze(0).clone())
                chains.append(torch.stack(chain))
                LOGGER.info(
                    f"Observation {observation_index + 1}/{len(observations)}: Metropolis-Hastings accepted "
                    f"{accepted}/{total_steps}, acceptance rate {accepted / total_steps:.3e}"
                )
        chain_size_min = min([len(chain) for chain in chains])
        chains = [chain[:chain_size_min] for chain in chains]
        return torch.stack(chains).cpu()


class HamiltonianMonteCarloBatchSampler(BaseBatchSampler):
    """Batched Hamiltonian Monte Carlo using :mod:`torchebm`."""

    def __init__(
        self,
        likelihood_model: LikelihoodBase,
        num_steps: int,
        *,
        burn_in: int = 200,
        step_size: float = 1e-3,
        num_leapfrog_steps: int = 10,
        mass: float | torch.Tensor | None = None,
        seed: int = 42,
    ) -> None:
        super().__init__(likelihood_model, num_steps)
        if burn_in < 0:
            raise ValueError("burn_in must be non-negative.")
        if step_size <= 0 or num_leapfrog_steps <= 0:
            raise ValueError("step_size and num_leapfrog_steps must be positive.")
        self.burn_in = burn_in
        self.step_size = step_size
        self.num_leapfrog_steps = num_leapfrog_steps
        self.mass = mass
        self.seed = seed

    def sample(self, theta_obs: torch.Tensor, theta_init: torch.Tensor) -> torch.Tensor:
        """Evolve all observations as parallel chains in one torchebm call."""
        from torchebm.samplers import HamiltonianMonteCarlo

        observations, initial_parameters = self._prepare_inputs(theta_obs, theta_init)
        energy = ConditionalLikelihoodEnergy(
            self.likelihood_model,
            dtype=initial_parameters.dtype,
            device=initial_parameters.device,
        )

        LOGGER.info(f"Creating HMC sampler with step size {self.step_size}, num leapfrog steps {self.num_leapfrog_steps}, mass {self.mass}")
        sampler = HamiltonianMonteCarlo(
            energy,
            step_size=self.step_size,
            n_leapfrog_steps=self.num_leapfrog_steps,
            mass=self.mass,
            dtype=initial_parameters.dtype,
            device=initial_parameters.device,
        )
        generator = torch.Generator(device=initial_parameters.device).manual_seed(self.seed)
        self.likelihood_model.eval()
        LOGGER.info(f"Sampling {len(initial_parameters)} observations and {self.burn_in + self.num_steps} steps with HMC")
        trajectory, diagnostics = sampler.sample(
            x=energy.to_unconstrained_space(initial_parameters),
            n_steps=self.burn_in + self.num_steps,
            n_samples=len(initial_parameters),
            return_trajectory=True,
            model_kwargs={"theta_obs": observations},
            generator=generator,
            return_diagnostics=True,
        )
        a = diagnostics["acceptance_rate"]
        print("mean acceptance :", a.mean().item())
        print("min acceptance  :", a.min().item())
        print("max acceptance  :", a.max().item())
        samples = energy.to_parameter_space(trajectory[:, self.burn_in :])
        return samples.detach().cpu()


# A concise alias matching the name commonly used for the algorithm.
HMCBatchSampler = HamiltonianMonteCarloBatchSampler
