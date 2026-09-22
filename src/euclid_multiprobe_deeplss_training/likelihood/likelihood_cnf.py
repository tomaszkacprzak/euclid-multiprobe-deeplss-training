"""Conditional normalizing flow trained with flow matching."""

from __future__ import annotations

import math

import torch
from torch import nn
from torchdiffeq import odeint

from .likelihood_base import LikelihoodBase


class ConditionalNormalizingFlowFM(LikelihoodBase):
    """Model ``p(theta_obs | theta_true)`` with a probability-flow ODE.

    Training uses conditional flow matching along straight paths between a
    standard-normal base sample and an observation.  Likelihood evaluation
    integrates the learned ODE backwards with :func:`torchdiffeq.odeint` and
    accumulates its exact divergence using autograd.
    """

    def __init__(
        self,
        num_parameters: int,
        *,
        num_layers: int = 2,
        hidden_dim: int = 64,
        ode_steps: int = 24,
    ) -> None:
        super().__init__()
        if min(num_parameters, num_layers, hidden_dim, ode_steps) <= 0:
            raise ValueError("All CNF dimensions and ode_steps must be positive.")
        self.num_parameters = num_parameters
        self.ode_steps = ode_steps

        layers: list[nn.Module] = []
        input_dim = 2 * num_parameters + 1
        for _ in range(num_layers):
            layers.extend((nn.Linear(input_dim, hidden_dim), nn.SiLU()))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, num_parameters))
        self.velocity = nn.Sequential(*layers)

    def vector_field(
        self, state: torch.Tensor, time: torch.Tensor, theta_true: torch.Tensor
    ) -> torch.Tensor:
        """Evaluate the conditional time-dependent velocity field."""
        if time.ndim == 0:
            time = time.expand(state.shape[0], 1)
        elif time.ndim == 1:
            time = time[:, None]
        elif time.shape[0] == 1:
            time = time.expand(state.shape[0], 1)
        return self.velocity(torch.cat((state, theta_true, time), dim=-1))

    def training_loss(self, theta_obs: torch.Tensor, theta_true: torch.Tensor) -> torch.Tensor:
        """Return the conditional flow-matching regression objective."""
        self._validate_inputs(theta_obs, theta_true)
        base = torch.randn_like(theta_obs)
        time = torch.rand(theta_obs.shape[0], 1, device=theta_obs.device, dtype=theta_obs.dtype)
        state = (1.0 - time) * base + time * theta_obs
        target_velocity = theta_obs - base
        return (self.vector_field(state, time, theta_true) - target_velocity).square().mean()

    def log_likelihood(self, theta_obs: torch.Tensor, theta_true: torch.Tensor) -> torch.Tensor:
        """Evaluate log ``p(theta_obs | theta_true)`` and return shape ``(N,)``."""
        self._validate_inputs(theta_obs, theta_true)
        # Divergence requires autograd even when callers evaluate under
        # torch.no_grad(), as the shared fit loop does for validation.
        with torch.enable_grad():
            density_change = torch.zeros(
                theta_obs.shape[0], device=theta_obs.device, dtype=theta_obs.dtype
            )
            integration_times = torch.linspace(
                1.0,
                0.0,
                self.ode_steps + 1,
                device=theta_obs.device,
                dtype=theta_obs.dtype,
            )

            def augmented_dynamics(
                time: torch.Tensor, augmented_state: tuple[torch.Tensor, torch.Tensor]
            ) -> tuple[torch.Tensor, torch.Tensor]:
                state, _ = augmented_state
                state = state.detach().requires_grad_(True)
                velocity = self.vector_field(state, time, theta_true)
                divergence = torch.zeros_like(density_change)
                for coordinate in range(self.num_parameters):
                    gradient = torch.autograd.grad(
                        velocity[:, coordinate].sum(), state, retain_graph=True
                    )[0]
                    divergence = divergence + gradient[:, coordinate]
                # Along the probability-flow ODE, d(log p) / dt = -div(v).
                # Integrating this augmented state from data time 1 to base
                # time 0 yields the amount subtracted from the base density.
                return velocity.detach(), -divergence.detach()

            states, density_changes = odeint(
                augmented_dynamics,
                (theta_obs.detach(), density_change),
                integration_times,
                method="euler",
            )
            base_state = states[-1]
            density_change = density_changes[-1]

        log_base = -0.5 * (
            base_state.square().sum(dim=-1) + self.num_parameters * math.log(2.0 * math.pi)
        )
        return log_base - density_change

    def _validate_inputs(self, theta_obs: torch.Tensor, theta_true: torch.Tensor) -> None:
        self._validate_pairs(theta_obs, theta_true, "likelihood")
        if theta_obs.shape[1] != self.num_parameters:
            raise ValueError(f"inputs must have shape (N, {self.num_parameters}).")


LikelihoodCNFFM = ConditionalNormalizingFlowFM
