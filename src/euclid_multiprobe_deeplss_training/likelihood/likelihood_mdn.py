"""Gaussian mixture density network for conditional likelihood modelling."""

from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Categorical, Independent, MixtureSameFamily, Normal

from .likelihood_base import LikelihoodBase


class GaussianMixtureMDN(LikelihoodBase):
    """Model ``p(theta_obs | theta_true)`` as a diagonal Gaussian mixture."""

    def __init__(
        self,
        num_parameters: int,
        *,
        num_components: int = 10,
        num_layers: int = 2,
        hidden_dim: int = 64,
        min_scale: float = 1e-4,
    ) -> None:
        super().__init__()
        if min(num_parameters, num_components, num_layers, hidden_dim) <= 0 or min_scale <= 0:
            raise ValueError("All MDN dimensions and min_scale must be positive.")
        self.num_parameters = num_parameters
        self.num_components = num_components
        self.min_scale = min_scale

        # The network maps (N, M) true parameters to K weights, K*M means, and K*M scales.
        layers: list[nn.Module] = []
        input_dim = num_parameters
        for _ in range(num_layers):
            layers.extend((nn.Linear(input_dim, hidden_dim), nn.ReLU()))
            input_dim = hidden_dim
        output_dim = num_components * (1 + 2 * num_parameters)
        layers.append(nn.Linear(input_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def mixture(self, theta_true: torch.Tensor) -> MixtureSameFamily:
        """Construct the batch of K-component distributions for ``(N, M)`` input."""
        if theta_true.ndim != 2 or theta_true.shape[1] != self.num_parameters:
            raise ValueError(f"theta_true must have shape (N, {self.num_parameters}).")
        output = self.network(theta_true)
        n = theta_true.shape[0]
        k, m = self.num_components, self.num_parameters
        logits = output[:, :k]  # (N, K)
        means = output[:, k : k + k * m].reshape(n, k, m)  # (N, K, M)
        raw_scales = output[:, k + k * m :].reshape(n, k, m)  # (N, K, M)
        scales = torch.nn.functional.softplus(raw_scales) + self.min_scale
        components = Independent(Normal(means, scales), 1)
        return MixtureSameFamily(Categorical(logits=logits), components)

    def log_likelihood(self, theta_obs: torch.Tensor, theta_true: torch.Tensor) -> torch.Tensor:
        """Evaluate log ``p(theta_obs | theta_true)`` and return shape ``(N,)``."""
        self._validate_pairs(theta_obs, theta_true, "likelihood")
        if theta_obs.shape[1] != self.num_parameters:
            raise ValueError(f"inputs must have shape (N, {self.num_parameters}).")
        return self.mixture(theta_true).log_prob(theta_obs)


# Short alias used by callers that refer to the implementation simply as an MDN.
LikelihoodMDN = GaussianMixtureMDN
