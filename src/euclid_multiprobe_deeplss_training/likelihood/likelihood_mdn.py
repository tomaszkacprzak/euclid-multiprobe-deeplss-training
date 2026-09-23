"""Gaussian mixture density network for conditional likelihood modelling."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .likelihood_base import LikelihoodBase


class GaussianMixtureMDN(LikelihoodBase):
    """Model p(x_obs | x_true) using K full-covariance Gaussians.

    Each component has precision P = L @ L.T, where L is lower triangular
    with a positive diagonal. L is NOT a covariance Cholesky factor.

    Args:
        input_dim: Number of conditioning features in x_true (theta_true).
        output_dim: Number of observed features in x_obs.
        num_components: Number K of Gaussian mixture components.
        hidden_dims: One width repeated num_layers times, or a sequence
            containing exactly num_layers widths.
        num_layers: Number of hidden layers; zero gives linear heads.
        min_diag: Positive floor added to the diagonal of L.

    Inputs must be real floating-point tensors on the same device and
    with the same dtype as the model.
    """

    def __init__(
        self,
        num_parameters: int,
        num_components: int,
        *,
        hidden_dims: int | Sequence[int] = 128,
        num_layers: int = 2,
        min_diag: float = 1e-4,
    ) -> None:
        super().__init__()

        input_dim = output_dim = num_parameters

        for name, value in (
            ("input_dim", input_dim),
            ("output_dim", output_dim),
            ("num_components", num_components),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")

        if (
            isinstance(num_layers, bool)
            or not isinstance(num_layers, int)
            or num_layers < 0
        ):
            raise ValueError("num_layers must be a nonnegative integer.")

        if not math.isfinite(min_diag) or min_diag <= 0:
            raise ValueError("min_diag must be finite and strictly positive.")

        if isinstance(hidden_dims, int):
            if isinstance(hidden_dims, bool) or hidden_dims < 1:
                raise ValueError("hidden_dims must be a positive integer.")
            widths = (hidden_dims,) * num_layers
        else:
            widths = tuple(hidden_dims)

        if len(widths) != num_layers:
            raise ValueError("hidden_dims must contain num_layers widths.")

        if any(
            isinstance(w, bool) or not isinstance(w, int) or w < 1
            for w in widths
        ):
            raise ValueError(
                "Every hidden-layer width must be a positive integer."
            )

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_components = num_components
        self.min_diag = float(min_diag)
        self.num_tril = output_dim * (output_dim + 1) // 2

        # Gaussian normalization constant, computed only once.
        self._log_normalizer = 0.5 * output_dim * math.log(2.0 * math.pi)

        # Shared MLP.
        layers: list[nn.Module] = []
        width = input_dim
        for next_width in widths:
            layers.extend([nn.Linear(width, next_width), nn.SiLU()])
            width = next_width
        self.backbone = nn.Sequential(*layers)

        # Separate heads for mixture logits, means, and packed L entries.
        self.weight_head = nn.Linear(width, num_components)
        self.mean_head = nn.Linear(width, num_components * output_dim)
        self.precision_head = nn.Linear(
            width, num_components * self.num_tril
        )

        # Fixed buffers: created once and moved with model.to(...).
        # Nonpersistent because the constructor recreates them.
        rows, cols = torch.tril_indices(output_dim, output_dim)
        self.register_buffer("_tril_rows", rows, persistent=False)
        self.register_buffer("_tril_cols", cols, persistent=False)
        self.register_buffer("_diag_mask", rows == cols, persistent=False)

    def _predict(self, x_true: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return log_weights (B,K), means (B,K,D), and L (B,K,D,D)."""
        if x_true.ndim != 2 or x_true.shape[1] != self.input_dim:
            raise ValueError(
                f"x_true must have shape (batch, {self.input_dim})."
            )
        if not x_true.is_floating_point():
            raise TypeError("x_true must be a real floating-point tensor.")

        batch = x_true.shape[0]
        k, d = self.num_components, self.output_dim

        features = self.backbone(x_true)

        # Preserve log weights directly for stable likelihood evaluation.
        log_weights = F.log_softmax(
            self.weight_head(features), dim=-1
        )                                                     # (B,K)
        means = self.mean_head(features).reshape(batch, k, d)  # (B,K,D)
        raw_tril = self.precision_head(features).reshape(
            batch, k, self.num_tril
        )                                                     # (B,K,T)

        # Positive diagonal; unrestricted off-diagonal entries.
        packed_tril = torch.where(
            self._diag_mask,
            F.softplus(raw_tril) + self.min_diag,
            raw_tril,
        )

        # This assignment preserves gradients to packed_tril.
        L = packed_tril.new_zeros((batch, k, d, d))
        L[..., self._tril_rows, self._tril_cols] = packed_tril

        return log_weights, means, L

    def forward(self, x_true: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return mixture parameters for x_true of shape (B,input_dim).

        Returns:
            weights: (B,K), normalized over components.
            means:   (B,K,D).
            L:       (B,K,D,D), with precision P = L @ L.transpose(-1,-2).
        """
        log_weights, means, L = self._predict(x_true)
        return log_weights.exp(), means, L

    def log_likelihood(
        self,
        x_obs: Tensor,
        x_true: Tensor,
        *,
        reduction: Literal["none", "mean", "sum"] = "none",
    ) -> Tensor:
        """Compute log p(x_obs[b] | x_true[b]) for each batch entry.

        Args:
            x_true: (B,input_dim).
            x_obs:  (B,output_dim).
            reduction: 'none', 'mean', or 'sum'.

        Returns:
            Shape (B,) for 'none'; a scalar for 'mean' or 'sum'.
        """
        if reduction not in ("none", "mean", "sum"):
            raise ValueError(
                "reduction must be 'none', 'mean', or 'sum'."
            )

        # LikelihoodBase consistently exposes p(theta_obs | theta_true) with
        # the observation first. Condition the mixture on the standardized
        # true parameters without transforming their [0, 1] values.
        log_weights, means, L = self._predict(x_true)

        if (
            x_obs.ndim != 2
            or x_obs.shape != (x_true.shape[0], self.output_dim)
        ):
            raise ValueError(
                f"x_obs must have shape "
                f"({x_true.shape[0]}, {self.output_dim})."
            )

        if x_obs.device != x_true.device or x_obs.dtype != x_true.dtype:
            raise ValueError(
                "x_obs and x_true must have the same device and dtype."
            )

        residual = x_obs.unsqueeze(1) - means                 # (B,K,D)

        # P = L L^T, so r^T P r = ||L^T r||^2, NOT ||L r||^2.
        transformed = torch.matmul(
            L.transpose(-1, -2), residual.unsqueeze(-1)
        ).squeeze(-1)                                        # (B,K,D)
        mahalanobis_sq = transformed.square().sum(dim=-1)     # (B,K)

        # 0.5 * log(det(P)) = sum(log(diag(L))).
        half_log_det_precision = (
            L.diagonal(dim1=-2, dim2=-1).log().sum(dim=-1)
        )                                                     # (B,K)

        component_log_prob = (
            half_log_det_precision
            - self._log_normalizer
            - 0.5 * mahalanobis_sq
        )                                                     # (B,K)

        log_prob = torch.logsumexp(
            log_weights + component_log_prob, dim=-1
        )                                                     # (B,)

        if reduction == "mean":
            return log_prob.mean()
        if reduction == "sum":
            return log_prob.sum()
        return log_prob
        

# Short alias used by callers that refer to the implementation simply as an MDN.
LikelihoodMDN = GaussianMixtureMDN
