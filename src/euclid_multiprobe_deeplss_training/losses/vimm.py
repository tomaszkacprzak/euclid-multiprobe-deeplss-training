import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, MixtureSameFamily, MultivariateNormal


class FullCovMixtureDensityRegressor(nn.Module):
    """
    Multivariate Gaussian-mixture density regressor with full covariance matrices.

    The module maps an input feature vector ``x`` to the parameters of
    ``q_phi(y | x)``, a ``K``-component mixture of multivariate Gaussian
    distributions. Each component has its own mean vector and full covariance
    matrix parameterized through a lower-triangular Cholesky factor.

    Args:
        x_dim: Number of features in each input sample ``x``.
        y_dim: Number of regression targets in each output/event sample ``y``.
        hidden_dim: Width of the two hidden encoder layers.
        n_components: Number of Gaussian mixture components ``K``.
        min_scale: Positive value added to every Cholesky diagonal entry after
            ``softplus`` to keep component covariance matrices positive definite.

    Inputs:
        x: Tensor with shape ``[batch_size, x_dim]`` containing input features.
        y: Tensor with shape ``[batch_size, y_dim]`` containing target values;
            used by :meth:`nll` when computing the negative conditional
            log-likelihood.

    Outputs:
        forward: A ``torch.distributions.MixtureSameFamily`` distribution with
            batch shape ``[batch_size]`` and event shape ``[y_dim]``. Its mixture
            distribution has logits with shape ``[batch_size, n_components]``;
            its component distribution has means with shape
            ``[batch_size, n_components, y_dim]`` and Cholesky factors with shape
            ``[batch_size, n_components, y_dim, y_dim]``.
        nll: Scalar tensor containing the mean negative log-likelihood
            ``-log q_phi(y | x)`` across the batch.
        predict_mean: Tensor with shape ``[batch_size, y_dim]`` containing the
            mixture mean ``E[Y | x]``.
    """

    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        hidden_dim: int = 64,
        n_components: int = 5,
        min_scale: float = 1e-3,
    ):
        super().__init__()

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.n_components = n_components
        self.min_scale = min_scale

        # Number of entries in a lower-triangular D x D matrix.
        self.n_tril = y_dim * (y_dim + 1) // 2

        self.encoder = nn.Sequential(
            nn.Linear(x_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # The decoder predicts:
        #   mixture logits:           K
        #   component means:          K * D
        #   Cholesky parameters:      K * D * (D + 1) / 2
        output_dim = n_components + n_components * y_dim + n_components * self.n_tril

        self.decoder = nn.Linear(hidden_dim, output_dim)

        # Indices used to fill lower-triangular Cholesky matrices.
        tril_idx = torch.tril_indices(row=y_dim, col=y_dim, offset=0)
        self.register_buffer("tril_idx", tril_idx)

    def distribution(self, x: torch.Tensor) -> MixtureSameFamily:
        """
        Build ``q_phi(y | x)`` or score targets with its mean NLL.

        Args:
            x: Tensor with shape ``[batch_size, x_dim]`` containing input
                features. The tensor device and dtype are used for all generated
                distribution parameters.

        Returns:
            A``MixtureSameFamily`` distribution with batch
            shape ``[batch_size]`` and event shape ``[y_dim]``. 
        """
        batch_size = x.shape[0]
        K = self.n_components
        D = self.y_dim

        z = self.encoder(x)
        raw = self.decoder(z)

        # 1. Mixture logits: shape [B, K]
        logits = raw[:, :K]

        # 2. Component means: shape [B, K, D]
        start = K
        end = start + K * D
        loc = raw[:, start:end].reshape(batch_size, K, D)

        # 3. Raw lower-triangular Cholesky entries: shape [B, K, D(D+1)/2]
        raw_tril = raw[:, end:].reshape(batch_size, K, self.n_tril)

        # Build scale_tril: shape [B, K, D, D]
        scale_tril = x.new_zeros(batch_size, K, D, D)
        scale_tril[:, :, self.tril_idx[0], self.tril_idx[1]] = raw_tril

        # Make diagonal strictly positive so covariance = L L^T is positive definite.
        diag_idx = torch.arange(D, device=x.device)
        scale_tril[:, :, diag_idx, diag_idx] = F.softplus(scale_tril[:, :, diag_idx, diag_idx]) + self.min_scale

        # Mixture distribution over K components.
        mixture_dist = Categorical(logits=logits)

        # Component distribution:
        # batch shape: [B, K]
        # event shape: [D]
        component_dist = MultivariateNormal(
            loc=loc,
            scale_tril=scale_tril,
        )

        # Final distribution:
        # batch shape: [B]
        # event shape: [D]
        dist = MixtureSameFamily(
            mixture_distribution=mixture_dist,
            component_distribution=component_dist,
        )
        return dist

    def nll(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the mean negative conditional log-likelihood.

        This is the VIMM loss for maximizing the variational lower bound on
        ``I(Z; Y)``, ignoring the constant ``H(Y)``.

        Args:
            x: Tensor with shape ``[batch_size, x_dim]`` containing input
                features.
            y: Tensor with shape ``[batch_size, y_dim]`` containing regression
                targets/events to score under ``q_phi(y | x)``.

        Returns:
            Scalar tensor equal to ``-dist.log_prob(y).mean()``.
        """
        dist = self.distribution(x)
        return -dist.log_prob(y).mean()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the mean negative conditional log-likelihood.

        This is the VIMM loss for maximizing the variational lower bound on
        ``I(Z; Y)``, ignoring the constant ``H(Y)``.
        """
        return self.nll(x, y)

    @torch.no_grad()
    def predict_mean(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict the conditional mixture mean ``E[Y | x]``.

        Args:
            x: Tensor with shape ``[batch_size, x_dim]`` containing input
                features.

        Returns:
            Tensor with shape ``[batch_size, y_dim]`` containing the weighted
            mean across Gaussian mixture components.
        """
        return self.distribution(x).mean
