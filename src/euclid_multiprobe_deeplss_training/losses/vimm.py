import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, MultivariateNormal, MixtureSameFamily


class FullCovMixtureDensityRegressor(nn.Module):
    """
    Multivariate regression model:

        x -> z = encoder(x)
        z -> q_phi(y | z)

    q_phi(y | z) is a K-component Gaussian mixture.
    Each Gaussian component has a full covariance matrix.

    The training loss is:

        L = -log q_phi(y | z)

    which is the variational conditional log-likelihood objective for
    maximizing a lower bound on I(Z; Y).
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
        output_dim = (
            n_components
            + n_components * y_dim
            + n_components * self.n_tril
        )

        self.decoder = nn.Linear(hidden_dim, output_dim)

        # Indices used to fill lower-triangular Cholesky matrices.
        tril_idx = torch.tril_indices(row=y_dim, col=y_dim, offset=0)
        self.register_buffer("tril_idx", tril_idx)

    def forward(self, x: torch.Tensor) -> MixtureSameFamily:
        """
        Returns a torch.distributions.MixtureSameFamily object representing
        q_phi(y | x).
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
        scale_tril[:, :, diag_idx, diag_idx] = (
            F.softplus(scale_tril[:, :, diag_idx, diag_idx])
            + self.min_scale
        )

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
        return MixtureSameFamily(
            mixture_distribution=mixture_dist,
            component_distribution=component_dist,
        )

    def nll(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Negative conditional log-likelihood:

            -log q_phi(y | x)

        This is the VIMM loss for maximizing the variational lower bound
        on I(Z; Y), ignoring the constant H(Y).
        """
        dist = self.forward(x)
        return -dist.log_prob(y).mean()

    @torch.no_grad()
    def predict_mean(self, x: torch.Tensor) -> torch.Tensor:
        """
        Mixture mean E[Y | x].
        """
        return self.forward(x).mean

