"""Common interface for conditional likelihood density estimators."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class LikelihoodBase(nn.Module):
    """Base class for models of ``p(theta_obs | theta_true)``.

    Subclasses only need to implement :meth:`log_likelihood`.  Both parameter
    tensors passed to this class have shape ``(N, M)``: N samples containing M
    observed/true parameters.
    """

    def log_likelihood(self, theta_obs: torch.Tensor, theta_true: torch.Tensor) -> torch.Tensor:
        """Return one log likelihood per sample, as a tensor of shape ``(N,)``."""
        raise NotImplementedError

    def forward(self, theta_obs: torch.Tensor, theta_true: torch.Tensor) -> torch.Tensor:
        """Delegate module calls to :meth:`log_likelihood`."""
        return self.log_likelihood(theta_obs, theta_true)

    def fit(
        self,
        theta_obs_training: torch.Tensor,
        theta_true_training: torch.Tensor,
        theta_obs_validation: torch.Tensor,
        theta_true_validation: torch.Tensor,
        *,
        num_epochs: int = 100,
        batch_size: int = 128,
        learning_rate: float = 1e-3,
        device: torch.device | str | None = None,
    ) -> dict[str, list[float]]:
        """Fit the estimator and print training/validation likelihood each epoch."""
        self._validate_pairs(theta_obs_training, theta_true_training, "training")
        self._validate_pairs(theta_obs_validation, theta_true_validation, "validation")
        if num_epochs <= 0 or batch_size <= 0 or learning_rate <= 0:
            raise ValueError("num_epochs, batch_size, and learning_rate must be positive.")

        target_device = torch.device(device or next(self.parameters()).device)
        self.to(target_device)
        # Each loader batch contains two tensors of shape (B, M), where B <= batch_size.
        loader = DataLoader(
            TensorDataset(theta_obs_training.float(), theta_true_training.float()),
            batch_size=batch_size,
            shuffle=True,
        )
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        history: dict[str, list[float]] = {"training": [], "validation": []}

        for epoch in range(1, num_epochs + 1):
            self.train()
            total_log_likelihood = 0.0
            total_samples = 0
            for theta_obs, theta_true in loader:
                theta_obs, theta_true = theta_obs.to(target_device), theta_true.to(target_device)
                optimizer.zero_grad()
                log_prob = self.log_likelihood(theta_obs, theta_true)  # (B,)
                (-log_prob.mean()).backward()
                optimizer.step()
                total_log_likelihood += log_prob.detach().sum().item()
                total_samples += len(theta_obs)

            self.eval()
            with torch.no_grad():
                validation_log_likelihood = self.log_likelihood(
                    theta_obs_validation.float().to(target_device),
                    theta_true_validation.float().to(target_device),
                ).mean().item()
            training_log_likelihood = total_log_likelihood / total_samples
            history["training"].append(training_log_likelihood)
            history["validation"].append(validation_log_likelihood)
            print(
                f"Epoch {epoch:4d}/{num_epochs}: training log likelihood "
                f"{training_log_likelihood:.6f}, validation log likelihood {validation_log_likelihood:.6f}"
            )
        return history

    @staticmethod
    def _validate_pairs(theta_obs: torch.Tensor, theta_true: torch.Tensor, name: str) -> None:
        """Check that observed and true inputs are matching ``(N, M)`` tensors."""
        if theta_obs.ndim != 2 or theta_true.ndim != 2:
            raise ValueError(f"{name} tensors must have shape (N, M).")
        if theta_obs.shape != theta_true.shape:
            raise ValueError(f"{name} observed and true tensors must have the same shape.")
        if len(theta_obs) == 0:
            raise ValueError(f"{name} tensors must not be empty.")

    def save(self, path: str | Path) -> None:
        """Store model parameters in a portable PyTorch checkpoint."""
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": self.state_dict()}, output)

    def load(self, path: str | Path, *, map_location: torch.device | str | None = None) -> LikelihoodBase:
        """Load model parameters into this instance and return it."""
        checkpoint = torch.load(Path(path), map_location=map_location, weights_only=True)
        self.load_state_dict(checkpoint["model_state_dict"])
        return self
