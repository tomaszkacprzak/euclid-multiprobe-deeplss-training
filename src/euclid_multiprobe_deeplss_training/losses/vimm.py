import torch
from torch import nn

def gaussian_nll(y, mu, logvar, dim=1):

    """

    Negative log likelihood of y under diagonal Gaussian:

    q(y|z) = N(mu(z), diag(exp(logvar(z))))

    """

    return 0.5 * (

        logvar + (y - mu) ** 2 / torch.exp(logvar)

    ).sum(dim=dim).mean()


class VIMMLoss(nn.Module):

    def __init__(self, num_targets: int, lambda_mi: float = 1.0, dim: int = 1):
        super().__init__()
        self.lambda_mi = lambda_mi
        self.dim = dim
        self.num_targets = num_targets
        self.loss_mse = nn.MSELoss()
        self.mi_mu_head = nn.Linear(self.num_targets, self.num_targets)
        self.mi_logvar_head = nn.Linear(self.num_targets, self.num_targets)

    def loss_components(self, inputs, targets):

        mu = self.mi_mu_head(inputs)
        logvar = self.mi_logvar_head(inputs)
        logvar = torch.clamp(logvar, min=-10.0, max=5.0)
        mse_  = self.loss_mse(inputs, targets)
        nll_ = gaussian_nll(targets, mu, logvar)

        return mse_, nll_
        

    def forward(self, inputs, targets):

        mse_, nll_ = self.loss_components(inputs, targets)
        total_loss = mse_ + self.lambda_mi * nll_

        return total_loss

    