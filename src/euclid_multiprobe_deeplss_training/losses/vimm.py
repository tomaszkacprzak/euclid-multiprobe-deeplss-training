import torch
from torch import nn
from ..utils.logger import get_logger
LOGGER = get_logger(__file__)

def mlp(in_dim, out_dim, hidden_dim):

    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, out_dim),
    )

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
        self.mi_mu_head = mlp(self.num_targets, self.num_targets, hidden_dim=128)
        self.mi_logvar_head = mlp(self.num_targets, self.num_targets, hidden_dim=128)

    def loss_components(self, inputs, targets):

        mu = self.mi_mu_head(inputs)
        logvar = self.mi_logvar_head(inputs)
        logvar = torch.clamp(logvar, min=-10.0, max=5.0)
        mse_  = self.loss_mse(inputs, targets)
        nll_ = self.lambda_mi * gaussian_nll(targets, mu, logvar)
        LOGGER.debug(f'VIMM loss: inputs.shape={inputs.shape} targets.shape={targets.shape} mu.shape={mu.shape} logvar.shape={logvar.shape}')
        return {'mse': mse_, 'nll': nll_}
        

    def forward(self, inputs, targets):

        components_ = self.loss_components(inputs, targets)
        total_loss = components_['mse'] + components_['nll']

        return total_loss



    