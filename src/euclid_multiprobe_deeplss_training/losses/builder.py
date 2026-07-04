from ..utils.logger import get_logger
from .vimm import FullCovMixtureDensityRegressor
from torch import nn

LOGGER = get_logger(__file__)
    
def build_loss(loss_name: str,
               num_targets: int,
               loss_args: dict = {}):

    if loss_name == "mse":

        loss_fn = nn.MSELoss()

    elif loss_name == "vimm_gmm":

        loss_fn = FullCovMixtureDensityRegressor(
            x_dim=num_targets,
            y_dim=num_targets,
            **loss_args,
        )
        
    else:

        raise ValueError(f"Loss {loss_name} not supported")


    return loss_fn

