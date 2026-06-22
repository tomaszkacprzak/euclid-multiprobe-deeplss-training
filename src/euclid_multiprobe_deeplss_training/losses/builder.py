from ..utils.logger import get_logger
from torch import nn

LOGGER = get_logger(__file__)
    
def build_loss(loss_name: str,
               model: nn.Module,
               num_targets: int):

    if loss_name == "mse":

        loss_fn = nn.MSELoss()

    elif loss_name == "vimm":

        from .vimm import VIMMLoss
        loss_fn = VIMMLoss(num_targets=num_targets, lambda_mi=0.1, dim=1)

    else:

        raise ValueError(f"Loss {loss_name} not supported")


    return loss_fn

