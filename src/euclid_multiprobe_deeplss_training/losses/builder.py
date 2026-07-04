from ..utils.logger import get_logger
from torch import nn

LOGGER = get_logger(__file__)
    
def build_loss(loss_name: str,
               num_targets: int):

    if loss_name == "mse":

        loss_fn = nn.MSELoss()

    elif loss_name == "vimm":

        raise ValueError(f"Loss {loss_name} not supported")
        
    else:

        raise ValueError(f"Loss {loss_name} not supported")


    return loss_fn

