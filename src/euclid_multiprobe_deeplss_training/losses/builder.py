from ..utils.logger import get_logger
from torch import nn

LOGGER = get_logger(__file__)


def build_loss(loss_name: str,
               embed_dim: int,
               num_targets: int,
               loss_args: dict = {}):

    if loss_name == "mse":

        from .mse import MSEHead
        loss_fn = MSEHead(embed_dim, num_targets)

    elif loss_name == "vimm_gmm":

        from .vimm import VIMMGMMHead
        loss_fn = VIMMGMMHead(embed_dim, num_targets, **loss_args)
        
    else:

        raise ValueError(f"Loss {loss_name} not supported")


    return loss_fn
