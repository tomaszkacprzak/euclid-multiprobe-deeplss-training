"""Placeholder for the linear on-the-fly physics model."""

from torch import nn


class OntheflyPhysicsModelLinear(nn.Module):
    """Linear physics model whose local implementation has not been supplied."""

    def __init__(self, *_args, **_kwargs) -> None:
        super().__init__()
        # TODO: Port the linear physics model implementation into this package.
        raise NotImplementedError(
            "OntheflyPhysicsModelLinear has not yet been implemented in this package."
        )
