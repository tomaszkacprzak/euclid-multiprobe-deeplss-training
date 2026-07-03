"""Map-only DeepSphere regression model in PyTorch.

This is the PyTorch equivalent of the historical TensorFlow/Keras
``HealpyGCNN`` example that stacks pseudo-convolutions, Chebyshev graph
convolutions, residual graph-convolution blocks, and a dense regression head.
"""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from deepsphere import HealpyGCNN, healpy_layers
from deepsphere.utils import extend_indices


class LazyLayerNorm(nn.Module):
    """LayerNorm whose normalized shape is inferred on the first forward pass."""

    def __init__(self):
        super().__init__()
        self.norm = None

    def forward(self, x):
        if self.norm is None:
            self.norm = nn.LayerNorm(x.shape[1:]).to(device=x.device, dtype=x.dtype)
        return self.norm(x)



class ResnetDeepSphereRegressor(HealpyGCNN):
    
    def __init__(self, 
        n_side, 
        indices, 
        batch_size, 
        n_channels, 
        out_features, 
        n_filters=32, 
        downsampling_layers=3, 
        cheby_layers=2, 
        residual_layers=6, 
        poly_degree=5, 
        n_neighbors=20,
        activation=F.relu):
        

        # HealpyGCNN validates that the footprint can be reduced by the layers that
        # downsample. Extend sparse footprints to full NEST parent-pixel groups
        # before constructing the model. If your supplied indices already satisfy
        # this, the call below returns the same set.
        n_side_out = n_side // (2 ** (downsampling_layers + cheby_layers))
        indices = extend_indices(np.asarray(indices), nside_in=n_side, nside_out=n_side_out)

        x_batch = torch.zeros((batch_size, len(indices), n_channels), dtype=torch.float32)
        layers = []

        # Downsampling / pseudo-convolution stack.
        for _ in range(downsampling_layers):
            layers.append(healpy_layers.HealpyPseudoConv(p=1, Fout=n_filters, activation=activation))
            n_filters *= 2

        # Chebyshev graph-convolution downsampling blocks.
        for _ in range(cheby_layers):
            layers.append(healpy_layers.HealpyChebyshev(K=poly_degree, Fout=n_filters, activation=activation))
            layers.append(nn.LayerNorm(n_filters))
            layers.append(healpy_layers.HealpyPseudoConv(p=1, Fout=n_filters, activation=activation))

        # Residual Chebyshev graph-convolution blocks.
        for _ in range(residual_layers):
            layers.append(
                healpy_layers.Healpy_ResidualLayer(
                    "CHEBY",
                    layer_kwargs={"K": poly_degree, "activation": activation, "use_bias": True},
                    use_bn=True,
                    bn_kwargs={},
                    norm_type="layer_norm",
                )
            )

        # Dense regression head: Flatten -> LayerNorm -> Dense(out_features).
        # Lazy modules infer the flattened feature count on the first forward pass.
        layers.append(nn.Flatten())
        layers.append(LazyLayerNorm())
        layers.append(nn.LazyLinear(out_features))

        model = HealpyGCNN(
            nside=n_side,
            indices=indices,
            layers=layers,
            n_neighbors=n_neighbors,
            max_batch_size=batch_size,
            initial_Fin=n_channels,
        )
        return model, x_batch

