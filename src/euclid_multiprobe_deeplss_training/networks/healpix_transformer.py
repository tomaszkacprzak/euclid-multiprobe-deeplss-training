from .nested_transfomer import NestedHierarchicalLocalWindowTransformer
import healpy as hp
import numpy as np
import torch



class HealpixNestedHierarchicalLocalWindowTransformer(NestedHierarchicalLocalWindowTransformer):

    def __init__(self, num_pixels, nside, nside_down, in_channels, **kwargs):

        assert nside>nside_down, "nside must be greater than nside_down"
        
        self.nside = nside
        self.nside_down = nside_down
        self.num_pixels = num_pixels
        self.in_channels = in_channels
        self.num_nested_levels =  hp.nside2order(nside) - hp.nside2order(nside_down)
       
        # get the number of top-level healpix pixels present in the map, at nside_down
        num_pixels_per_top_level_token = hp.nside2npix(nside) // hp.nside2npix(nside_down)
        assert self.num_pixels % num_pixels_per_top_level_token ==0, f"Cannot split {num_pixels} pixels into {num_top_level_tokens} top-level tokens"
        num_top_level_tokens = num_pixels // num_pixels_per_top_level_token
        
        # get the number of nested levels of each top-level pixel
        nested_shape = (4,) * self.num_nested_levels
        self.nested_shape = (self.in_channels, num_top_level_tokens, *nested_shape)

        super().__init__(num_nested_levels=self.num_nested_levels, in_channels=self.in_channels, **kwargs)


    def batch_flat_to_nested(self, batch: torch.Tensor) -> torch.Tensor:
        """Convert pipeline batch shaped ``(B, P, C)`` to nested transformer input."""

        # dimensions: (batch_size, num_channels, num_top_level_tokens, *nested_shape)
        return batch.movedim(2, 1).contiguous().reshape(batch.shape[0], *self.nested_shape)

    def forward(self, x):
        return super().forward(self.batch_flat_to_nested(x))
