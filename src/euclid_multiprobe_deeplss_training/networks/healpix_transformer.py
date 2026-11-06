from .nested_transfomer import NestedHierarchicalLocalWindowTransformer
import healpy as hp
import numpy as np
import torch
import torch.nn as nn


class HealpixNestedHierarchicalLocalWindowTransformer(nn.Module):

    def __init__(self, batch_size, spins, num_pixels, nside, nside_down, in_channels, **kwargs):

        super().__init__()

        assert nside>nside_down, "nside must be greater than nside_down"

        self.batch_size = batch_size
        self.spins = spins
        self.nside = nside
        self.nside_down = nside_down
        self.num_pixels = num_pixels
        self.in_channels = in_channels
        self.scalar_channels = torch.sum(torch.where(spins==0, 1, spins))
        self.num_nested_levels =  hp.nside2order(nside) - hp.nside2order(nside_down)
       
        # get the number of top-level healpix pixels present in the map, at nside_down
        num_pixels_per_top_level_token = hp.nside2npix(nside) // hp.nside2npix(nside_down)
        assert self.num_pixels % num_pixels_per_top_level_token ==0, f"Cannot split {num_pixels} pixels into {num_pixels_per_top_level_token} top-level tokens"
        num_top_level_tokens = num_pixels // num_pixels_per_top_level_token
        
        # get the number of nested levels of each top-level pixel
        nested_shape = (4,) * self.num_nested_levels
        self.nested_shape = (self.scalar_channels, num_top_level_tokens, *nested_shape)

        self.transformer = NestedHierarchicalLocalWindowTransformer(num_nested_levels=self.num_nested_levels, in_channels=self.scalar_channels, **kwargs)

        # super().__init__(num_nested_levels=self.num_nested_levels, in_channels=self.scalar_channels, **kwargs)

        self.trainable_sequence = self.transformer
        self.trainable_input_shape = (self.batch_size, *self.nested_shape)


    def batch_flat_to_nested(self, batch: torch.Tensor) -> torch.Tensor:
        """Convert pipeline batch shaped ``(B, P, C)`` to nested transformer input."""

        # dimensions: (batch_size, num_channels, num_top_level_tokens, *nested_shape)
        return batch.movedim(2, 1).contiguous().reshape(batch.shape[0], *self.nested_shape)

    def forward(self, x):

        # x is a list of channel tensors with shape (batch_size, num_pixels), they can have scalars or complex data types
        x_stacked = []
        for i, x_i in enumerate(x):
            if x_i.is_complex(): # convert complex tensors to real,imag channels if needed
                x_stacked.append(torch.stack([x_i.real, x_i.imag], dim=-1))
            else: # stack scalar tensors
                x_stacked.append(x_i)
        x_stacked = torch.stack(x_stacked, dim=-1)

        return self.transformer(self.batch_flat_to_nested(x_stacked))
