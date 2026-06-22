import healpy as hp
import torch
import torch.nn as nn

class NestDownsampler(nn.Module):
    """
    A layer that manipulates the nest resolution of a Healpix map.
    """

    def __init__(self, nside, nside_base, nside_lower, operator="sum"):
        super().__init__()
        self.nside = int(nside)
        self.nside_base = int(nside_base)
        self.nside_lower = int(nside_lower)
        operators = {"sum": torch.sum, "mean": torch.mean}

        try:
            self.operator = operators[operator]
        except KeyError as exc:
            raise ValueError(f"Invalid operator: {operator}") from exc

    def forward(self, x):

        batch_size, npix, num_channels = x.shape
        nord = hp.nside2order(self.nside)
        nord_base = hp.nside2order(self.nside_base)
        npix_base = npix // ( hp.nside2npix(self.nside) // hp.nside2npix(self.nside_base)) 
        nord_lower = hp.nside2order(self.nside_lower)

        shape = [batch_size, npix_base] + [4] * (nord - nord_base) + [num_channels]
        dims_sum = tuple(range(1 + nord_lower - nord_base + 1, 1 + nord - nord_base + 1))  # 1+ because of the batch dimension
        x_lower = self.operator(x.reshape(shape), dim=dims_sum, keepdims=False)
                
        return x_lower.reshape(batch_size, -1, num_channels)
        

class NestChannelDownsampler(nn.Module):
    """
    A layer that downsamples a Healpix map to a lower resolution.
    """

    def __init__(self, nside, nside_base, nside_lower, operator="sum"):
        """
        Args:
            nside: The healpy nside of the input.
            nside_base: The healpy nside of the base resolution.
            nside_lower: The healpy nside of the lower resolution.
            operator: The operator to use for the downsampling (sum or mean).
        """

        super().__init__()
        self.nside = nside
        self.nside_base = nside_base
        self.nside_lower = nside_lower
        operators = {"sum": torch.sum, "mean": torch.mean}
        try:
            self.operator = operators[operator]
        except KeyError as exc:
            raise ValueError(f"Invalid operator: {operator}") from exc

    def forward(self, x):

        batch_size, npix, num_channels = x.shape
        nord = hp.nside2order(self.nside)
        nord_base = hp.nside2order(self.nside_base)
        npix_base = npix // ( hp.nside2npix(self.nside) // hp.nside2npix(self.nside_base)) 
        nord_lower = torch.tensor([hp.nside2order(nside) for nside in self.nside_lower])

        # each channel is lowered to a different nside
        assert x.shape[-1] == len(self.nside_lower)

        shape = [batch_size, npix_base] + [4] * (nord - nord_base)

        # loop over channels
        list_channels_lower = []
        for i in range(num_channels):
            nord_lower_ = int(nord_lower[i].item())
            dims_sum = tuple(range(1 + nord_lower_ - nord_base + 1, 1 + nord - nord_base + 1))  # 1+ because of the batch dimension

            x_channel = x[..., i].reshape(shape)
            if dims_sum:
                x_channel_lower = self.operator(x_channel, dim=dims_sum, keepdim=True).expand_as(x_channel)
            else:
                x_channel_lower = x_channel

            list_channels_lower.append(x_channel_lower.reshape(batch_size, -1))

        return torch.stack(list_channels_lower, dim=-1)
