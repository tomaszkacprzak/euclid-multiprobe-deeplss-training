import healpy as hp
import torch
import torch.nn as nn


class NestDownsampler(nn.Module):
    """
    A layer that manipulates the nest resolution of a Healpix map.
    """

    def __init__(self, nside, nside_base, nside_lower, operator="mean"):

        assert nside > nside_base, "nside must be greater than nside_base"
        assert nside > nside_lower, "nside must be greater than nside_lower"
        assert nside_lower > nside_base, "nside_lower must be greater than nside_base"

        super().__init__()
        self.nside = int(nside)
        self.nside_base = int(nside_base)
        self.nside_lower = int(nside_lower)
        self.nord_lower = hp.nside2order(self.nside_lower)
        self.nord_base = hp.nside2order(self.nside_base)
        self.nord = hp.nside2order(self.nside)
        self.npix = hp.nside2npix(self.nside)
        self.npix_base = hp.nside2npix(self.nside_base)
        operators = {"sum": torch.sum, "mean": torch.mean}

        try:
            self.operator = operators[operator]
        except KeyError as exc:
            raise ValueError(f"Invalid operator: {operator}") from exc

    def forward(self, x):

        batch_size, npix, num_bins, num_fields = x.shape # assumes the same number of bins for each field
        npix_base = npix // ( self.npix // self.npix_base) 
        

        shape = [batch_size, npix_base] + [4] * (self.nord - self.nord_base) + [num_bins, num_fields]
        dims_sum = tuple(range(1 + self.nord_lower - self.nord_base + 1, 1 + self.nord - self.nord_base + 1))  # 1+ because of the batch dimension
        x_lower = self.operator(x.reshape(shape), dim=dims_sum, keepdims=False)
                
        return x_lower.reshape(batch_size, -1, num_bins, num_fields)
        

class NestChannelScalecut(nn.Module):
    """Aggregate NESTED pixels with statistical weights and repeat to input resolution.

    Each channel has its own target resolution. A target pixel contains a
    contiguous group of input pixels; its weighted sum or weighted mean is
    repeated over that group. Groups with zero total weight return zero.
    """

    def __init__(self, nside, nside_base, nside_lower, operator="sum"):
        """
        Args:
            nside: The healpy nside of the input.
            nside_base: The healpy nside of the base resolution.
            nside_lower: One target nside per channel, between nside_base and nside.
            operator: Weighted sum (default) or weighted mean.
        """

        super().__init__()
        if operator not in ("sum", "mean"):
            raise ValueError(f"Invalid operator: {operator}")
        self.operator = operator

        def validate_nside(value, name):
            try:
                integer = int(value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"{name} must be a valid NESTED nside") from exc
            if integer != value or not hp.isnsideok(integer, nest=True):
                raise ValueError(f"{name} must be a valid NESTED nside")
            return integer

        self.nside = validate_nside(nside, "nside")
        self.nside_base = validate_nside(nside_base, "nside_base")
        self.nside_lower = [validate_nside(value, "nside_lower") for value in nside_lower]
        if self.nside_base > self.nside:
            raise ValueError("nside_base must not exceed nside")
        if not self.nside_lower or any(not self.nside_base <= value <= self.nside for value in self.nside_lower):
            raise ValueError("nside_lower must contain target nsides between nside_base and nside")
        self._base_group_size = (self.nside // self.nside_base) ** 2
        self._group_sizes = tuple((self.nside // value) ** 2 for value in self.nside_lower)

    def forward(self, x, w):
        """Apply the scale cut to ``x`` with real nonnegative statistical weights ``w``.

        ``x`` has shape ``(batch, pixels, channels)`` and ``w`` must broadcast to
        that shape on the same device. Pixels must comprise complete NESTED
        base-resolution patches. Zero-weight pixels are ignored, even when
        their map values are NaN or infinite. PyTorch dtype promotion applies.
        """
        if not isinstance(x, torch.Tensor) or x.ndim != 3:
            raise ValueError("x must be a tensor with shape (batch, pixels, channels)")
        batch_size, npix, num_channels = x.shape
        if num_channels != len(self.nside_lower):
            raise ValueError("x must have one channel per nside_lower entry")
        if npix == 0 or npix % self._base_group_size:
            raise ValueError("x pixels must contain complete NESTED base-resolution patches")
        if not isinstance(w, torch.Tensor):
            raise TypeError("w must be a tensor of statistical weights")
        if w.device != x.device:
            raise ValueError("w must be on the same device as x")
        if w.is_complex():
            raise ValueError("w must contain real statistical weights")
        try:
            weights = torch.broadcast_to(w, x.shape)
        except RuntimeError as exc:
            raise ValueError("w must broadcast to the shape of x") from exc
        if torch.any(~torch.isfinite(w) | (w < 0)):
            raise ValueError("w must contain finite nonnegative statistical weights")

        channels = []
        for channel, group_size in enumerate(self._group_sizes):
            shape = (batch_size, npix // group_size, group_size)
            values = x[..., channel].reshape(shape)
            channel_weights = weights[..., channel].reshape(shape)
            # Mask before multiplying: 0 * NaN would otherwise contaminate the
            # reduction and the derivative with respect to the weights.
            values = torch.where(channel_weights > 0, values, torch.zeros_like(values))
            reduced = (values * channel_weights).sum(dim=-1, keepdim=True)
            if self.operator == "mean":
                total_weight = channel_weights.sum(dim=-1, keepdim=True)
                denominator = torch.where(total_weight > 0, total_weight, torch.ones_like(total_weight))
                reduced = reduced / denominator
            channels.append(reduced.expand(shape).reshape(batch_size, npix))
        return torch.stack(channels, dim=-1)
