"""Small DeepSphere-style residual networks for HEALPix regression."""

from __future__ import annotations

from collections.abc import Callable

try:
    from torch import Tensor, nn
except ImportError as exc:  # pragma: no cover - exercised only in incomplete environments.
    raise ImportError(
        "HealGCNNResNet requires PyTorch. Install the project dependencies with "
        "`pip install -e .` before importing euclid_multiprobe_deeplss_training.networks.resnet."
    ) from exc


def _load_spherical_cheb_conv() -> type[nn.Module]:
    """Return DeepSphere's Chebyshev spherical convolution with a clear error."""
    try:
        from deepsphere.layers.chebyshev import SphericalChebConv
    except ImportError as exc:  # pragma: no cover - depends on optional runtime dependency.
        raise ImportError(
            "HealGCNNResNet uses deepsphere-cosmo-pytorch's "
            "`deepsphere.layers.chebyshev.SphericalChebConv`. Install the project "
            "dependencies with `pip install -e .` so the `deepsphere` package is available."
        ) from exc

    return SphericalChebConv


def _activation_factory(name: str) -> Callable[[], nn.Module]:
    """Create a small activation factory from a human-readable name."""
    activations: dict[str, Callable[[], nn.Module]] = {
        "gelu": nn.GELU,
        "relu": nn.ReLU,
        "silu": nn.SiLU,
    }
    try:
        return activations[name.lower()]
    except KeyError as exc:
        supported = ", ".join(sorted(activations))
        raise ValueError(f"Unsupported activation {name!r}; choose one of: {supported}.") from exc


class HealGCNNResidualBlock(nn.Module):
    """Residual graph-convolution block for HEALPix-like spherical maps.

    Parameters
    ----------
    channels:
        Number of per-pixel feature channels entering and leaving the block.
    laplacian:
        Sparse graph Laplacian used by DeepSphere for the current HEALPix or
        part-sky sampling. It is passed directly to ``SphericalChebConv``.
    kernel_size:
        Chebyshev polynomial order used by the DeepSphere convolution.
    dropout:
        Dropout probability applied after the first activation. ``0.0`` disables
        dropout.
    activation:
        One of ``"gelu"``, ``"relu"``, or ``"silu"``.

    Notes
    -----
    The block follows DeepSphere's tensor convention internally: inputs and
    outputs have shape ``(batch, pixels, channels)``. ``HealGCNNResNet`` handles
    conversion from the more common map convention ``(batch, channels, pixels)``.
    """

    def __init__(
        self,
        channels: int,
        laplacian: Tensor,
        kernel_size: int = 5,
        dropout: float = 0.0,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        spherical_cheb_conv = _load_spherical_cheb_conv()
        make_activation = _activation_factory(activation)

        self.conv1 = spherical_cheb_conv(channels, channels, laplacian, kernel_size)
        self.conv2 = spherical_cheb_conv(channels, channels, laplacian, kernel_size)
        self.activation = make_activation()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """Apply the residual block to ``x`` with shape ``(batch, pixels, channels)``."""
        residual = x
        x = self.conv1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.conv2(x)
        return self.activation(x + residual)


class HealGCNNResNet(nn.Module):
    """Compact DeepSphere/HEALPix residual graph CNN for regression.

    The model is intentionally small and readable: it projects input map channels
    to a hidden feature width, applies a stack of residual DeepSphere Chebyshev
    graph/spherical convolution blocks, pools over pixels, and predicts target
    values with a linear head.

    Parameters
    ----------
    in_channels:
        Number of input map channels.
    num_targets:
        Number of regression targets. The forward pass returns
        ``(batch, num_targets)`` float predictions.
    laplacian:
        Sparse graph Laplacian for the HEALPix pixels included in the input map.
        This is normally built by the DeepSphere tooling for a full-sky or
        part-sky HEALPix graph and is reused by every spherical convolution.
    hidden_channels:
        Feature width used after the input projection.
    num_blocks:
        Number of residual spherical graph-convolution blocks.
    kernel_size:
        Chebyshev polynomial order passed to DeepSphere's ``SphericalChebConv``.
    dropout:
        Optional dropout probability inside residual blocks.
    activation:
        One of ``"gelu"``, ``"relu"``, or ``"silu"``.

    Input shape
    -----------
    ``forward`` expects batched maps in ``channels_first`` format
    ``(batch, channels, pixels)`` by default. Set ``channels_last=True`` when
    calling ``forward`` to pass ``(batch, pixels, channels)`` tensors directly.
    """

    def __init__(
        self,
        in_channels: int,
        num_targets: int,
        laplacian: Tensor,
        hidden_channels: int = 32,
        num_blocks: int = 4,
        kernel_size: int = 5,
        dropout: float = 0.0,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        make_activation = _activation_factory(activation)

        self.input_projection = nn.Linear(in_channels, hidden_channels)
        self.blocks = nn.Sequential(
            *[
                HealGCNNResidualBlock(
                    hidden_channels,
                    laplacian=laplacian,
                    kernel_size=kernel_size,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(num_blocks)
            ]
        )
        self.activation = make_activation()
        self.head = nn.Linear(hidden_channels, num_targets)

    def forward(self, maps: Tensor, *, channels_last: bool = False) -> Tensor:
        """Predict regression targets from batched HEALPix maps.

        Parameters
        ----------
        maps:
            Input tensor with shape ``(batch, channels, pixels)`` by default, or
            ``(batch, pixels, channels)`` when ``channels_last=True``.
        channels_last:
            Whether ``maps`` is already in DeepSphere's ``(batch, pixels,
            channels)`` convention.

        Returns
        -------
        Tensor
            Float predictions with shape ``(batch, num_targets)``.
        """
        if maps.ndim != 3:
            raise ValueError(
                "HealGCNNResNet expects a 3D tensor shaped "
                "(batch, channels, pixels) or (batch, pixels, channels)."
            )

        x = maps if channels_last else maps.transpose(1, 2)
        x = self.input_projection(x)
        x = self.activation(x)
        x = self.blocks(x)
        x = x.mean(dim=1)
        return self.head(x)


__all__ = ["HealGCNNResidualBlock", "HealGCNNResNet"]
