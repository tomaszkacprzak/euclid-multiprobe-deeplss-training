import torch
import torch.nn as nn

from ..utils.logger import get_logger

LOGGER = get_logger(__file__)

    
def build_encoder(encoder_name: str,
                  num_channels: int,
                  embed_dim: int,
                  nside: int,
                  nside_down: int,
                  num_pixels: int,
                  batch_size: int = None,
                  indices: list[int] = None,
                  encoder_args: dict | None = None,
                  unstack_function
                   = None,
                  device: torch.device | str | None = None):

    if encoder_name == "nested_transformer":
        
        from .healpix_transformer import HealpixNestedHierarchicalLocalWindowTransformer as TransformerClass

        constructor_args = {
            "base_embed_dim": 256,
            "growth": "128",
            "num_heads": 4,
            "window_levels": 3,
            "local_blocks_per_level": 1,
            "global_blocks": 1,
            "mlp_ratio": 4,
        }

        constructor_args.update(encoder_args or {})
        model = TransformerClass(
            nside=nside,
            nside_down=nside_down,
            num_pixels=num_pixels,
            in_channels=num_channels,
            embed_dim=embed_dim,
            **constructor_args,
        )

        LOGGER.info(f"Built {TransformerClass.__name__} {encoder_name}")
        LOGGER.info(f"  num_channels: {num_channels}, embed_dim: {embed_dim}")
        LOGGER.info(f"  encoder_args: {constructor_args}")




    elif encoder_name == "deep_nested_transformer":
            
        from .healpix_deep_transformer import HealpixDeepNestedHierarchicalLocalWindowTransformer as TransformerClass

        constructor_args = {
            "base_embed_dim": 256,
            "growth": "128",
            "num_heads": 4,
            "window_levels": 3,
            "local_blocks_per_level": 2,
            "global_blocks": 2,
            "mlp_ratio": 4,
            "drop_path_rate": 0.1,
            "drop_path_schedule": "linear",
            "pre_norm": True,
            "residual_dropout": 0.0,
        }

        constructor_args.update(encoder_args or {})
        model = TransformerClass(
            nside=nside,
            nside_down=nside_down,
            num_pixels=num_pixels,
            in_channels=num_channels,
            embed_dim=embed_dim,
            **constructor_args,
        )

        LOGGER.info(f"Built {TransformerClass.__name__} {encoder_name}")
        LOGGER.info(f"  num_channels: {num_channels}, embed_dim: {embed_dim}")
        LOGGER.info(f"  encoder_args: {constructor_args}")


    elif encoder_name == "deepsphere_resnet":

        assert batch_size is not None, "batch_size is required for deepsphere_resnet"
        assert indices is not None, "indices are required for deepsphere_resnet"

        from .deepsphere_resnet import ResnetDeepSphereRegressor
        
        # Defaults
        constructor_args = {
            "n_side": nside,
            "indices": indices,
            "batch_size": batch_size,
            "n_channels": num_channels,
            "embed_dim": embed_dim,
            "n_filters": 32,
            "downsampling_layers": 2,
            "cheby_layers": 2,
            "residual_layers": 6,
            "poly_degree": 5,
            "n_neighbors": 20,
        }
        # Update with encoder_args
        constructor_args.update(encoder_args or {})

        # Build model
        model = ResnetDeepSphereRegressor(
            **constructor_args,
        )

        LOGGER.info(f"Built ResnetDeepSphereRegressor {encoder_name}")
        LOGGER.info(f"  num_channels: {num_channels}, embed_dim: {embed_dim}")
        LOGGER.info(f"  n_side: {nside}, n_side_down: {nside_down}, num_pixels: {num_pixels}")
        LOGGER.info(f"  encoder_args: {constructor_args}")

    elif encoder_name == "cls_linear":

        assert indices is not None, "indices are required for cls_linear"

        from .linear_cls import LinearClsNetwork

        constructor_args = {
            "indices": indices,
            "nside": nside,
            "num_channels": num_channels,
            "embed_dim": embed_dim,
            "unstack_function": unstack_function,
            "device": device,
        }
        constructor_args.update(encoder_args or {})
        model = LinearClsNetwork(**constructor_args)

        LOGGER.info(f"Built {LinearClsNetwork.__name__} {encoder_name}")
        LOGGER.info(f"  num_channels: {num_channels}, embed_dim: {embed_dim}")
        LOGGER.info(f"  encoder_args: {constructor_args}")

    elif encoder_name == "cls_transformer":

        assert indices is not None, "indices are required for cls_transformer"

        from .transformer_cls import ShiftedWindowTransformerClsNetwork
        constructor_args = {
            "indices": indices,
            "nside": nside,
            "num_channels": num_channels,
            "embed_dim": embed_dim,
            "unstack_function": unstack_function,
            "device": device,
            "inner_embed_dim": 32,
            "depth": 6,
            "num_heads": 8,
            "window_size": 64,
        }
        constructor_args.update(encoder_args or {})
        model = ShiftedWindowTransformerClsNetwork(**constructor_args)

        LOGGER.info(f"Built {ShiftedWindowTransformerClsNetwork.__name__} {encoder_name}")
        LOGGER.info(f"  num_channels: {num_channels}, embed_dim: {embed_dim}")
        LOGGER.info(f"  encoder_args: {constructor_args}")

    elif encoder_name == "cls_cnn":

        assert indices is not None, "indices are required for cls_cnn"

        from .cnn_cls import ConvolutionalResidualClsNetwork
        constructor_args = {
            "indices": indices,
            "nside": nside,
            "num_channels": num_channels,
            "embed_dim": embed_dim,
            "unstack_function": unstack_function,
            "device": device,
            "inner_channels": 32,
            "downsampling_layers": 3,
            "residual_layers": 3,
            "kernel_size": 3,
        }
        constructor_args.update(encoder_args or {})
        model = ConvolutionalResidualClsNetwork(**constructor_args)

        LOGGER.info(f"Built {ConvolutionalResidualClsNetwork.__name__} {encoder_name}")
        LOGGER.info(f"  num_channels: {num_channels}, embed_dim: {embed_dim}")
        LOGGER.info(f"  encoder_args: {constructor_args}")


    else:

        raise ValueError(f"Encoder {encoder_name} not supported")


    return model



def build_loss(loss_name: str,
               encoder: nn.Module,
               embed_dim: int,
               num_targets: int,
               batch_size: int,
               loss_args: dict | None = None):

    if loss_name == "mse":

        from .mse_loss import MSEModel
        model = MSEModel(encoder, num_targets)
        

    elif loss_name == "vimm_gmm":

        from .vimm_loss import VIMMGMMModel
        model = VIMMGMMModel(encoder, num_targets, loss_args)

    elif loss_name == "flowmatching":

        from .flowmatching_loss import CNFFMModel
        model = CNFFMModel(encoder, y_dim=num_targets, vectorfield_kwargs=loss_args)

    else:

        raise ValueError(f"Loss {loss_name} not supported")

    


    return model
