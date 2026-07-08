import torch

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
                  device: torch.device | str | None = None):

    if encoder_name in {"nested_transformer", "deep_nested_transformer"}:
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
        else:
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
            num_outputs=embed_dim,
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
            "out_features": embed_dim,
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

    else:

        raise ValueError(f"Encoder {encoder_name} not supported")


    return model



def build_loss(loss_name: str,
               embed_dim: int,
               num_targets: int,
               loss_args: dict | None = None):

    if loss_name == "mse":

        from .mse_loss import MSEHead
        loss_fn = MSEHead(embed_dim, num_targets)

    elif loss_name == "vimm_gmm":

        from .vimm_loss import VIMMGMMHead
        loss_fn = VIMMGMMHead(embed_dim, num_targets, **(loss_args or {}))
        
    else:

        raise ValueError(f"Loss {loss_name} not supported")


    return loss_fn
