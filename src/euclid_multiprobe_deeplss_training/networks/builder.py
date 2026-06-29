from ..utils.logger import get_logger

LOGGER = get_logger(__file__)




    
def build_model(model_name: str,
                num_channels: int,
                num_targets: int,
                nside: int,
                nside_down: int,
                num_pixels: int,
                model_args: dict | None = None):

    if model_name == "nested_transformer":

        from .healpix_transformer import HealpixNestedHierarchicalLocalWindowTransformer
        constructor_args = {
            "base_embed_dim": 256,
            "growth": "128",
            "num_heads": 4,
            "window_levels": 3,
            "local_blocks_per_level": 1,
            "global_blocks": 1,
            "mlp_ratio": 4,
        }
        constructor_args.update(model_args or {})
        model = HealpixNestedHierarchicalLocalWindowTransformer(
            nside=nside,
            nside_down=nside_down,
            num_pixels=num_pixels,
            in_channels=num_channels,
            num_outputs=num_targets,
            **constructor_args,
        )

        LOGGER.info(f"Built HealpixNestedHierarchicalLocalWindowTransformer {model_name}")
        LOGGER.info(f"  num_channels: {num_channels}, num_targets: {num_targets}")


    else:

        raise ValueError(f"Model {model_name} not supported")


    return model
