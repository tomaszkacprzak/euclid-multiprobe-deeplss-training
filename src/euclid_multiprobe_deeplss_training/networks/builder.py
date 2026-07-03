from ..utils.logger import get_logger

LOGGER = get_logger(__file__)




    
def build_model(model_name: str,
                num_channels: int,
                num_targets: int,
                nside: int,
                nside_down: int,
                num_pixels: int,
                batch_size: int = None,
                indices: list[int] = None,
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


    elif model_name == "deepsphere_resnet":

        assert batch_size is not None, "batch_size is required for deepsphere_resnet"
        assert indices is not None, "indices are required for deepsphere_resnet"

        from .resnet import ResnetDeepSphereRegressor
        constructor_args = {
            "n_side": nside,
            "indices": indices,
            "batch_size": batch_size,
            "n_channels": num_channels,
            "out_features": num_targets,
        }
        constructor_args.update(model_args or {})
        model = ResnetDeepSphereRegressor(
            **constructor_args,
        )

        LOGGER.info(f"Built ResnetDeepSphereRegressor {model_name}")
        LOGGER.info(f"  num_channels: {num_channels}, num_targets: {num_targets}")
        LOGGER.info(f"  n_side: {nside}, n_side_down: {nside_down}, num_pixels: {num_pixels}")
        LOGGER.info(f"  model_args: {constructor_args}")

    else:

        raise ValueError(f"Model {model_name} not supported")


    return model
