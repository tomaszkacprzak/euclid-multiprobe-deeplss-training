from .utils.logger import get_logger

LOGGER = get_logger(__file__)

def build_model(model_name: str,
                num_channels: int,
                num_targets: int):

    if model_name == "nested_transformer":

        from .networks.nested_transfomer import NestedHierarchicalLocalWindowTransformer
        model = NestedHierarchicalLocalWindowTransformer(
                    in_channels=num_channels,
                    num_outputs=num_targets,
                    embed_dim=128,
                    num_heads=4,
                    window_levels=3,
                    local_blocks_per_level=1,
                    global_blocks=1,
                    mlp_ratio=4,
                )

        LOGGER.info(f"Built NestedHierarchicalLocalWindowTransformer {model_name}")
        LOGGER.info(f"  num_pixels: {num_pixels}, num_channels: {num_channels}, num_targets: {num_targets}")


    else:

        raise ValueError(f"Model {model_name} not supported")


    return model

