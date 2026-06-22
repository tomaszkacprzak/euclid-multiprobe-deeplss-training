from ..utils.logger import get_logger
import healpy as hp

LOGGER = get_logger(__file__)

def get_num_nested_levels(nside: int, nside_down: int) -> int:

    nord = hp.nside2order(nside)
    nord_down = hp.nside2order(nside_down)
    return nord - nord_down


    
def build_model(model_name: str,
                num_channels: int,
                num_targets: int,
                nside: int,
                nside_down: int):

    if model_name == "nested_transformer":

        num_nested_levels = get_num_nested_levels(nside, nside_down)

        from .nested_transfomer import NestedHierarchicalLocalWindowTransformer
        model = NestedHierarchicalLocalWindowTransformer(
            in_channels=num_channels,
            num_outputs=num_targets,
            num_nested_levels=num_nested_levels,
            base_embed_dim=128,
            growth="constant",
            num_heads=4,
            window_levels=3,
            local_blocks_per_level=1,
            global_blocks=1,
            mlp_ratio=4,
        )

        LOGGER.info(f"Built NestedHierarchicalLocalWindowTransformer {model_name}")
        LOGGER.info(f"  num_channels: {num_channels}, num_targets: {num_targets}")


    else:

        raise ValueError(f"Model {model_name} not supported")


    return model

