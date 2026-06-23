from ..utils.logger import get_logger
import healpy as hp

LOGGER = get_logger(__file__)




    
def build_model(model_name: str,
                num_channels: int,
                num_targets: int,
                nside: int,
                nside_down: int,
                num_pixels: int):

    if model_name == "nested_transformer":

        from .healpix_transformer import HealpixNestedHierarchicalLocalWindowTransformer
        model = HealpixNestedHierarchicalLocalWindowTransformer(
            nside=nside,
            nside_down=nside_down,
            num_pixels=num_pixels,
            in_channels=num_channels,
            num_outputs=num_targets,
            base_embed_dim=128,
            growth="constant",
            num_heads=4,
            window_levels=3,
            local_blocks_per_level=1,
            global_blocks=1,
            mlp_ratio=4,
        )

        LOGGER.info(f"Built HealpixNestedHierarchicalLocalWindowTransformer {model_name}")
        LOGGER.info(f"  num_channels: {num_channels}, num_targets: {num_targets}")

    elif model_name == "angular_power_spectra":
        from .angular_power_spectra import AngularPowerSpectra
        model = AngularPowerSpectra(
            nside=nside,
            pixel_file=pixel_file,
            lmax=lmax,
            mmax=mmax,
            quad_weights=quad_weights,
            pixel_dataset=pixel_dataset,
        )
        LOGGER.info(f"Built AngularPowerSpectra {model_name}")
        LOGGER.info(f"  num_channels: {num_channels}, num_targets: {num_targets}")


    else:

        raise ValueError(f"Model {model_name} not supported")


    return model

