"""Build WebDataset shards from post-processed CosmoGrid full-sky maps.
"""

from __future__ import annotations

import os
import time
from contextlib import ExitStack
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from .utils.config import Config, ConfigPaths, load_config, load_pixel_file
from .utils.logger import get_logger

LOGGER = get_logger(__file__)


def full_sky_to_patch(
    m: Any,
    conf: dict[str, Any],
    pixel_file: tuple[Any, dict[str, Any], dict[str, Any], Any],
    i_z: int,
    i_patch: int,
    sample: str,
) -> Any:
    """Reorder one full-sky map into a flattened survey patch.

    ``m`` has shape ``(12 * n_side**2,)``.  The returned array has shape
    ``(n_data_vector_pixels,)``.  Complex maps encode shear as ``gamma1 +
    1j * gamma2``; reflected patches therefore require a gamma2 sign change.
    """
    import healpy as hp
    import numpy as np

    # Allocate a full HEALPix workspace with shape ``(12 * n_side**2,)``.
    n_side = conf["analysis"]["n_side"]
    n_pix = hp.nside2npix(n_side)
    data_vec_pix, patches_pix_dict, corresponding_pix_dict, gamma2_signs = pixel_file
    patches_pix = patches_pix_dict[sample][i_z]
    corresponding_pix = corresponding_pix_dict[sample][i_z]
    data_vec_len = len(data_vec_pix)
    base_patch_pix = patches_pix[0]

    # Map the selected patch to the canonical patch's pixel positions.  Both
    # index arrays have shape ``(n_patch_pixels,)``.
    patch_pix = patches_pix[i_patch]
    m_patch = np.zeros(n_pix, dtype=m.dtype)
    m_patch[base_patch_pix] = m[patch_pix]

    # Rotations preserve shear components in this convention, whereas a
    # reflection flips gamma2 only.  This handles all complex precisions.
    if np.issubdtype(m.dtype, np.complexfloating):
        imag_sign = gamma2_signs[i_patch]
        LOGGER.debug(f"Using imag sign {imag_sign} for patch index {i_patch}")
        m_patch = m_patch.real + 1j * m_patch.imag * imag_sign

    # Compress the canonical full-sky workspace into shape
    # ``(n_data_vector_pixels,)`` using the precomputed output indices.
    m_dv = np.zeros(data_vec_len, dtype=m_patch.dtype)
    m_dv[corresponding_pix] = m_patch[base_patch_pix]
    return m_dv


def read_full_sky_bin(conf: dict[str, Any], full_maps_file: str | Path, in_map_type: str, z_bin: Any) -> Any:
    """Read one HEALPix redshift-bin map with shape ``(12 * n_side**2,)``."""
    import h5py
    import healpy as hp

    # Determine the required one-dimensional HEALPix map length.
    n_side = conf["analysis"]["n_side"]
    n_pix = hp.nside2npix(n_side)

    # Load the one-dimensional map and resample it when its stored n_side does
    # not match the analysis n_side.
    LOGGER.timer.start("load_map")
    map_dir = f"map/{in_map_type}/{z_bin}"
    with h5py.File(full_maps_file, "r") as f:
        map_full = f[map_dir][:]

        # ud_grade if the stored map is at a different resolution than the analysis n_side
        if map_full.shape[0] != n_pix:
            map_full = hp.ud_grade(map_full, nside_out=n_side, order_in="RING", order_out="RING", pess=True)

    LOGGER.debug(f"Loaded {map_dir} from {full_maps_file} after {LOGGER.timer.elapsed('load_map')}")
    return map_full


def convert_kappa_to_gamma_alm(kappa_full_sky: Any, hp_datapath: str, kappa2gamma_fac: Any, n_side: int) -> tuple[Any, Any]:
    """Convert a convergence map into two full-sky shear component maps."""
    import healpy as hp
    import numpy as np

    # Transform the input map, shape ``(12 * n_side**2,)``, to packed spherical
    # harmonic coefficients, shape ``(n_alm,)``.
    kappa_alm = hp.map2alm(
        kappa_full_sky,
        use_pixel_weights=True,
        datapath=hp_datapath,
    )

    # Apply the element-wise Kaiser--Squires factors; all three packed arrays
    # have shape ``(n_alm,)``.
    gamma_alm = kappa_alm * kappa2gamma_fac
    dummy_alm = np.zeros_like(gamma_alm)
    _, gamma1_full, gamma2_full = hp.alm2map([dummy_alm, gamma_alm, dummy_alm], nside=n_side)

    # Each real component map has shape ``(12 * n_side**2,)``.
    return gamma1_full, gamma2_full


def get_kaiser_squires_factors(l_max: int) -> tuple[Any, Any, Any]:
    """Factors for a spherical Kaiser Squires transformation
    from eq. (11) in https://academic.oup.com/mnras/article/505/3/4626/6287258
    """
    import healpy as hp
    import numpy as np

    # ``l`` and every returned factor array have packed shape ``(n_alm,)``,
    # where ``n_alm = hp.Alm.getsize(l_max)``.
    multipoles = hp.Alm.getlm(l_max)[0]

    kappa2gamma_fac = np.where(
        np.logical_and(multipoles != 1, multipoles != 0),
        -np.sqrt(((multipoles + 2.0) * (multipoles - 1)) / ((multipoles + 1) * multipoles)),
        0,
    )
    gamma2kappa_fac = np.where(
        np.logical_and(multipoles != 1, multipoles != 0),
        1 / kappa2gamma_fac,
        0,
    )
    l_mask_fac = np.where(np.logical_and(multipoles != 1, multipoles != 0), 1.0, 0.0)

    return kappa2gamma_fac, gamma2kappa_fac, l_mask_fac


def get_full_sky_perm(cosmogrid_version: str, conf: dict[str, Any], cosmo_dir_in: str | Path, i_perm: int) -> str:
    """Return the conventional CosmoGrid full-map filename for a permutation."""
    with_bary = conf["analysis"]["modelling"]["baryonified"]

    # Select the established CosmoGrid filename for the requested simulation
    # version and dark-matter-only (DMO) or baryonified (DMB) variant.
    perm_dir_in = os.path.join(cosmo_dir_in, f"perm_{i_perm:04d}")
    filenames = {
        ("1", False): "projected_probes_maps_nobaryons512.h5",
        ("1", True): "projected_probes_maps_baryonified512.h5",
        ("1.1", False): "projected_probes_maps_v11dmo.h5",
        ("1.1", True): "projected_probes_maps_v11dmb.h5",
    }
    full_maps_file = os.path.join(perm_dir_in, filenames[(cosmogrid_version, with_bary)])

    return full_maps_file


def get_hard_parameters(conf, cosmo_params_info, i_cosmo):
    """Return cosmological (and optional baryonic) parameters, shape ``(n_parameters,)``."""
    import numpy as np

    # Preserve the configured parameter order because it defines tensor axes.
    cosmo_params = conf["analysis"]["params"]["cosmo"].copy()
    baryonified = conf["analysis"]["modelling"]["baryonified"]
    if baryonified:
        cosmo_params += conf["analysis"]["params"]["bary"]
    cosmo = [cosmo_params_info[cosmo_param][i_cosmo] for cosmo_param in cosmo_params]
    cosmo = np.array(cosmo, dtype=np.float32)
    return cosmo


def get_filename_webdataset(out_dir: str | Path, index: int, tag: str, simset: str, with_bary: bool = False) -> str:
    """Build a stable WebDataset shard path."""
    # Encode all dataset-disambiguating fields in the basename.
    matter_label = "dmb" if with_bary else "dmo"
    filename = f"{tag}_{simset}_{matter_label}_{index:04d}.tar"
    return os.path.join(out_dir, filename)


def get_cosmo_params_info(meta_info_file, simset="grid"):
    """Returns directories on the level of cosmo_000001 and cosmo_delta_H0_p and so on

    Args:
        meta_info_file (str): path to the modified (in permutations_list.ipynb) CosmoGridV1 metainfo file
        simset (str, optional): Either "grid" or "fiducial". Defaults to 'grid'.

    Returns:
        ndarray: List containing the metainfo for all the unique cosmological parameters
    """
    import h5py

    # The structured array has shape ``(n_cosmologies,)`` and includes one
    # field per parameter plus the ``path_par`` directory field.
    with h5py.File(meta_info_file, "r") as f:
        params_info = f[f"parameters/{simset}"][:]

    return params_info


@dataclass(frozen=True, slots=True)
class WebDatasetSettings:
    """Execution settings stored under ``forward_model.webdataset``."""

    input_dir: Path
    output_dir: Path
    indices: str | list[int] = "0"
    cosmogrid_version: str = "1.1"
    file_suffix: str = ""
    max_sleep: float = 120.0
    n_cosmos_per_file: int = 25
    debug: bool = False

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> WebDatasetSettings:
        """Validate and normalize the YAML workflow settings."""
        missing = [name for name in ("input_dir", "output_dir") if not values.get(name)]
        if missing:
            print('values', values)
            raise ValueError("Missing forward_model.webdataset setting(s): " + ", ".join(missing))
        settings = cls(
            input_dir=Path(values["input_dir"]),
            output_dir=Path(values["output_dir"]),
            indices=values.get("indices", "0"),
            cosmogrid_version=str(values.get("cosmogrid_version", "1.1")),
            file_suffix=str(values.get("file_suffix", "")),
            max_sleep=float(values.get("max_sleep", 120)),
            n_cosmos_per_file=int(values.get("n_cosmos_per_file", 25)),
            debug=bool(values.get("debug", False)),
        )
        if settings.n_cosmos_per_file <= 0:
            raise ValueError("n_cosmos_per_file must be positive.")
        if settings.max_sleep < 0:
            raise ValueError("max_sleep must be non-negative.")
        if settings.cosmogrid_version not in {"1", "1.1"}:
            raise ValueError("cosmogrid_version must be '1' or '1.1'.")
        return settings


def parse_indices(value: str | list[int]) -> list[int]:
    """Parse comma-separated indices and inclusive ``start>stop`` ranges."""
    if isinstance(value, list):
        indices = value
    else:
        indices = []
        for part in value.split(","):
            part = part.strip()
            if not part:
                continue
            if ">" in part:
                start_text, stop_text = part.split(">", maxsplit=1)
                start, stop = int(start_text), int(stop_text)
                if stop < start:
                    raise ValueError(f"Invalid descending index range: {part}")
                indices.extend(range(start, stop + 1))
            else:
                indices.append(int(part))
    if not indices or any(index < 0 for index in indices):
        raise ValueError("indices must contain at least one non-negative integer.")
    return indices


def webdataset_from_config(
    paths: ConfigPaths,
    *,
    input_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    indices: str | None = None,
    cosmogrid_version: str | None = None,
    file_suffix: str | None = None,
    max_sleep: float | None = None,
    n_cosmos_per_file: int | None = None,
    debug: bool | None = None,
) -> int:
    """Load the merged application config and create the requested shards.

    Keyword arguments are command-line overrides.  Unspecified values come
    from ``forward_model.webdataset`` in the merged YAML configuration.
    """
    raw_config = load_config(paths)
    print('raw_config', raw_config)
    config = Config.from_mapping(raw_config)
    print('config', config)

    webdataset_default = {
          "input_dir": "/capstor/store/cscs/swissai/a0158/tomaszk/CosmoGridV1",
          "output_dir": "/capstor/scratch/cscs/tomaszk/260205_euclid_multiprobe_sbi/webdataset",
          "indices": "0",
          "cosmogrid_version": "1.1",
          "file_suffix": "",
          "max_sleep": 0,
          "n_cosmos_per_file": 25,
          "debug": False
    }

    configured = config.forward_model.get("webdataset", webdataset_default)
    if not isinstance(configured, dict):
        raise TypeError("forward_model.webdataset must be a mapping.")
    print('configured', configured)
    settings = WebDatasetSettings.from_mapping(configured)
    overrides = {
        "input_dir": Path(input_dir) if input_dir is not None else None,
        "output_dir": Path(output_dir) if output_dir is not None else None,
        "indices": indices,
        "cosmogrid_version": cosmogrid_version,
        "file_suffix": file_suffix,
        "max_sleep": max_sleep,
        "n_cosmos_per_file": n_cosmos_per_file,
        "debug": debug,
    }
    settings = replace(settings, **{key: value for key, value in overrides.items() if value is not None})
    print('settings', settings)

    # Validate values supplied as overrides as well as values read from YAML.
    settings = WebDatasetSettings.from_mapping({name: getattr(settings, name) for name in settings.__dataclass_fields__})

    
    print('raw_config', raw_config)
    return build_webdataset(config.forward_model, settings, raw_config=raw_config)


def build_webdataset(
    forward_model: dict[str, Any],
    settings: WebDatasetSettings,
    *,
    raw_config: dict[str, Any] | None = None,
) -> int:
    """Create shards and return the total number of written examples."""
    import numpy as np
    import torch
    import webdataset

    # Resolve requested shard indices and persist the effective configuration
    # alongside the output for reproducibility.
    indices = parse_indices(settings.indices)
    output_dir = settings.output_dir / "debug" if settings.debug else settings.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(raw_config or {"forward_model": forward_model}, handle, sort_keys=False)

    # Stagger independent jobs to avoid a burst of simultaneous filesystem I/O.
    delay = 0.0 if settings.debug else float(np.random.uniform(0, settings.max_sleep))
    LOGGER.info("Waiting %.2fs before starting I/O", delay)
    time.sleep(delay)

    # Read the structured cosmology table, shape ``(n_cosmologies,)``, and
    # turn its path field into one input directory per cosmology.
    files = forward_model["files"]
    analysis = forward_model["analysis"]
    survey = forward_model["survey"]
    meta_info_file = Path(files["meta_info"])
    cosmo_params_info = get_cosmo_params_info(str(meta_info_file), "grid")
    cosmo_dirs = [path.decode() if isinstance(path, bytes) else str(path) for path in cosmo_params_info["path_par"]]
    cosmo_dirs_in = [settings.input_dir / path for path in cosmo_dirs]
    n_cosmos = len(cosmo_dirs)
    if n_cosmos % settings.n_cosmos_per_file:
        raise ValueError(f"{n_cosmos} cosmologies cannot be divided into files of {settings.n_cosmos_per_file}.")

    # Load patch lookup arrays and collect the configured map channel names.
    pixel_indices = load_pixel_file(forward_model)
    n_patches = int(analysis["n_patches"])
    n_perms = int(analysis["grid"]["n_perms_per_cosmo"])
    maps_to_store = survey["WL"]["map_types"]["onthefly_store"] + survey["GC"]["map_types"]["onthefly_store"]
    samples = {"kg": "WL", "ia": "WL", "gg": "WL", "ga": "WL", "ds": "WL", "gd": "WL", "dg": "GC", "qg": "GC"}
    total = 0
    # Each requested index selects a contiguous cosmology block, one
    # permutation, and one output tar writer per sky patch.
    for index in indices:
        start = (index % n_cosmos) * settings.n_cosmos_per_file % n_cosmos
        stop = start + settings.n_cosmos_per_file
        permutation = index * (n_patches * n_perms) // n_cosmos
        if permutation >= n_perms:
            raise ValueError(f"Index {index} selects permutation {permutation}, but only {n_perms} exist.")
        with ExitStack() as stack:
            writers = []
            for patch in range(n_patches):
                filename = get_filename_webdataset(
                    str(output_dir),
                    tag=f"{survey['name']}_patch{patch:02d}{settings.file_suffix}",
                    index=index,
                    simset="grid",
                    with_bary=bool(analysis["modelling"]["baryonified"]),
                )
                writers.append(stack.enter_context(webdataset.TarWriter(filename, encoder=True)))

            # Process every cosmology once and write its patches to their
            # respective tar archives.
            for i_cosmo, cosmo_dir in zip(range(start, stop), cosmo_dirs_in[start:stop], strict=True):
                cosmo = get_hard_parameters(forward_model, cosmo_params_info, i_cosmo)
                i_sobol = int(cosmo_dir.name.split('_')[-1])
                full_maps_file = get_full_sky_perm(settings.cosmogrid_version, forward_model, str(cosmo_dir), permutation)
                full_maps = get_postprocessed_maps(forward_model, full_maps_file)
                for patch in range(n_patches):
                    # Each stored map starts with shape
                    # ``(n_data_vector_pixels, n_redshift_bins, 1)``.  Complex
                    # shear maps become two real channels in the final axis.
                    stored_maps = []
                    channels = []
                    for map_name in maps_to_store:
                        bins = []
                        for redshift_bin, full_map in enumerate(full_maps[map_name]):
                            cutout = full_sky_to_patch(full_map, forward_model, pixel_indices, redshift_bin, patch, sample=samples[map_name])
                            bins.append(cutout[..., np.newaxis])
                        patch_map = np.concatenate(bins, axis=-1)[..., np.newaxis]
                        if np.issubdtype(patch_map.dtype, np.complexfloating):
                            stored_maps.extend((patch_map.real, patch_map.imag))
                            channels.extend((f"{map_name}1", f"{map_name}2"))
                        elif np.issubdtype(patch_map.dtype, np.floating):
                            stored_maps.append(patch_map)
                            channels.append(map_name)
                        else:
                            raise TypeError(f"Unsupported map dtype for {map_name}: {patch_map.dtype}")

                    # Serialize a map tensor with shape
                    # ``(n_data_vector_pixels, n_redshift_bins, n_channels)``,
                    # metadata with shape ``(7,)``, and parameters with shape
                    # ``(n_parameters,)``.
                    signal = index * n_cosmos * n_perms * n_patches + i_cosmo * n_perms * n_patches + permutation * n_patches + patch
                    sample = {
                        "__key__": f"{signal:09d}",
                        "maps_float32.pth": torch.from_numpy(np.concatenate(stored_maps, axis=-1).astype(np.float32)),
                        "vec_int32.pth": torch.tensor(
                            [signal, i_sobol, i_cosmo, permutation, patch, analysis["n_side"], analysis["n_side_down"]],
                            dtype=torch.int32,
                        ),
                        "vec_float32.pth": torch.as_tensor(cosmo, dtype=torch.float32),
                    }
                    writers[patch].write(sample)
                    total += 1
                    LOGGER.debug("Wrote %s with channels %s", sample["__key__"], channels)
    LOGGER.info("Wrote %d examples", total)
    return total


def get_postprocessed_maps(forward_model: dict[str, Any], full_maps_file: str | Path) -> dict[str, list[Any]]:
    """Read derived full-sky maps for all configured redshift bins."""
    import numpy as np

    # Initialize one list per physical field; each appended full-sky array has
    # shape ``(12 * n_side**2,)``.
    analysis = forward_model["analysis"]
    survey = forward_model["survey"]
    n_side = int(analysis["n_side"])
    hp_data = str(forward_model["files"]["healpy_data"])
    kappa_to_gamma, _, _ = get_kaiser_squires_factors(3 * n_side - 1)
    maps: dict[str, list[np.ndarray]] = {name: [] for name in ("kg", "ia", "gg", "ga", "gd", "ds", "dg", "qg")}

    # Derive all weak-lensing fields for every source redshift bin.
    for redshift_bin in survey["WL"]["z_bins"]:
        ##
        ## Lensing convergence
        ##

        kg = read_full_sky_bin(forward_model, full_maps_file, "kg", redshift_bin)
        maps["kg"].append(kg.astype(np.float32))

        ##
        ## Linear intrinsic alignment convergence
        ##

        # kappa to shear conversion for intrinsic alignment
        ia = read_full_sky_bin(forward_model, full_maps_file, "ia", redshift_bin)
        maps["ia"].append(ia.astype(np.float32))

        ##
        ## Source sample galaxy counts
        ##

        # source sample galaxy counts for shape noise
        ds = read_full_sky_bin(forward_model, full_maps_file, "dg", redshift_bin)
        maps["ds"].append(ds.astype(np.float32))

        ##
        ## Lensing shear
        ##

        # kappa to shear conversion for lensing signal
        g1, g2 = convert_kappa_to_gamma_alm(kg, hp_data, kappa_to_gamma, n_side)
        maps["gg"].append((g1 + 1j * g2).astype(np.complex64))

        ##
        ## Linear intrinsic alignment shape
        ##

        # kappa to shear conversion for intrinsic alignment
        g1, g2 = convert_kappa_to_gamma_alm(ia, hp_data, kappa_to_gamma, n_side)
        ga = g1 + 1j * g2
        maps["ga"].append(ga.astype(np.complex64))

        ##
        ## Delta-NLA intrinsic alignment
        ##

        # delta-NLA component approximation
        dg = ga * ds
        maps["gd"].append(dg.astype(np.complex64))

    # Derive scalar galaxy-clustering fields for every lens redshift bin.
    for redshift_bin in survey["GC"]["z_bins"]:
        ##
        ## Galaxy counts
        ##

        dg = read_full_sky_bin(forward_model, full_maps_file, "dg", redshift_bin)
        maps["dg"].append(dg.astype(np.float32))

        ##
        ## Quadratic galaxy counts
        ##

        # quadratic galaxy counts for shape noise
        qg = dg**2
        maps["qg"].append(qg.astype(np.float32))

    return maps
