"""Build WebDataset shards from post-processed CosmoGrid full-sky maps.

The expensive forward-model dependencies are imported only when this workflow
is run.  This keeps the rest of the training package importable on machines
which do not have the private ``msfm`` forward-model package installed.
"""

from __future__ import annotations

import time
from contextlib import ExitStack
from dataclasses import dataclass, replace
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

from .utils.config import Config, ConfigPaths, load_config, load_pixel_indices
from .utils.logger import get_logger

LOGGER = get_logger(__file__)


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
            raise ValueError(
                "Missing forward_model.webdataset setting(s): " + ", ".join(missing)
            )
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


def _forward_model_modules() -> SimpleNamespace:
    """Load modules supplied by the separate forward-model installation."""
    try:
        return SimpleNamespace(
            cosmogrid=import_module("msfm.utils.cosmogrid"),
            filenames=import_module("msfm.utils.filenames"),
            lensing=import_module("msfm.utils.lensing"),
            prior=import_module("msfm.utils.prior"),
            postprocessing=import_module("msfm.postprocessing"),
            webdataset=import_module("webdataset"),
        )
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "The webdataset workflow requires the 'webdataset' and private "
            "'msfm' forward-model packages. Install them in this environment."
        ) from error


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
    config = Config.from_mapping(raw_config)
    configured = config.forward_model.get("webdataset", {})
    if not isinstance(configured, dict):
        raise TypeError("forward_model.webdataset must be a mapping.")
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
    # Validate values supplied as overrides as well as values read from YAML.
    settings = WebDatasetSettings.from_mapping(
        {name: getattr(settings, name) for name in settings.__dataclass_fields__}
    )
    return build_webdataset(config.forward_model, settings, raw_config=raw_config)


def build_webdataset(
    forward_model: dict[str, Any],
    settings: WebDatasetSettings,
    *,
    raw_config: dict[str, Any] | None = None,
    modules: SimpleNamespace | None = None,
) -> int:
    """Create shards and return the total number of written examples."""
    import numpy as np
    import torch

    modules = modules or _forward_model_modules()
    indices = parse_indices(settings.indices)
    output_dir = settings.output_dir / "debug" if settings.debug else settings.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(raw_config or {"forward_model": forward_model}, handle, sort_keys=False)

    delay = 0.0 if settings.debug else float(np.random.uniform(0, settings.max_sleep))
    LOGGER.info("Waiting %.2fs before starting I/O", delay)
    time.sleep(delay)

    files = forward_model["files"]
    analysis = forward_model["analysis"]
    survey = forward_model["survey"]
    meta_info_file = Path(files["meta_info"])
    cosmo_params_info = modules.cosmogrid.get_cosmo_params_info(str(meta_info_file), "grid")
    cosmo_dirs = [path.decode() if isinstance(path, bytes) else str(path) for path in cosmo_params_info["path_par"]]
    cosmo_dirs_in = [settings.input_dir / "grid" / path for path in cosmo_dirs]
    n_cosmos = len(cosmo_dirs)
    if n_cosmos % settings.n_cosmos_per_file:
        raise ValueError(
            f"{n_cosmos} cosmologies cannot be divided into files of "
            f"{settings.n_cosmos_per_file}."
        )

    pixel_indices = load_pixel_indices(forward_model)
    n_patches = int(analysis["n_patches"])
    n_perms = int(analysis["grid"]["n_perms_per_cosmo"])
    maps_to_store = (
        survey["WL"]["map_types"]["onthefly_store"]
        + survey["GC"]["map_types"]["onthefly_store"]
    )
    samples = {"kg": "WL", "ia": "WL", "gg": "WL", "ga": "WL", "ds": "WL", "gd": "WL", "dg": "GC", "qg": "GC"}
    args = SimpleNamespace(
        dir_in=str(settings.input_dir),
        dir_out=str(output_dir),
        cosmogrid_version=settings.cosmogrid_version,
        debug=settings.debug,
    )
    total = 0
    for index in indices:
        start = (index % n_cosmos) * settings.n_cosmos_per_file % n_cosmos
        stop = start + settings.n_cosmos_per_file
        permutation = index * (n_patches * n_perms) // n_cosmos
        if permutation >= n_perms:
            raise ValueError(f"Index {index} selects permutation {permutation}, but only {n_perms} exist.")
        with ExitStack() as stack:
            writers = []
            for patch in range(n_patches):
                filename = modules.filenames.get_filename_webdataset(
                    str(output_dir),
                    tag=f"{survey['name']}_patch{patch:02d}{settings.file_suffix}",
                    index=index,
                    simset="grid",
                    with_bary=bool(analysis["modelling"]["baryonified"]),
                )
                writers.append(stack.enter_context(modules.webdataset.TarWriter(filename, encoder=True)))

            for i_cosmo, cosmo_dir in zip(range(start, stop), cosmo_dirs_in[start:stop], strict=True):
                cosmo = modules.prior.get_hard_parameters(forward_model, cosmo_params_info, i_cosmo)
                i_sobol = int(cosmo_dir.name[-7:-1])
                full_maps_file = modules.postprocessing._get_full_sky_perm(args, forward_model, str(cosmo_dir), permutation)
                full_maps = get_postprocessed_maps(forward_model, full_maps_file, modules=modules)
                for patch in range(n_patches):
                    stored_maps = []
                    channels = []
                    for map_name in maps_to_store:
                        bins = []
                        for redshift_bin, full_map in enumerate(full_maps[map_name]):
                            cutout = modules.postprocessing.full_sky_to_patch(
                                full_map, forward_model, pixel_indices, redshift_bin, patch, sample=samples[map_name]
                            )
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


def get_postprocessed_maps(
    forward_model: dict[str, Any], full_maps_file: str | Path, *, modules: SimpleNamespace | None = None
) -> dict[str, list[Any]]:
    """Read derived full-sky maps for all configured redshift bins."""
    import numpy as np

    modules = modules or _forward_model_modules()
    analysis = forward_model["analysis"]
    survey = forward_model["survey"]
    n_side = int(analysis["n_side"])
    hp_data = str(forward_model["files"]["healpy_data"])
    kappa_to_gamma, _, _ = modules.lensing.get_kaiser_squires_factors(3 * n_side - 1)
    maps: dict[str, list[np.ndarray]] = {name: [] for name in ("kg", "ia", "gg", "ga", "gd", "ds", "dg", "qg")}

    for redshift_bin in survey["WL"]["z_bins"]:
        
        ##
        ## Lensing convergence
        ##

        kg = modules.postprocessing._read_full_sky_bin(forward_model, full_maps_file, "kg", redshift_bin)
        maps["kg"].append(kg.astype(np.float32))

        ##
        ## Linear intrinsic alignment convergence
        ##

        # kappa to shear conversion for intrinsic alignment
        ia = modules.postprocessing._read_full_sky_bin(forward_model, full_maps_file, "ia", redshift_bin)
        maps["ia"].append(ia.astype(np.float32))

        ##
        ## Source sample galaxy counts
        ##

        # source sample galaxy counts for shape noise
        ds = modules.postprocessing._read_full_sky_bin(forward_model, full_maps_file, "dg", redshift_bin)
        maps["ds"].append(ds.astype(np.float32))

        ##
        ## Lensing shear
        ##

        # kappa to shear conversion for lensing signal
        g1, g2 = modules.lensing.kappa_to_gamma(kg, hp_data, kappa_to_gamma, n_side)
        maps["gg"].append((g1 + 1j * g2).astype(np.complex64))
        
        ##
        ## Linear intrinsic alignment shape
        ##

        # kappa to shear conversion for intrinsic alignment
        g1, g2 = modules.lensing.kappa_to_gamma(ia, hp_data, kappa_to_gamma, n_side)
        ga = g1 + 1j * g2
        maps["ga"].append(ga.astype(np.complex64))
        

        ##
        ## Delta-NLA intrinsic alignment
        ##

        # delta-NLA component approximation
        dg = ga * ds
        maps["gd"].append(dg.astype(np.complex64))

    for redshift_bin in survey["GC"]["z_bins"]:


        ##
        ## Galaxy counts
        ##

        dg = modules.postprocessing._read_full_sky_bin(forward_model, full_maps_file, "dg", redshift_bin)
        maps["dg"].append(dg.astype(np.float32))

        ##
        ## Quadratic galaxy counts
        ##

        # quadratic galaxy counts for shape noise
        qg = dg**2
        maps["qg"].append(qg.astype(np.float32))

    return maps
