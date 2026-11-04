"""Calculate correlations for generated training maps."""

from __future__ import annotations

import html
import json
import math
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import h5py
import healpy as hp
import numpy as np
import torch

from .training import load_physics_model_class
from .utils.config import Config, ConfigPaths, config_paths, load_config, load_pixel_file
from .utils.logger import get_logger

LOGGER = get_logger(__file__)

COOLWARM_COLORSCALE = [
    [0.0, "#3b4cc0"],
    [0.1, "#5977e3"],
    [0.2, "#7b9ff9"],
    [0.3, "#9ebeff"],
    [0.4, "#c0d4f5"],
    [0.5, "#dddddd"],
    [0.6, "#f2cbb7"],
    [0.7, "#f7a889"],
    [0.8, "#ee8468"],
    [0.9, "#d65244"],
    [1.0, "#b40426"],
]


def calccorrs(
    config_or_path: ConfigPaths | Mapping[str, Any] | Config,
    *,
    output_dir: Path,
    num_batches_per_file: int = 100,
    file_index: int = 0,
    dataset_split: str = "training",
    device: str | torch.device | None = None,
) -> list[Path]:
    """Write real auto/cross correlations to HDF5, one group per input batch.

    ``correlations`` has shape ``(batch, channels, channels, bins)`` and contains
    the real plus-like correlation for spin fields. ``separations`` uses the
    output bins of the pre-generated Nestcorr pair cache. Labels and source
    indices retain the same batch dimension as the correlation tensors.
    """

    LOGGER.info(f"Calculating correlations {file_index}")
    if num_batches_per_file <= 0:
        raise ValueError("num_batches_per_file must be positive.")

    from nestcorr.pairs import resolve_pair_metadata

    config, raw_config = _coerce_config(config_or_path)
    correlation_config = raw_config.get("calccorrs", config.training.get("calccorrs", {}))
    analysis = config.forward_model["analysis"]
    nside = int(analysis["n_side"])
    nside_down = int(analysis["n_side_down"])
    L = hp.nside2order(nside)
    R = hp.nside2order(nside_down)
    pairs_filename = correlation_config.get("pairs_filename")
    if not pairs_filename:
        raise ValueError("calccorrs requires a pair cache: set calccorrs.pairs_filename.")
    pairs_path = Path(pairs_filename).expanduser()
    if not pairs_path.is_file():
        raise FileNotFoundError(
            f"Pair cache {pairs_path} does not exist. Create it with 'nestcorr generate_pairs {L}' "
            "and set calccorrs.pairs_filename to its path."
        )
    # The file fixes the hierarchy, opening distance, and output bins. Validate
    # it before expensive survey/physics/data initialization.
    pair_metadata = resolve_pair_metadata(pairs_path, L=L)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        LOGGER.info(f"Created output directory {output_dir}")
    else:
        LOGGER.info(f"Using existing output directory {output_dir}")

    from .dataloaders import OntheflyPipeline

    requested_device = device if device is not None else correlation_config.get("device", config.training.get("device"))
    run_device = torch.device(requested_device or ("cuda" if torch.cuda.is_available() else "cpu"))
    LOGGER.info(f"Running on {run_device}")
    indices = np.asarray(load_pixel_file(config.forward_model)[0], dtype=np.int64)
    footprint_indices = get_footprint_indices(indices, L, R)
    LOGGER.info(f"indices min={indices.min()}, max={indices.max()}")
    LOGGER.info(f"footprint_indices min={footprint_indices.min()}, max={footprint_indices.max()}")
    LOGGER.info(f"L={L}, R={R}, footprint_indices {len(footprint_indices)}/{hp.nside2npix(nside_down)}")

    OntheflyPhysicsModel = load_physics_model_class(config.training['physics_model'])
    physics_model = OntheflyPhysicsModel(
        config.forward_model,
        scalers=False,
        device=run_device,
        seed=file_index * 1001,
        nside=nside,
        **config.training.get('physics_model_args', {}),
    ).to(run_device)
    loader = OntheflyPipeline(
        webds_pattern=config.training['records_pattern'],
        batch_size=config.training['batch_size'],
        physics_model=physics_model,
        downsampler=None,
        smoother=None,
        num_workers=config.training['num_workers'],
        device=run_device,
        seed=file_index * 1000,
        validation=dataset_split == "validation",
    )

    examples_written = 0
    compression_args = {"compression": "lzf", "shuffle": True}

    from nestcorr.nestcorr import NestcorrFastFootprint

    LOGGER.info(f"Using NestcorrFastFootprint with L={L}")
    spins = torch.as_tensor(physics_model.channel_spins).detach().cpu().tolist()
    correlator = NestcorrFastFootprint(
        L,
        spins,
        R_footprint=R,
        matmul_precision="high",
        footprint_indices=footprint_indices,
        symmetrize=True,
        pairs_filename=pairs_path,
    ).to(run_device)

    time_start = time.time()
    time_corr = 0
    with torch.no_grad():
        shard_path = os.path.join(output_dir, f"corrs_{dataset_split}_{file_index:06d}.h5")
        with h5py.File(shard_path, "w") as f:
            f.attrs["pairs_filename"] = str(pairs_path.resolve())
            f.attrs["fine_order"] = L
            f.attrs["neighbour_distance"] = pair_metadata["nd"]
            f.attrs["output_binning"] = pair_metadata["binning"]
            f.attrs["num_output_bins"] = correlator.num_output_bins
            f.attrs["separation_units"] = "radian"
            if pair_metadata["data_vec_pix"] is not None:
                f.attrs["selected_pixel_count"] = len(pair_metadata["data_vec_pix"])
            if pair_metadata["binning"] == "angular":
                f.attrs["bin_order"] = "increasing angular interval"
                f.attrs["bins_per_octave"] = pair_metadata["bins_per_octave"]
                f.create_dataset("angular_bin_edges", data=pair_metadata["angular_bin_edges"])
            else:
                f.attrs["bin_order"] = "descending inherited output order (L, L-1, ..., 0)"
                f.create_dataset("output_orders", data=np.arange(L, -1, -1))
            f.create_dataset("separations", data=correlator.theta.cpu().numpy())
            f.create_dataset("theta_range", data=correlator.theta_range.cpu().numpy())
            LOGGER.info(
                f"Writing correlations to {shard_path}, starting {num_batches_per_file} batches per file "
                f"with {config.training['batch_size']} examples per batch"
            )

            for batch_index, (maps, labels, inds) in enumerate(loader):
                # raw batch of shape (batch_size, num_pixels, num_channels)
                map_dtype = torch.complex64 if torch.is_complex(maps) else torch.float32
                maps = maps.to(device=run_device, dtype=map_dtype)

                # calculate wegiths used for correlations
                # weights of shape (batch_size, num_pixels, num_channels)
                weights = physics_model.get_weight_maps(maps)

                # convert the raw counts to density contrast, subtract the mean from shear/kappa maps
                # (batch_size, num_pixels, num_channels)
                maps = physics_model.preprocess_for_correlations(maps)

                if batch_index == 0:
                    LOGGER.info(f"maps    shape={maps.shape}    dtype={maps.dtype}")
                    LOGGER.info(f"weights shape={weights.shape} dtype={weights.dtype}")

                # Main magic - calculate the correlations
                maps = torch.movedim(maps, -1, 1)  # -> (batch_size, num_channels, num_pixels)
                weights = torch.movedim(weights, -1, 1)  # -> (batch_size, num_channels, num_pixels)
                time_start_corr = time.time()

                # main magic - calculate the correlations
                # The cache determines the bin count: L+1 for standard bins,
                # or its saved angular intervals for angular bins.
                correlations = correlator(maps, weights)

                time_corr += time.time() - time_start_corr
                separations = correlator.theta
                LOGGER.info(
                    f"Batch {batch_index + 1}: maps.shape={maps.shape} correlations.shape={correlations.shape}, separations={separations.shape}"
                )

                # write the separations and example correlations to a separate file

                f.create_dataset(f"batch{batch_index:04d}/separations", data=separations.cpu().numpy(), **compression_args)
                f.create_dataset(f"batch{batch_index:04d}/correlations", data=correlations.cpu().numpy(), **compression_args)
                f.create_dataset(f"batch{batch_index:04d}/labels", data=labels.cpu().numpy(), **compression_args)
                f.create_dataset(f"batch{batch_index:04d}/inds", data=inds.cpu().numpy(), **compression_args)

                examples_written += maps.shape[0]

                if batch_index == num_batches_per_file - 1:
                    break

    LOGGER.info(f"Wrote correlations in {shard_path} for {examples_written} examples")
    LOGGER.info(f"Time total: {time.time() - time_start:.2f}s, time corr: {time_corr:.2f}s")

    dashboard_path = Path(shard_path).with_suffix(".html")
    create_correlations_dashboard(
        shard_path,
        dashboard_path,
        parameter_names=[str(name) for name in physics_model.params],
        model_information={"physics_model": config.training['physics_model'], "forward_model": config.forward_model},
    )
    LOGGER.info("Wrote interactive correlations dashboard to %s", dashboard_path)
    return [Path(shard_path)]


def create_correlations_dashboard(
    correlations_path: str | Path,
    dashboard_path: str | Path,
    *,
    parameter_names: Sequence[str],
    model_information: Mapping[str, Any] | None = None,
) -> Path:
    """Create a self-contained Plotly dashboard from all batches in a correlation file."""
    import plotly.graph_objects as go
    import plotly.io as pio
    from plotly.colors import sample_colorscale
    from plotly.subplots import make_subplots

    correlations_by_batch = []
    labels_by_batch = []
    separations: np.ndarray | None = None
    with h5py.File(correlations_path, "r") as source:
        bin_information = {
            name: value.item() if isinstance(value, np.generic) else value
            for name, value in source.attrs.items()
        }
        for name in ("angular_bin_edges", "output_orders"):
            if name in source:
                bin_information[name] = source[name][:].tolist()
        batch_names = sorted(name for name in source if name.startswith("batch"))
        if not batch_names:
            raise ValueError("No batch groups were found in the correlations file.")
        for batch_name in batch_names:
            group = source[batch_name]
            batch_correlations = np.asarray(group["correlations"])
            batch_labels = np.asarray(group["labels"])
            batch_separations = np.asarray(group["separations"]).reshape(-1)
            if batch_correlations.ndim != 4 or batch_correlations.shape[1] != batch_correlations.shape[2]:
                raise ValueError(
                    f"{batch_name}/correlations must have shape (batch, channels, channels, correlations), got {batch_correlations.shape}."
                )
            if batch_labels.ndim != 2 or batch_labels.shape[0] != batch_correlations.shape[0]:
                raise ValueError(f"{batch_name}/labels must be 2D and have the same batch size as correlations.")
            if batch_separations.size != batch_correlations.shape[-1]:
                raise ValueError(f"{batch_name}/separations length does not match the correlation dimension.")
            if separations is None:
                separations = batch_separations
            elif not np.array_equal(separations, batch_separations, equal_nan=True):
                raise ValueError("All batches must use identical separations.")
            correlations_by_batch.append(batch_correlations)
            labels_by_batch.append(batch_labels)

    correlations = np.concatenate(correlations_by_batch, axis=0)
    labels = np.concatenate(labels_by_batch, axis=0)
    if labels.shape[1] != len(parameter_names):
        raise ValueError(f"Physics model supplies {len(parameter_names)} parameter names for {labels.shape[1]} label columns.")
    num_channels = correlations.shape[1]
    if any(values.shape[1:] != correlations.shape[1:] for values in correlations_by_batch):
        raise ValueError("All batches must have the same channel and correlation dimensions.")

    upper_rows, upper_columns = np.triu_indices(num_channels)
    unique_correlations = correlations[:, upper_rows, upper_columns, :]
    pair_names = [f"Channels {first} × {second}" for first, second in zip(upper_rows, upper_columns, strict=True)]
    num_pairs = len(pair_names)
    columns = math.ceil(math.sqrt(num_pairs))
    rows = math.ceil(num_pairs / columns)
    figure = make_subplots(rows=rows, cols=columns, subplot_titles=pair_names)
    figure.update_xaxes(type="log")

    colors_by_parameter = []
    for parameter_index in range(labels.shape[1]):
        values = labels[:, parameter_index]
        low, high = float(np.nanmin(values)), float(np.nanmax(values))
        normalized = np.zeros_like(values, dtype=float) if high == low else (values - low) / (high - low)
        colors_by_parameter.append(list(sample_colorscale(COOLWARM_COLORSCALE, normalized, colortype="rgb")))

    trace_examples = []
    for pair_index, pair_values in enumerate(np.moveaxis(unique_correlations, 1, 0)):
        row, column = divmod(pair_index, columns)
        for example_index, curve in enumerate(pair_values):
            figure.add_trace(
                go.Scattergl(
                    x=separations,
                    y=curve,
                    mode="lines",
                    line={"color": colors_by_parameter[0][example_index], "width": 1},
                    name=f"Example {example_index}",
                    showlegend=False,
                    hovertemplate=f"example={example_index}<br>separation=%{{x:.6g}}<br>correlation=%{{y:.6g}}<extra></extra>",
                ),
                row=row + 1,
                col=column + 1,
            )
            trace_examples.append(example_index)
        units = bin_information.get("separation_units")
        separation_title = f"Separation ({units})" if units else "Separation"
        figure.update_xaxes(title_text=separation_title if row == rows - 1 else None, row=row + 1, col=column + 1)
        figure.update_yaxes(title_text="Correlation" if column == 0 else None, row=row + 1, col=column + 1)

    figure.add_trace(
        go.Scatter(
            x=[None] * len(labels),
            y=[None] * len(labels),
            mode="markers",
            marker={"color": labels[:, 0], "colorscale": COOLWARM_COLORSCALE, "showscale": True, "colorbar": {"title": parameter_names[0]}},
            showlegend=False,
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )
    line_trace_count = len(trace_examples)
    trace_indices = list(range(line_trace_count + 1))
    buttons = []
    for parameter_index, parameter_name in enumerate(parameter_names):
        values = labels[:, parameter_index]
        low, high = float(np.nanmin(values)), float(np.nanmax(values))
        buttons.append(
            {
                "label": parameter_name,
                "method": "restyle",
                "args": [
                    {
                        "line.color": [*[colors_by_parameter[parameter_index][example] for example in trace_examples], "rgba(0,0,0,0)"],
                        "marker.color": [*[None] * line_trace_count, values],
                        "marker.cmin": [*[low] * line_trace_count, low],
                        "marker.cmax": [*[high] * line_trace_count, high],
                        "marker.colorbar.title.text": [*[parameter_name] * line_trace_count, parameter_name],
                    },
                    trace_indices,
                ],
            }
        )
    figure.update_layout(
        title={"text": "Two-point Correlations Across Cosmologies", "x": 0.5},
        height=max(500, 360 * rows),
        template="plotly_white",
        hovermode="closest",
        updatemenus=[{"buttons": buttons, "direction": "down", "x": 0, "xanchor": "left", "y": 1.12}],
        annotations=[
            *figure.layout.annotations,
            {"text": "Color parameter:", "showarrow": False, "x": 0, "xanchor": "left", "xref": "paper", "y": 1.17, "yref": "paper"},
        ],
    )

    plot_html = pio.to_html(figure, full_html=False, include_plotlyjs=True, config={"responsive": True})
    information = dict(model_information or {})
    information["correlation_bins"] = bin_information
    information = html.escape(json.dumps(information, indent=2, default=str))
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Correlations Dashboard</title></head><body>
<h1>Correlations Dashboard</h1>
<details><summary>Model information</summary><pre>{information}</pre></details>
{plot_html}
</body></html>"""
    destination = Path(dashboard_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(document, encoding="utf-8")
    return destination


def get_footprint_indices(indices, L, R):
    assert np.all(np.diff(indices) >= 0), "indices must be sorted"
    return np.unique(np.asarray(indices) // ((hp.order2nside(L) // hp.order2nside(R)) ** 2))


def calccorrs_from_config(
    config_path: ConfigPaths,
    *,
    output_dir: str | Path = "corrs",
    file_index: int = 0,
    num_batches_per_file: int = 100,
    dataset_split: str = "training",
    device: str | torch.device | None = None,
) -> list[Path]:
    """Merge YAML configurations from left to right and calculate correlations."""
    paths = config_paths(config_path)
    raw_config = load_config(paths)
    return calccorrs(
        raw_config, output_dir=output_dir, file_index=file_index, num_batches_per_file=num_batches_per_file,
        dataset_split=dataset_split, device=device,
    )


def _coerce_config(config_or_path: ConfigPaths | Mapping[str, Any] | Config) -> tuple[Config, dict[str, Any]]:
    """Normalize config input while retaining calccorrs-specific settings."""
    if isinstance(config_or_path, Config):
        raw_config = {
            "forward_model": config_or_path.forward_model,
            "training": config_or_path.training,
        }
        return config_or_path, raw_config
    if not isinstance(config_or_path, Mapping):
        paths = config_paths(config_or_path)
        raw_config = load_config(paths)
    else:
        raw_config = dict(config_or_path)
    return Config.from_mapping(raw_config), raw_config
