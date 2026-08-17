"""Calculate auto and cross power spectra for generated training maps."""

from __future__ import annotations

import gc
import html
import json
import math
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from .training import TrainingConfig, load_physics_model_class
from .utils.cls_cuhpx import PartSkyAutoCls, PartSkyCls
from .utils.config import load_config, load_pixel_indices, with_forward_model_config
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

# inds: i_signal, i_sobol, i_cosmo, i_perm, i_patch, nside, nside_down


def calccls(
    config_or_path: str | Path | Mapping[str, Any] | TrainingConfig,
    *,
    output_path: str | Path = "cls.h5",
    num_examples: int = 100,
    device: torch.device | str | None = None,
) -> Path:
    """Calculate auto and cross spectra for batches in a training-set epoch.

    ``output_path`` contains one extensible auto-spectrum dataset per map
    probe, named
    ``cls_0``, ``cls_1``, and so on, plus ``labels`` and ``inds`` datasets.
    A second file, whose name has ``_cross`` appended to the output stem,
    contains all auto/cross pairs in ``cls_0`` in the ordering documented by
    :class:`PartSkyCls`. Each batch is appended immediately so neither CPU nor
    GPU memory use grows over either pass through the data.
    """
    from msfm.onthefly_pipeline import OntheflyPipeline

    config, raw_config = _coerce_config(config_or_path)
    requested_device = device or raw_config.get("device")
    run_device = torch.device(requested_device or ("cuda" if torch.cuda.is_available() else "cpu"))

    indices = load_pixel_indices(config.forward_model)
    analysis = config.forward_model["analysis"]
    nside = int(analysis["n_side"])
    cls_config = raw_config.get("calccls", {}) or {}
    if not isinstance(cls_config, Mapping):
        raise TypeError("The optional 'calccls' configuration section must be a mapping.")
    lmax = int(cls_config.get("lmax", raw_config.get("lmax", analysis.get("l_max", 3 * nside))))

    physics_model_class = load_physics_model_class(config.physics_model)
    physics_model = physics_model_class(
        config.forward_model,
        scalers=True,
        device=run_device,
        seed=int(time.time()),
        nside=nside,
    ).to(run_device)
    loader = OntheflyPipeline(
        webds_pattern=config.records_pattern,
        batch_size=config.batch_size,
        physics_model=physics_model,
        downsampler=None,
        smoother=None,
        num_workers=config.num_workers,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cross_output_path = _cross_output_path(output_path)
    LOGGER.info(f"Calculating auto power spectra for {num_examples} examples")
    # Opening in write mode creates an empty file before loader iteration and
    # deliberately replaces an earlier result at the requested path.

    cls_calculator = PartSkyAutoCls(
        indices=torch.as_tensor(indices, dtype=torch.long, device=run_device),
        nside=nside,
        lmax=lmax,
    ).to(run_device)

    j = 0
    i = 0
    with h5py.File(output_path, "w") as output_file, torch.no_grad():
        for maps, labels, inds in loader:
            j += 1
            i += len(maps)
            maps = maps.to(device=run_device, dtype=torch.float32)
            maps_list = physics_model.unstack_batch_channels(maps)
            batch_spectra_list = cls_calculator.forward(*maps_list)
            _append_batch(output_file, batch_spectra_list, labels, inds)
            output_file.flush()
            batch_spectra = torch.stack(batch_spectra_list, dim=-1)
            LOGGER.debug(f"Batch {j: 5d}: input maps shape: {maps.shape}, output spectra shape: {batch_spectra.shape}")

            if i >= num_examples:
                LOGGER.info(f"Calculated {i} examples, stopping at requested {num_examples} examples.")
                break

    parameter_names = [str(name) for name in physics_model.params]
    model_information = {
        "physics_model": config.physics_model,
        "shape_noise_std": _find_config_value(config.forward_model, "shape_noise_std"),
        "config_forward_model": config.config_forward_model,
    }
    dashboard_path = output_path.with_suffix(".html")
    create_power_spectra_dashboard(
        output_path,
        dashboard_path,
        parameter_names=parameter_names,
        model_information=model_information,
    )
    LOGGER.info("Wrote interactive auto-power-spectra dashboard to %s", dashboard_path)

    del cls_calculator
    gc.collect()

    LOGGER.info(f"Calculating cross power spectra for {num_examples} examples")

    cross_cls_calculator = PartSkyCls(
        indices=torch.as_tensor(indices, dtype=torch.long, device=run_device),
        nside=nside,
        lmax=lmax,
    ).to(run_device)

    j = 0
    i = 0
    with h5py.File(cross_output_path, "w") as output_file, torch.no_grad():
        for maps, labels, inds in loader:
            j += 1
            i += len(maps)
            maps = maps.to(device=run_device, dtype=torch.float32)
            maps_list = physics_model.unstack_batch_channels(maps)
            batch_spectra = cross_cls_calculator.forward(*maps_list)
            _append_batch(output_file, (batch_spectra,), labels, inds)
            output_file.flush()
            LOGGER.debug(f"Batch {j: 5d}: input maps shape: {maps.shape}, output cross spectra shape: {batch_spectra.shape}")

            if i >= num_examples:
                LOGGER.info(f"Calculated {i} examples, stopping at requested {num_examples} examples.")
                break

    del cross_cls_calculator
    gc.collect()

    cross_dashboard_path = cross_output_path.with_suffix(".html")
    create_cross_power_spectra_dashboard(
        cross_output_path,
        cross_dashboard_path,
        parameter_names=parameter_names,
        model_information=model_information,
    )
    LOGGER.info("Wrote cross power spectra to %s", cross_output_path)
    LOGGER.info("Wrote interactive cross-power-spectra dashboard to %s", cross_dashboard_path)
    return output_path


def _cross_output_path(output_path: Path) -> Path:
    """Return the cross-spectrum HDF5 path associated with an auto path."""
    return output_path.with_name(f"{output_path.stem}_cross{output_path.suffix}")


def calccls_from_config(config_path: str | Path, *, output_path: str | Path = "cls.h5", num_examples: int = 100) -> Path:
    """Load a YAML configuration file and calculate its training spectra."""
    path = Path(config_path)
    raw_config = with_forward_model_config(load_config(path), path.parent)
    return calccls(raw_config, output_path=output_path, num_examples=num_examples)


def _append_batch(
    output_file: h5py.File,
    spectra: tuple[torch.Tensor, ...],
    labels: torch.Tensor,
    inds: torch.Tensor,
) -> None:
    """Append spectra and their corresponding metadata to an HDF5 file."""
    batch_size = labels.shape[0]
    if inds.shape[0] != batch_size or any(spectrum.shape[0] != batch_size for spectrum in spectra):
        raise ValueError("Spectra, labels, and inds must have the same batch size.")

    _append_spectra(output_file, spectra)
    _append_tensor(output_file, "labels", labels)
    _append_tensor(output_file, "inds", inds)


def _append_spectra(output_file: h5py.File, spectra: tuple[torch.Tensor, ...]) -> None:
    """Append one spectra batch to an open HDF5 output file."""
    existing_spectra = [name for name in output_file if name.startswith("cls_")]
    if existing_spectra and len(existing_spectra) != len(spectra):
        raise ValueError(f"Expected {len(existing_spectra)} spectra tensors, got {len(spectra)}.")

    for probe_index, spectrum in enumerate(spectra):
        _append_tensor(output_file, f"cls_{probe_index}", spectrum)


def _append_tensor(output_file: h5py.File, dataset_name: str, tensor: torch.Tensor) -> None:
    """Append a tensor along its batch dimension to an extensible dataset."""
    values = tensor.detach().cpu().numpy()
    if dataset_name not in output_file:
        output_file.create_dataset(
            dataset_name,
            data=values,
            maxshape=(None, *values.shape[1:]),
            chunks=True,
        )
        return

    dataset = output_file[dataset_name]
    old_size = dataset.shape[0]
    dataset.resize(old_size + values.shape[0], axis=0)
    dataset[old_size:] = values


def create_power_spectra_dashboard(
    spectra_path: str | Path,
    dashboard_path: str | Path,
    *,
    parameter_names: list[str],
    model_information: Mapping[str, Any],
) -> Path:
    """Create a self-contained Plotly dashboard from a calccls HDF5 file."""
    import plotly.graph_objects as go
    import plotly.io as pio
    from plotly.colors import sample_colorscale
    from plotly.subplots import make_subplots

    with h5py.File(spectra_path, "r") as source:
        labels = np.asarray(source["labels"])
        probe_names = sorted(
            (name for name in source if name.startswith("cls_")),
            key=lambda name: int(name.removeprefix("cls_")),
        )
        spectra = [np.asarray(source[name]) for name in probe_names]

    if labels.ndim != 2:
        raise ValueError(f"labels must be a 2D array, got shape {labels.shape}.")
    if labels.shape[1] != len(parameter_names):
        raise ValueError(f"Physics model supplies {len(parameter_names)} parameter names for {labels.shape[1]} label columns.")
    if labels.shape[1] < 2:
        raise ValueError("At least two label columns (Om and s8) are required to calculate S8.")
    if not probe_names:
        raise ValueError("No cls_<probe index> datasets were found.")

    s8 = labels[:, 1] * np.sqrt(labels[:, 0] / 0.3) ** 0.5
    color_values = np.column_stack((labels, s8))
    color_parameter_names = [*parameter_names, "S8"]

    columns = 6
    rows = math.ceil(len(probe_names) / columns)
    figure = make_subplots(rows=rows, cols=columns, subplot_titles=[f"Probe {i}" for i in range(len(probe_names))])
    label_colors: list[list[str]] = []
    for parameter_index in range(color_values.shape[1]):
        values = color_values[:, parameter_index]
        low, high = float(np.nanmin(values)), float(np.nanmax(values))
        normalized = np.zeros_like(values, dtype=float) if high == low else (values - low) / (high - low)
        label_colors.append(list(sample_colorscale(COOLWARM_COLORSCALE, normalized, colortype="rgb")))

    line_trace_count = 0
    for probe_index, values in enumerate(spectra):
        if values.shape[0] != labels.shape[0]:
            raise ValueError(f"{probe_names[probe_index]} and labels have different numbers of examples.")
        row, column = divmod(probe_index, columns)
        panel_maximum: float | None = None
        for example_index, spectrum in enumerate(values):
            curves = np.asarray(spectrum).reshape(-1, spectrum.shape[-1])
            ell = np.arange(curves.shape[-1])
            scale = ell * (ell + 1) / (2 * np.pi)
            for component_index, curve in enumerate(curves):
                scaled_curve = curve * scale
                smoothed_curve = _smooth_spectrum(scaled_curve)
                maximum_slice = smoothed_curve[ell >= 100]
                if maximum_slice.size == 0:
                    maximum_slice = smoothed_curve
                finite_values = maximum_slice[np.isfinite(maximum_slice)]
                if finite_values.size:
                    curve_maximum = float(np.max(finite_values))
                    panel_maximum = curve_maximum if panel_maximum is None else max(panel_maximum, curve_maximum)
                figure.add_trace(
                    go.Scattergl(
                        x=ell,
                        y=smoothed_curve,
                        mode="lines",
                        line={"color": label_colors[0][example_index], "width": 1},
                        name=f"Example {example_index}",
                        legendgroup=f"example-{example_index}",
                        showlegend=False,
                        hovertemplate=(
                            f"example={example_index}<br>component={component_index}<br>ell=%{{x}}<br>scaled Cℓ=%{{y:.6g}}<extra></extra>"
                        ),
                    ),
                    row=row + 1,
                    col=column + 1,
                )
                line_trace_count += 1
        x_axis_title = "ell" if probe_index + columns >= len(probe_names) else None
        y_axis_title = "Cℓ × ell(ell+1)/(2π)" if column == 0 else None
        figure.update_xaxes(title_text=x_axis_title, row=row + 1, col=column + 1)
        y_axis_range = [0, panel_maximum] if panel_maximum is not None and panel_maximum > 0 else None
        figure.update_yaxes(title_text=y_axis_title, range=y_axis_range, row=row + 1, col=column + 1)

    # An invisible marker trace supplies the shared continuous color bar.
    initial_values = color_values[:, 0]
    figure.add_trace(
        go.Scatter(
            x=[None] * len(initial_values),
            y=[None] * len(initial_values),
            mode="markers",
            marker={
                "color": initial_values,
                "colorscale": COOLWARM_COLORSCALE,
                "showscale": True,
                "colorbar": {"title": color_parameter_names[0]},
            },
            showlegend=False,
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )

    buttons = []
    trace_indices = list(range(line_trace_count + 1))
    examples_per_trace = []
    for values in spectra:
        for example_index, spectrum in enumerate(values):
            examples_per_trace.extend([example_index] * np.asarray(spectrum).reshape(-1, spectrum.shape[-1]).shape[0])
    for parameter_index, parameter_name in enumerate(color_parameter_names):
        values = color_values[:, parameter_index]
        line_colors = [label_colors[parameter_index][example] for example in examples_per_trace]
        low, high = float(np.nanmin(values)), float(np.nanmax(values))
        buttons.append(
            {
                "label": parameter_name,
                "method": "restyle",
                "args": [
                    {
                        "line.color": [*line_colors, "rgba(0,0,0,0)"],
                        "marker.color": [[float(values[example])] for example in examples_per_trace] + [values],
                        "marker.cmin": [*[low] * line_trace_count, low],
                        "marker.cmax": [*[high] * line_trace_count, high],
                        "marker.colorbar.title.text": [*[parameter_name] * line_trace_count, parameter_name],
                    },
                    trace_indices,
                ],
            }
        )
    figure.update_layout(
        title={"text": "Angular Power Spectra Across Cosmologies", "x": 0.5},
        height=max(430, 360 * rows),
        template="plotly_white",
        hovermode="closest",
        updatemenus=[{"buttons": buttons, "direction": "down", "x": 0, "xanchor": "left", "y": 1.12}],
        annotations=[
            *figure.layout.annotations,
            {"text": "Color parameter:", "showarrow": False, "x": 0, "xanchor": "left", "xref": "paper", "y": 1.17, "yref": "paper"},
        ],
    )

    information = html.escape(json.dumps(dict(model_information), indent=2, default=str))
    hover_script = f"""
const dashboard = document.getElementById('{{plot_id}}');
let highlightedTrace = null;
dashboard.on('plotly_hover', function(event) {{
    const traceIndex = event.points[0].curveNumber;
    if (traceIndex >= {line_trace_count}) return;
    if (highlightedTrace !== null && highlightedTrace !== traceIndex) {{
        Plotly.restyle(dashboard, {{'line.width': 1}}, [highlightedTrace]);
    }}
    Plotly.restyle(dashboard, {{'line.width': 4}}, [traceIndex]);
    highlightedTrace = traceIndex;
}});
dashboard.on('plotly_unhover', function() {{
    if (highlightedTrace !== null) {{
        Plotly.restyle(dashboard, {{'line.width': 1}}, [highlightedTrace]);
        highlightedTrace = null;
    }}
}});
"""
    plot_html = pio.to_html(
        figure,
        full_html=False,
        include_plotlyjs=True,
        config={"responsive": True},
        post_script=hover_script,
    )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Angular Power Spectra Dashboard</title></head><body>
<h1>Angular Power Spectra Dashboard</h1>
<details><summary>Model information</summary><pre>{information}</pre></details>
{plot_html}
</body></html>"""
    destination = Path(dashboard_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(document, encoding="utf-8")
    return destination


def create_cross_power_spectra_dashboard(
    spectra_path: str | Path,
    dashboard_path: str | Path,
    *,
    parameter_names: list[str],
    model_information: Mapping[str, Any],
) -> Path:
    """Create a single-panel dashboard for selectable cross spectra."""
    import plotly.graph_objects as go
    import plotly.io as pio
    from plotly.colors import sample_colorscale

    with h5py.File(spectra_path, "r") as source:
        labels = np.asarray(source["labels"])
        spectra = np.asarray(source["cls_0"])

    if labels.ndim != 2:
        raise ValueError(f"labels must be a 2D array, got shape {labels.shape}.")
    if labels.shape[1] != len(parameter_names):
        raise ValueError(f"Physics model supplies {len(parameter_names)} parameter names for {labels.shape[1]} label columns.")
    if labels.shape[1] < 2:
        raise ValueError("At least two label columns (Om and s8) are required to calculate S8.")
    if spectra.ndim != 3 or spectra.shape[0] != labels.shape[0]:
        raise ValueError("cls_0 must have shape (number of examples, number of multipoles, number of probe pairs).")

    pair_count = spectra.shape[2]
    probe_count = (math.isqrt(8 * pair_count + 1) - 1) // 2
    if probe_count * (probe_count + 1) // 2 != pair_count:
        raise ValueError(f"The cls_0 pair dimension ({pair_count}) is not triangular.")
    probe_pairs = [(map1, map2) for map1 in range(probe_count) for map2 in range(map1, probe_count)]

    s8 = labels[:, 1] * np.sqrt(labels[:, 0] / 0.3) ** 0.5
    color_values = np.column_stack((labels, s8))
    color_parameter_names = [*parameter_names, "S8"]
    label_colors: list[list[str]] = []
    for parameter_index in range(color_values.shape[1]):
        values = color_values[:, parameter_index]
        low, high = float(np.nanmin(values)), float(np.nanmax(values))
        normalized = np.zeros_like(values, dtype=float) if high == low else (values - low) / (high - low)
        label_colors.append(list(sample_colorscale(COOLWARM_COLORSCALE, normalized, colortype="rgb")))

    figure = go.Figure()
    ell = np.arange(spectra.shape[1])
    scale = ell * (ell + 1) / (2 * np.pi)
    plotted_spectra = np.empty_like(spectra, dtype=float)
    for pair_index in range(pair_count):
        for example_index in range(spectra.shape[0]):
            plotted_spectra[example_index, :, pair_index] = _smooth_spectrum(spectra[example_index, :, pair_index] * scale)

    finite_values = plotted_spectra[np.isfinite(plotted_spectra)]
    high_ell_values = plotted_spectra[:, ell > 100, :]
    finite_high_ell_values = high_ell_values[np.isfinite(high_ell_values)]
    y_axis_range = None
    if finite_values.size and finite_high_ell_values.size:
        y_axis_range = [float(np.min(finite_values)), float(np.max(finite_high_ell_values))]

    for pair_index, (map1, map2) in enumerate(probe_pairs):
        for example_index in range(spectra.shape[0]):
            figure.add_trace(
                go.Scattergl(
                    x=ell,
                    y=plotted_spectra[example_index, :, pair_index],
                    mode="lines",
                    line={"color": label_colors[0][example_index], "width": 1},
                    name=f"Example {example_index}",
                    visible=pair_index == 0,
                    meta={"map1": map1, "map2": map2},
                    showlegend=False,
                    hovertemplate=(f"example={example_index}<br>map1={map1}<br>map2={map2}<br>ell=%{{x}}<br>scaled Cℓ=%{{y:.6g}}<extra></extra>"),
                )
            )

    line_trace_count = len(figure.data)
    initial_values = color_values[:, 0]
    figure.add_trace(
        go.Scatter(
            x=[None] * len(initial_values),
            y=[None] * len(initial_values),
            mode="markers",
            marker={
                "color": initial_values,
                "colorscale": COOLWARM_COLORSCALE,
                "showscale": True,
                "colorbar": {"title": color_parameter_names[0]},
            },
            showlegend=False,
            hoverinfo="skip",
        )
    )

    buttons = []
    trace_indices = list(range(line_trace_count + 1))
    examples_per_trace = list(range(spectra.shape[0])) * pair_count
    for parameter_index, parameter_name in enumerate(color_parameter_names):
        values = color_values[:, parameter_index]
        line_colors = [label_colors[parameter_index][example] for example in examples_per_trace]
        low, high = float(np.nanmin(values)), float(np.nanmax(values))
        buttons.append(
            {
                "label": parameter_name,
                "method": "restyle",
                "args": [
                    {
                        "line.color": [*line_colors, "rgba(0,0,0,0)"],
                        "marker.color": [[float(values[example])] for example in examples_per_trace] + [values],
                        "marker.cmin": [*[low] * line_trace_count, low],
                        "marker.cmax": [*[high] * line_trace_count, high],
                        "marker.colorbar.title.text": [*[parameter_name] * line_trace_count, parameter_name],
                    },
                    trace_indices,
                ],
            }
        )

    figure.update_layout(
        title={"text": "Cross Angular Power Spectra Across Cosmologies", "x": 0.5},
        template="plotly_white",
        hovermode="closest",
        xaxis_title="ell",
        yaxis_title="Cℓ × ell(ell+1)/(2π)",
        yaxis_range=y_axis_range,
        updatemenus=[{"buttons": buttons, "direction": "down", "x": 0, "xanchor": "left", "y": 1.12}],
        annotations=[
            {"text": "Color parameter:", "showarrow": False, "x": 0, "xanchor": "left", "xref": "paper", "y": 1.17, "yref": "paper"},
        ],
    )

    information = html.escape(json.dumps(dict(model_information), indent=2, default=str))
    controls = f"""<div class="probe-controls">
<label for="map1-probe">Map 1 probe: <output id="map1-value">0</output></label>
<input id="map1-probe" type="range" min="0" max="{probe_count - 1}" value="0" step="1">
<label for="map2-probe">Map 2 probe: <output id="map2-value">0</output></label>
<input id="map2-probe" type="range" min="0" max="{probe_count - 1}" value="0" step="1">
</div>"""
    interaction_script = f"""
const dashboard = document.getElementById('{{plot_id}}');
const map1Slider = document.getElementById('map1-probe');
const map2Slider = document.getElementById('map2-probe');
function selectProbePair() {{
    const selected1 = Number(map1Slider.value);
    const selected2 = Number(map2Slider.value);
    document.getElementById('map1-value').value = selected1;
    document.getElementById('map2-value').value = selected2;
    const low = Math.min(selected1, selected2);
    const high = Math.max(selected1, selected2);
    const visibility = dashboard.data.slice(0, {line_trace_count}).map(
        trace => trace.meta.map1 === low && trace.meta.map2 === high
    );
    Plotly.restyle(dashboard, {{visible: visibility}}, [...Array({line_trace_count}).keys()]);
}}
map1Slider.addEventListener('input', selectProbePair);
map2Slider.addEventListener('input', selectProbePair);
let highlightedTrace = null;
dashboard.on('plotly_hover', function(event) {{
    const traceIndex = event.points[0].curveNumber;
    if (traceIndex >= {line_trace_count}) return;
    if (highlightedTrace !== null && highlightedTrace !== traceIndex) {{
        Plotly.restyle(dashboard, {{'line.width': 1}}, [highlightedTrace]);
    }}
    Plotly.restyle(dashboard, {{'line.width': 4}}, [traceIndex]);
    highlightedTrace = traceIndex;
}});
dashboard.on('plotly_unhover', function() {{
    if (highlightedTrace !== null) {{
        Plotly.restyle(dashboard, {{'line.width': 1}}, [highlightedTrace]);
        highlightedTrace = null;
    }}
}});
"""
    plot_html = pio.to_html(
        figure,
        full_html=False,
        include_plotlyjs=True,
        config={"responsive": True},
        post_script=interaction_script,
    )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Cross Angular Power Spectra Dashboard</title></head><body>
<h1>Cross Angular Power Spectra Dashboard</h1>
<details><summary>Model information</summary><pre>{information}</pre></details>
{controls}
{plot_html}
</body></html>"""
    destination = Path(dashboard_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(document, encoding="utf-8")
    return destination


def _find_config_value(config: Mapping[str, Any], key: str) -> Any:
    """Find the first nested config value named ``key``."""
    if key in config:
        return config[key]
    for value in config.values():
        if isinstance(value, Mapping):
            found = _find_config_value(value, key)
            if found is not None:
                return found
    return None


def _smooth_spectrum(values: np.ndarray) -> np.ndarray:
    """Smooth a spectrum with a length-ten uniform kernel without changing its length."""
    kernel = np.ones(10) / 10
    values_convolved = np.convolve(values, kernel, mode="full")
    start = (kernel.size - 1) // 2
    return values_convolved[start : start + values.size]


def _coerce_config(
    config_or_path: str | Path | Mapping[str, Any] | TrainingConfig,
) -> tuple[TrainingConfig, dict[str, Any]]:
    """Normalize config input while retaining calccls-specific settings."""
    if isinstance(config_or_path, TrainingConfig):
        raw_config = {**config_or_path.extra}
        for field_name in config_or_path.__dataclass_fields__:
            if field_name != "extra":
                raw_config[field_name] = getattr(config_or_path, field_name)
        return config_or_path, raw_config
    if isinstance(config_or_path, str | Path):
        path = Path(config_or_path)
        raw_config = with_forward_model_config(load_config(path), path.parent)
    else:
        raw_config = dict(config_or_path)
    return TrainingConfig.from_mapping(raw_config), raw_config
