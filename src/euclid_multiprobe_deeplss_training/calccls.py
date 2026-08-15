"""Calculate auto power spectra for generated DeepLSS training maps."""

from __future__ import annotations

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
from .utils.cls_cuhpx import PartSkyAutoCls
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
    """Calculate auto spectra for every batch in one training-set epoch.

    The output contains one extensible dataset per map probe, named
    ``cls_0``, ``cls_1``, and so on, plus ``labels`` and ``inds`` datasets.
    Each batch is appended immediately so neither CPU nor GPU memory use grows
    over the epoch.
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
    cls_calculator = PartSkyAutoCls(
        indices=torch.as_tensor(indices, dtype=torch.long, device=run_device),
        nside=nside,
        lmax=lmax,
    ).to(run_device)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Calculating auto power spectra for one training epoch")
    # Opening in write mode creates an empty file before loader iteration and
    # deliberately replaces an earlier result at the requested path.
    

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

    dashboard_path = output_path.with_suffix(".html")
    create_power_spectra_dashboard(
        output_path,
        dashboard_path,
        parameter_names=[str(name) for name in physics_model.params],
        model_information={
            "physics_model": config.physics_model,
            "shape_noise_std": _find_config_value(config.forward_model, "shape_noise_std"),
            "config_forward_model": config.config_forward_model,
        },
    )
    LOGGER.info("Wrote interactive power-spectra dashboard to %s", dashboard_path)
    return output_path


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
        raise ValueError(
            f"Physics model supplies {len(parameter_names)} parameter names for {labels.shape[1]} label columns."
        )
    if not probe_names:
        raise ValueError("No cls_<probe index> datasets were found.")

    columns = 6
    rows = math.ceil(len(probe_names) / columns)
    figure = make_subplots(rows=rows, cols=columns, subplot_titles=[f"Probe {i}" for i in range(len(probe_names))])
    label_colors: list[list[str]] = []
    for parameter_index in range(labels.shape[1]):
        values = labels[:, parameter_index]
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
                            f"example={example_index}<br>component={component_index}<br>"
                            "ell=%{x}<br>scaled Cℓ=%{y:.6g}<extra></extra>"
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
    initial_values = labels[:, 0]
    figure.add_trace(
        go.Scatter(
            x=[None] * len(initial_values),
            y=[None] * len(initial_values),
            mode="markers",
            marker={
                "color": initial_values,
                "colorscale": COOLWARM_COLORSCALE,
                "showscale": True,
                "colorbar": {"title": parameter_names[0]},
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
    for parameter_index, parameter_name in enumerate(parameter_names):
        values = labels[:, parameter_index]
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
    plot_html = pio.to_html(figure, full_html=False, include_plotlyjs=True, config={"responsive": True})
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
    full_convolution = np.convolve(values, kernel, mode="full")
    start = (kernel.size - 1) // 2
    return full_convolution[start : start + values.size]


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
