"""Plot one training example after configured physics, smoothing and scaling."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

# Initialize PyTorch before healpy's native libraries. The opposite order can
# crash sparse CSR construction on macOS with conflicting OpenMP runtimes.
import torch

# isort: split
import healpy as hp
import numpy as np

from .dataloaders import OntheflyPipeline
from .training import load_physics_model_class
from .utils.config import Config, ConfigPaths, config_paths, load_config, load_pixel_file
from .utils.logger import get_logger

LOGGER = get_logger(__file__)


def _view_rotation(nside: int, pixels: np.ndarray) -> tuple[float, float, float]:
    """Center on the footprint, snapping to a loaded pixel if its centroid is in a hole."""
    vectors = np.asarray(hp.pix2vec(nside, pixels, nest=True))
    center = vectors.mean(axis=1)
    if np.linalg.norm(center) < 1e-12:
        center = vectors[:, 0]
    center /= np.linalg.norm(center)
    if hp.vec2pix(nside, *center, nest=True) not in pixels:
        center = vectors[:, np.argmax(center @ vectors)]
    lon, lat = hp.vec2ang(center, lonlat=True)
    return float(lon[0]), float(lat[0]), 0.0


def _scalar_channels(
    channels: Sequence[torch.Tensor], config: Config,
) -> list[tuple[str, np.ndarray]]:
    """Keep both components of complex shear as separately labelled scalar rows."""
    wl_bins = config.forward_model["survey"]["WL"]["z_bins"]
    gc_bins = config.forward_model["survey"]["GC"]["z_bins"]
    labels = (
        [f"{z} lensing" for z in wl_bins]
        + [f"{z} source counts" for z in wl_bins]
        + [f"{z} lens counts" for z in gc_bins]
    )
    if len(labels) != len(channels):
        labels = [f"Channel {i + 1}" for i in range(len(channels))]
    rows = []
    for label, channel in zip(labels, channels, strict=True):
        if channel.ndim != 2 or channel.shape[0] != 1:
            raise ValueError("Expected each channel to have shape (1, pixels).")
        values = channel[0].detach().cpu().numpy()
        if np.iscomplexobj(values):
            rows.extend(((label + r" $\gamma_1$", values.real), (label + r" $\gamma_2$", values.imag)))
        else:
            rows.append((label, values))
    return rows


def _gnomonic_resolution(xsize: int, width_deg: float = 10.0) -> float:
    """Healpy plane spacing in arcmin for an exact angular width at the center."""
    return float(np.rad2deg(2 * np.tan(np.deg2rad(width_deg / 2))) * 60 / (xsize - 1))


def _save_channel_figure(
    rows: Sequence[tuple[str, np.ndarray]], *, pixels: np.ndarray, nside: int,
    output_file: Path, dpi: int, sample_indices: np.ndarray, config_names: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not rows:
        raise ValueError("The physics model returned no map channels.")
    rotation = _view_rotation(nside, pixels)
    direction = hp.ang2vec(rotation[0], rotation[1], lonlat=True)
    visible_fraction = np.mean(direction @ np.asarray(hp.pix2vec(nside, pixels, nest=True)) >= 0)
    LOGGER.info(f"View center: longitude={rotation[0]:.4f}, latitude={rotation[1]:.4f} deg")
    LOGGER.info(f"Half-sky views include {visible_fraction:.1%} of loaded pixels; the remainder are behind the horizon.")
    fig = plt.figure(figsize=(12, 3.8 * len(rows) + 1.2))
    fig.suptitle(
        f"Physics-processed channel maps | NSIDE {nside} | batch size 1\n"
        f"{config_names}\n"
        f"Sample indices: {sample_indices.tolist()} | center ({rotation[0]:.2f}, {rotation[1]:.2f}) deg\n"
        f"Half-sky panels show {visible_fraction:.1%} of loaded pixels; the remainder are behind the horizon.\n"
        "Configured smoothing and training scalers; shared 1-99% color limits per row. "
        "Shear components use the original map basis.",
        fontsize=11, y=1 - 0.12 / fig.get_figheight(),
    )
    # Only one full-sky map is allocated at a time. Pixels outside the loaded
    # NESTED data vector are masked, rather than presented as measured zeros.
    full_map = np.full(hp.nside2npix(nside), hp.UNSEEN, dtype=np.float32)
    xsize = 800
    top = 1 - 1.2 / fig.get_figheight()
    height = top / len(rows)
    try:
        for row, (label, values) in enumerate(rows):
            if values.shape != pixels.shape:
                raise ValueError(f"{label}: {values.size} map values do not match {pixels.size} footprint pixels.")
            finite = np.isfinite(values) & ~hp.mask_bad(values)
            if not finite.any():
                raise ValueError(f"{label} contains no finite map values.")
            lo, hi = np.percentile(values[finite], [1, 99])
            signed = lo < 0
            if signed:
                hi = max(abs(lo), abs(hi))
                lo = -hi
            if lo == hi:
                delta = max(abs(lo) * 0.01, 1e-6)
                lo, hi = lo - delta, hi + delta
            full_map.fill(hp.UNSEEN)
            full_map[pixels[finite]] = values[finite]
            common = dict(
                fig=fig.number, nest=True, rot=rotation, min=lo, max=hi,
                cmap="RdBu_r" if signed else "viridis", badcolor="#e6e6e6",
                bgcolor="white", xsize=xsize, notext=True, format="%.2g",
                # Leave space above every map for its title, and below for the colorbar.
                margins=(0.02, height * 0.07, 0.02, height * 0.08),
            )
            # Explicit axes rectangles keep the title separate from the first row.
            bottom = top - (row + 1) * height
            for column in range(2):
                ax = fig.add_axes((column * 0.5, bottom, 0.5, height))
                plt.sca(ax)
                if column == 0:
                    hp.orthview(full_map, half_sky=True, title=f"{row + 1:02d}  {label} | half sky", hold=True, **common)
                else:
                    hp.gnomview(
                        full_map, ysize=xsize, reso=_gnomonic_resolution(xsize),
                        title=f"{label} | 10 x 10 deg", hold=True, **common,
                    )
            LOGGER.info(f"Plotted row {row + 1}/{len(rows)}: {label}")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            output_file, format="pdf", dpi=dpi,
            metadata={"Title": "Physics-processed channel maps", "Subject": config_names},
        )
    finally:
        plt.close(fig)


def plot_channel_maps_from_config(
    config_path: ConfigPaths, *, output_file: str | Path = "channel_maps.pdf",
    device: str | torch.device | None = None, dpi: int = 300,
) -> Path:
    """Load exactly one batch, using training's model and a worker-free loader."""
    output_file = Path(output_file)
    if output_file.suffix.lower() != ".pdf":
        raise ValueError("output_file must have a .pdf extension.")
    if dpi <= 0:
        raise ValueError("dpi must be positive.")
    paths = config_paths(config_path)
    config = Config.from_mapping(load_config(paths))
    run_device = torch.device(device or config.training.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    seed = int(config.training.get("seed", config.forward_model.get("master_seed", 0)))
    torch.manual_seed(seed)
    pixels, _, _, _ = load_pixel_file(config.forward_model)
    nside = int(config.forward_model["analysis"]["n_side"])
    physics_model_class = load_physics_model_class(config.training["physics_model"])
    physics_model = physics_model_class(
        config.forward_model, scalers=True, device=run_device, seed=seed,
        nside=nside, indices_pixels=pixels, **config.training.get("physics_model_args", {}),
    ).to(run_device)
    physics_model.eval()
    loader = OntheflyPipeline(
        webds_pattern=config.training["records_pattern"], batch_size=1,
        physics_model=physics_model, device=run_device, num_workers=0, seed=seed,
        infer_shapes=False,
    )
    batch_iterator = iter(loader)
    try:
        channels, _targets, indices = next(batch_iterator)
    except StopIteration as exc:
        raise ValueError("No training examples were found for plot_channel_maps.") from exc
    finally:
        batch_iterator.close()
    rows = _scalar_channels(channels, config)
    del loader, physics_model, channels, _targets
    _save_channel_figure(
        rows, pixels=np.asarray(pixels), nside=nside, output_file=output_file, dpi=dpi,
        sample_indices=indices.detach().cpu().numpy(), config_names=" + ".join(path.name for path in paths),
    )
    LOGGER.info(f"Saved {len(rows)} channel rows to {output_file.resolve()}")
    return output_file
