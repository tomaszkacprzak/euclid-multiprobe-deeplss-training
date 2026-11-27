from __future__ import annotations

import sys
import types

import healpy as hp
import numpy as np
import pytest
import torch

from euclid_multiprobe_deeplss_training import plot_channel_maps as plotting
from euclid_multiprobe_deeplss_training.cli import main
from euclid_multiprobe_deeplss_training.utils.config import Config


@pytest.mark.parametrize("config_args", [
    ["--config", "survey.yaml", "training.yaml"],
    ["--config", "survey.yaml", "--config", "training.yaml"],
    ["--config", "survey.yaml,training.yaml"],
])
def test_cli_dispatches_composite_configs(monkeypatch, config_args):
    calls = []
    fake = types.ModuleType(plotting.__name__)
    fake.plot_channel_maps_from_config = lambda config, **kwargs: calls.append((config, kwargs))
    monkeypatch.setitem(sys.modules, plotting.__name__, fake)

    assert main([*config_args, "plot_channel_maps", "--output-file", "maps.pdf", "--device", "cpu", "--dpi", "400"]) == 0
    paths, kwargs = calls[0]
    assert [str(path) for path in plotting.config_paths(paths)] == ["survey.yaml", "training.yaml"]
    assert kwargs == {"output_file": "maps.pdf", "device": "cpu", "dpi": 400}


def test_cli_requires_config():
    with pytest.raises(ValueError, match="plot_channel_maps command requires --config"):
        main(["plot_channel_maps"])


def test_plotting_loads_one_processed_batch_and_keeps_both_shear_components(tmp_path, monkeypatch):
    survey = tmp_path / "survey.yaml"
    survey.write_text("forward_model:\n  analysis: {n_side: 1}\n  survey:\n    WL: {z_bins: [WL1]}\n    GC: {z_bins: [GC1]}\n")
    training = tmp_path / "training.yaml"
    training.write_text(
        "training:\n  records_pattern: records/*.tar\n  physics_model: onthefly_linear_shear\n"
        "  device: cuda\n  seed: 17\n  batch_size: 12\n  num_workers: 16\n"
        "  physics_model_args:\n    small_scale_selection: {WL: 4, GC: 16}\n    large_scale_selection: 6\n"
    )
    calls = {}

    class Physics:
        def __init__(self, config, **kwargs):
            calls["physics"] = kwargs

        def to(self, device):
            return self

        def eval(self):
            return self

    class Loader:
        def __init__(self, **kwargs):
            calls["loader"] = kwargs

        def __iter__(self):
            try:
                calls["batches"] = calls.get("batches", 0) + 1
                yield [torch.full((1, 12), 2 + 3j), torch.full((1, 12), 4.), torch.full((1, 12), 5.)], torch.zeros(1, 2), torch.tensor([[7, 8, 9]])
                pytest.fail("Plotting requested more than one batch")
            finally:
                calls["closed"] = True

    monkeypatch.setattr(plotting, "load_pixel_file", lambda config: (np.arange(12), {}, {}, None))
    monkeypatch.setattr(plotting, "load_physics_model_class", lambda name: Physics)
    monkeypatch.setattr(plotting, "OntheflyPipeline", Loader)
    monkeypatch.setattr(plotting, "_save_channel_figure", lambda rows, **kwargs: calls.update(rows=rows, figure=kwargs))

    output = plotting.plot_channel_maps_from_config([survey, training], output_file=tmp_path / "maps.pdf", device="cpu")

    assert output == tmp_path / "maps.pdf"
    assert calls["physics"]["scalers"] is True
    assert calls["physics"]["small_scale_selection"] == {"WL": 4, "GC": 16}
    assert calls["physics"]["large_scale_selection"] == 6
    assert calls["physics"]["seed"] == 17
    assert calls["loader"]["batch_size"] == 1
    assert calls["loader"]["num_workers"] == 0
    assert calls["loader"]["infer_shapes"] is False
    assert calls["loader"]["device"] == torch.device("cpu")
    assert calls["batches"] == 1 and calls["closed"]
    assert len(calls["rows"]) == 4
    for (_, values), expected in zip(calls["rows"], [2, 3, 4, 5], strict=True):
        np.testing.assert_array_equal(values, np.full(12, expected))


def test_projection_center_stays_on_footprint_and_zoom_spans_ten_degrees():
    nside = 16
    pixels = hp.query_disc(nside, hp.ang2vec(359, -35, lonlat=True), np.deg2rad(12), nest=True)
    lon, lat, _ = plotting._view_rotation(nside, pixels)
    assert hp.ang2pix(nside, lon, lat, lonlat=True, nest=True) in pixels
    assert abs(lat + 35) < 1
    projection = hp.projector.GnomonicProj(xsize=800, ysize=800, reso=plotting._gnomonic_resolution(800))
    left = projection.xy2vec(projection.ij2xy(399.5, 0), direct=True)
    right = projection.xy2vec(projection.ij2xy(399.5, 799), direct=True)
    assert np.rad2deg(np.arccos(np.dot(left, right))) == pytest.approx(10)


def test_many_rows_render_to_pdf_without_negative_axes(tmp_path, monkeypatch):
    # Exercise the real healpy axes/layout with small projection images.
    for name in ("orthview", "gnomview"):
        original = getattr(hp, name)

        def small_view(*args, _original=original, **kwargs):
            kwargs["xsize"] = 32
            if "ysize" in kwargs:
                kwargs["ysize"] = 32
                kwargs["reso"] = plotting._gnomonic_resolution(32)
            return _original(*args, **kwargs)

        monkeypatch.setattr(hp, name, small_view)
    output = tmp_path / "channels.pdf"
    plotting._save_channel_figure(
        [(f"Channel {index}", np.arange(12, dtype=float)) for index in range(24)],
        pixels=np.arange(12), nside=1, output_file=output, dpi=30,
        sample_indices=np.array([[1, 2, 3]]), config_names="survey.yaml + training.yaml",
    )
    assert output.read_bytes().startswith(b"%PDF")


def test_complex_channels_produce_24_rows():
    config = Config.from_mapping({"forward_model": {"survey": {
        "WL": {"z_bins": [f"WL{i}" for i in range(1, 7)]},
        "GC": {"z_bins": [f"GC{i}" for i in range(1, 7)]},
    }}})
    channels = [torch.zeros(1, 12, dtype=torch.complex64 if i < 6 else torch.float32) for i in range(18)]
    rows = plotting._scalar_channels(channels, config)
    assert len(rows) == 24
    assert "gamma_1" in rows[0][0] and "gamma_2" in rows[1][0]
    assert rows[-1][0] == "GC6 lens counts"
