"""Exercise diagnostics without requiring CUDA transforms or a Nestcorr cache."""
from __future__ import annotations

import sys
import types

import h5py
import numpy as np
import torch

from euclid_multiprobe_deeplss_training import dataloaders
from euclid_multiprobe_deeplss_training.utils.config import Config


def _config():
    return Config.from_mapping({
        "forward_model": {"analysis": {"n_side": 2, "n_side_down": 1}},
        "training": {
            "physics_model": "fake", "physics_model_args": {"shape_noise_std": 0.1},
            "records_pattern": "unused-*.tar", "batch_size": 3, "num_workers": 0,
            "encoder_name": "corrs_transformer", "encoder_args": {}, "embed_dim": 2,
        },
    })


def _batches():
    for size in (3, 1):
        yield (
            [torch.full((size, 4), 1 + 2j), torch.full((size, 4), 3.0)],
            torch.ones(size, 2), torch.arange(size)[:, None],
        )


class _PhysicsModel(torch.nn.Module):
    num_channels = 2
    num_targets = 2
    channel_spins = [2, 0]
    params = ["omega_m", "sigma8"]

    def __init__(self, _config, **kwargs):
        super().__init__()
        assert kwargs["shape_noise_std"] == 0.1

    def get_weight_maps(self, maps):
        assert isinstance(maps, list)
        return [torch.ones_like(m.real) for m in maps]

    def preprocess_for_correlations(self, maps):
        assert isinstance(maps, list)
        maps[0] = maps[0] + 1j
        return maps


def test_calccls_preserves_spin_channels_and_counts_examples(monkeypatch, tmp_path):
    from euclid_multiprobe_deeplss_training import calccls

    calls = {"auto": [], "cross": []}

    class AutoSpectra(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()

        def forward(self, *maps):
            assert [m.dtype for m in maps] == [torch.complex64, torch.float32]
            assert torch.all(maps[0].imag == 2)
            size = maps[0].shape[0]
            calls["auto"].append(size)
            # Spin-2 and scalar auto spectra have different ranks.
            return torch.ones(size, 2, 4), torch.ones(size, 4)

    class CrossSpectra(AutoSpectra):
        def forward(self, *maps):
            assert maps[0].is_complex() and not maps[1].is_complex()
            calls["cross"].append(maps[0].shape[0])
            return torch.ones(maps[0].shape[0], 4, 3)

    transforms = types.ModuleType("euclid_multiprobe_deeplss_training.utils.cls_cuhpx")
    transforms.PartSkyAutoCls = AutoSpectra
    transforms.PartSkyCls = CrossSpectra
    monkeypatch.setitem(sys.modules, transforms.__name__, transforms)

    def pipeline(**kwargs):
        assert kwargs["device"] == torch.device("cpu")
        return list(_batches())

    monkeypatch.setattr(dataloaders, "OntheflyPipeline", pipeline)
    monkeypatch.setattr(calccls, "load_pixel_file", lambda _config: (np.arange(4), None, None, None))
    monkeypatch.setattr(calccls, "load_physics_model_class", lambda _name: _PhysicsModel)
    monkeypatch.setattr(calccls, "create_power_spectra_dashboard", lambda *args, **kwargs: None)
    monkeypatch.setattr(calccls, "create_cross_power_spectra_dashboard", lambda *args, **kwargs: None)

    path = calccls.calccls(_config(), output_path=tmp_path / "cls.h5", num_examples=3, device="cpu")
    # Two channels are not two examples: stop after the three-example batch.
    assert calls == {"auto": [3], "cross": [3]}
    with h5py.File(path, "r") as handle:
        assert handle["cls_0"].shape == (3, 2, 4)
        assert handle["cls_1"].shape == (3, 4)
        assert handle["labels"].shape == (3, 2)
    with h5py.File(tmp_path / "cls_cross.h5", "r") as handle:
        assert handle["cls_0"].shape == (3, 4, 3)


def test_calccorrs_stacks_channels_after_physics(monkeypatch, tmp_path):
    from euclid_multiprobe_deeplss_training import calccorrs

    class Correlator(torch.nn.Module):
        num_output_bins = 2
        theta = torch.tensor([0.1, 0.2])
        theta_range = torch.tensor([[0.05, 0.15], [0.15, 0.25]])

        def __init__(self, _order, spins, **kwargs):
            super().__init__()
            assert spins == [2, 0]

        def forward(self, maps, weights):
            assert maps.dtype == torch.complex64
            assert weights.dtype == torch.float32
            assert maps.shape[1:] == weights.shape[1:] == (2, 4)
            assert torch.all(maps[:, 0].imag == 3)
            assert torch.all(maps[:, 1].real == 3)
            assert torch.all(maps[:, 1].imag == 0)
            assert torch.all(weights == 1)
            return torch.ones(maps.shape[0], 2, 2, 2)

    nestcorr = types.ModuleType("nestcorr")
    pairs = types.ModuleType("nestcorr.pairs")
    pairs.resolve_pair_metadata = lambda *args, **kwargs: {"nd": 1, "binning": "standard", "data_vec_pix": None}
    correlators = types.ModuleType("nestcorr.nestcorr")
    correlators.NestcorrFastFootprint = Correlator
    for module in (nestcorr, pairs, correlators):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(dataloaders, "OntheflyPipeline", lambda **kwargs: list(_batches()))
    monkeypatch.setattr(calccorrs, "load_pixel_file", lambda _config: (np.arange(4), None, None, None))
    monkeypatch.setattr(calccorrs, "load_physics_model_class", lambda _name: _PhysicsModel)
    monkeypatch.setattr(calccorrs, "create_correlations_dashboard", lambda *args, **kwargs: None)
    pairs_path = tmp_path / "pairs.h5"
    pairs_path.touch()
    config = _config()
    config.training["encoder_args"]["pairs_filename"] = str(pairs_path)
    paths = calccorrs.calccorrs(config, output_dir=tmp_path / "corrs", num_batches_per_file=2, device="cpu")
    with h5py.File(paths[0], "r") as handle:
        assert handle["batch0000/correlations"].shape == (3, 2, 2, 2)
        assert handle["batch0001/correlations"].shape == (1, 2, 2, 2)
        assert handle["batch0001/labels"].shape == (1, 2)


def test_modelprofile_passes_channel_lists_to_configured_encoder(monkeypatch):
    from euclid_multiprobe_deeplss_training import modelprofile

    class Pipeline:
        num_pixels = 4

        def __init__(self, **kwargs):
            assert isinstance(kwargs["physics_model"], _PhysicsModel)
            assert kwargs["smoother"] is None

        def __iter__(self):
            return _batches()

    class Encoder(torch.nn.Module):
        def forward(self, maps):
            assert not torch.is_grad_enabled()
            assert not self.training
            assert isinstance(maps, list)
            return torch.stack((maps[0].imag.mean(-1), maps[1].mean(-1)), dim=-1)

    def build_encoder(name, **kwargs):
        assert name == "corrs_transformer"
        assert isinstance(kwargs["physics_model"], _PhysicsModel)
        assert kwargs["embed_dim"] == 2
        return Encoder()

    monkeypatch.setattr(modelprofile, "OntheflyPipeline", Pipeline)
    monkeypatch.setattr(modelprofile, "load_physics_model_class", lambda _name: _PhysicsModel)
    monkeypatch.setattr(modelprofile, "load_pixel_file", lambda _config: (np.arange(4), None, None, None))
    monkeypatch.setattr(modelprofile, "build_encoder", build_encoder)
    monkeypatch.setattr(modelprofile, "print_profiler_stats", lambda _prof: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    outputs = modelprofile.modelprofile(_config())
    assert [tuple(output.shape) for output in outputs] == [(3, 2), (1, 2)]
    torch.testing.assert_close(outputs[1], torch.tensor([[2.0, 3.0]]))
