from __future__ import annotations

import pytest


def test_datastats_prints_channel_statistics(monkeypatch, capsys, tmp_path) -> None:
    torch = pytest.importorskip("torch")
    monkeypatch.chdir(tmp_path)

    from euclid_multiprobe_deeplss_training import datastats

    class FakePhysicsModel:
        num_channels = 2

        def __init__(self, forward_model, *, scalers, device, nside):
            assert scalers is True
            assert nside == 1024
            self.forward_model = forward_model
            self.device = device

        def to(self, device):
            self.device = device
            return self

    class FakeSmoothingModel:
        def __init__(self, *, nside, nside_base, nside_lower, operator):
            assert nside == 1024
            assert nside_base == 128
            assert nside_lower == [512] * 2
            assert operator == "mean"

        def to(self, device):
            assert device == "cpu"
            return self

    class FakePipeline:
        def __init__(self, *, webds_pattern, physics_model, smoother, batch_size, num_workers, device):
            assert webds_pattern == "records/*.tar"
            assert physics_model.forward_model == {"survey": "euclid", "analysis": {"n_side": 1024, "n_side_down": 128}}
            assert isinstance(smoother, FakeSmoothingModel)
            assert batch_size == 1
            assert num_workers == 0
            assert device == "cpu"

            self.batch_size = batch_size

        def __iter__(self):
            yield (
                torch.tensor([[[1.0, 10.0], [2.0, 20.0]], [[3.0, 30.0], [4.0, 40.0]]]),
                torch.tensor([[0.0, 1.0], [2.0, 5.0]]),
                torch.tensor([0, 1]),
            )

    monkeypatch.setattr(datastats.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(datastats, "OntheflyPhysicsModelLinear", FakePhysicsModel)
    monkeypatch.setattr(datastats, "NestChannelScalecut", FakeSmoothingModel)
    monkeypatch.setattr(datastats, "OntheflyPipeline", FakePipeline)
    monkeypatch.setattr(datastats, "print_profiler_stats", lambda prof: None)

    batch_stats = datastats.datastats(
        {
            "records_pattern": "records/*.tar",
            "forward_model": {"survey": "euclid", "analysis": {"n_side": 1024, "n_side_down": 128}},
            "batch_size": 1,
            "num_workers": 0,
            "validation_fraction": 0.0,
            "use_wandb": False,
        }
    )

    captured = capsys.readouterr()

    assert len(batch_stats) == 1
    assert "channel=  0" in captured.out
    assert "min = 1.0000000000e+00" in captured.out
    assert "max = 4.0000000000e+00" in captured.out
    assert "mean= 2.5000000000e+00" in captured.out
    assert "channel=  1" in captured.out
    assert "min = 1.0000000000e+01" in captured.out
    assert "max = 4.0000000000e+01" in captured.out
    assert "mean= 2.5000000000e+01" in captured.out
    assert "Label statistics:" in captured.out
    assert "label=  0" in captured.out
    assert "mean= 1.0000000000e+00" in captured.out
    assert "label=  1" in captured.out
    assert "max = 5.0000000000e+00" in captured.out
    assert "mean= 3.0000000000e+00" in captured.out


def test_datastats_handles_ragged_complex_channels_across_batches(monkeypatch, tmp_path, capsys) -> None:
    import numpy as np
    import torch

    from euclid_multiprobe_deeplss_training import datastats

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(datastats, "print_profiler_stats", lambda _prof: None)
    batches = [
        ([torch.full((size, 2), 1 + 2j), torch.full((size, 4), 3.0)], torch.ones(size, 1), torch.arange(size))
        for size in (2, 1)
    ]
    map_batches, label_batches = datastats._collect_loader_stats(batches, split="train")
    summary = datastats._combine_batch_stats(map_batches)
    assert summary.feature_names == ["0_real", "0_imag", "1"]
    torch.testing.assert_close(summary.count, torch.tensor([6, 6, 12], dtype=torch.float64))
    torch.testing.assert_close(summary.total, torch.tensor([6, 12, 36], dtype=torch.float64))
    assert len(label_batches) == 2
    datastats._print_channel_stats(summary)
    output = capsys.readouterr().out
    assert "channel=0_real" in output
    assert "channel=0_imag" in output
    with np.load(tmp_path / "maps_batch_1.npz") as saved:
        assert saved["channel_0"].dtype == np.complex64
        np.testing.assert_array_equal(saved["channel_0"].imag, np.full((2, 2), 2.0))
        assert saved["channel_1"].shape == (2, 4)
