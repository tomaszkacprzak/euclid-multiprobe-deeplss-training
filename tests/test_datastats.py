from __future__ import annotations

import pytest


def test_datastats_prints_channel_statistics(monkeypatch, capsys) -> None:
    torch = pytest.importorskip("torch")

    from euclid_multiprobe_deeplss_training import datastats

    class FakePhysicsModel:
        def __init__(self, forward_model, *, device):
            self.forward_model = forward_model
            self.device = device

        def to(self, device):
            self.device = device
            return self

    class FakeSmoothingModel:
        def __init__(self, *, nside, nside_base, nside_lower, operator):
            assert nside == 1024
            assert nside_base == 128
            assert nside_lower == [512] * 24
            assert operator == "mean"

        def to(self, device):
            assert device == "cpu"
            return self

    class FakePipeline:
        def __init__(self, records_pattern, physics_model, smoothing_model, *, batch_size, num_workers, pin_memory, device):
            assert records_pattern == "records/*.tar"
            assert physics_model.forward_model == {"survey": "euclid", "analysis": {"n_side": 1024, "n_side_down": 128}}
            assert isinstance(smoothing_model, FakeSmoothingModel)
            assert batch_size == 1
            assert num_workers == 0
            assert pin_memory is True
            assert device == "cpu"

        def __iter__(self):
            yield torch.tensor([[[1.0, 10.0], [2.0, 20.0]], [[3.0, 30.0], [4.0, 40.0]]]), torch.tensor([[0.0, 1.0], [2.0, 5.0]])

    monkeypatch.setattr(datastats.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(datastats, "OntheflyPhysicsModelLinear", FakePhysicsModel)
    monkeypatch.setattr(datastats, "HealpyDownsampling", FakeSmoothingModel)
    monkeypatch.setattr(datastats, "OntheflyPipeline", FakePipeline)
    monkeypatch.setattr(datastats, "print_profiler_stats", lambda prof: None)

    batch_stats = datastats.datastats(
        {
            "records_pattern": "records/*.tar",
            "forward_model": {"survey": "euclid", "analysis": {"n_side": 1024, "n_side_down": 128}},
            "batch_size": 1,
            "validation_fraction": 0.0,
            "use_wandb": False,
        }
    )

    captured = capsys.readouterr()

    assert len(batch_stats) == 1
    assert "channel=  0" in captured.out
    assert "min =1.0000000000e+00" in captured.out
    assert "max =4.0000000000e+00" in captured.out
    assert "mean=2.5000000000e+00" in captured.out
    assert "channel=  1" in captured.out
    assert "min =1.0000000000e+01" in captured.out
    assert "max =4.0000000000e+01" in captured.out
    assert "mean=2.5000000000e+01" in captured.out
    assert "Label statistics:" in captured.out
    assert "label=  0" in captured.out
    assert "mean=1.0000000000e+00" in captured.out
    assert "label=  1" in captured.out
    assert "max =5.0000000000e+00" in captured.out
    assert "mean=3.0000000000e+00" in captured.out
