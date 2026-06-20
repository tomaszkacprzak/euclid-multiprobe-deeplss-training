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

    class FakePipeline:
        def __init__(self, records_pattern, physics_model, *, batch_size, num_workers, pin_memory, device):
            assert records_pattern == "records/*.tar"
            assert physics_model.forward_model == {"survey": "euclid"}
            assert batch_size == 1
            assert num_workers == 0
            assert pin_memory is True
            assert device == "cpu"

        def __iter__(self):
            yield torch.tensor([[[1.0, 10.0], [2.0, 20.0]], [[3.0, 30.0], [4.0, 40.0]]]), torch.tensor([0.0])

    monkeypatch.setattr(datastats.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(datastats, "OntheflyPhysicsModelLinear", FakePhysicsModel)
    monkeypatch.setattr(datastats, "OntheflyPipeline", FakePipeline)
    monkeypatch.setattr(datastats, "print_profiler_stats", lambda prof: None)

    batch_stats = datastats.datastats(
        {
            "records_pattern": "records/*.tar",
            "forward_model": {"survey": "euclid"},
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
