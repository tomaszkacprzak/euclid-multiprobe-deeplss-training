from __future__ import annotations

import pytest


def test_datastats_prints_channel_statistics(monkeypatch, capsys) -> None:
    torch = pytest.importorskip("torch")
    from torch.utils.data import IterableDataset

    from euclid_multiprobe_deeplss_training import datastats

    class TwoSampleDataset(IterableDataset):
        def __iter__(self):
            yield torch.tensor([[1.0, 2.0], [10.0, 20.0]]), torch.tensor([0.0])
            yield torch.tensor([[3.0, 4.0], [30.0, 40.0]]), torch.tensor([1.0])

    def fake_build_records_dataset(records_pattern, config):
        assert records_pattern == "records/*.tar"
        assert config["forward_model"] == {"survey": "euclid"}
        return TwoSampleDataset()

    monkeypatch.setattr(datastats, "build_records_dataset", fake_build_records_dataset)

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

    assert len(batch_stats) == 2
    assert "channel,min,max,mean,std" in captured.out
    assert "0,1,4,2.5" in captured.out
    assert "1,10,40,25" in captured.out
