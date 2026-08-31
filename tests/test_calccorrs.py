from __future__ import annotations

import sys
import types

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
pytest.importorskip("healpy")
pytest.importorskip("webdataset")
calccorrs_module = pytest.importorskip("euclid_multiprobe_deeplss_training.calccorrs")
_shard_path = calccorrs_module._shard_path
calculate_batch_correlations = calccorrs_module.calculate_batch_correlations


class _Catalog:
    def __init__(self, **kwargs):
        self.values = kwargs.get("k", kwargs.get("g1"))


class _Correlation:
    offset = 0

    def __init__(self, config):
        self.nbins = config["nbins"]

    def process(self, first, second):
        value = float(np.mean(first.values) + np.mean(second.values) + self.offset)
        self.xi = np.full(self.nbins, value)
        self.xip = np.full(self.nbins, value + 1)
        self.xim = np.full(self.nbins, value - 1)


class _GG(_Correlation):
    offset = 10


class _KK(_Correlation):
    offset = 20


class _KG(_Correlation):
    offset = 30


def test_calculate_batch_correlations_separates_shear_and_other_pairs(monkeypatch) -> None:

    # TODO: Implement this
    raise NotImplementedError("Not implemented yet")
    
    
    assert result["xi_p"].shape == (2, 1, 3)
    assert result["xi_m"].shape == (2, 1, 3)
    assert result["xi"].shape == (2, 2, 3)
    assert result["shear_pair_indices"].tolist() == [[1, 1]]
    assert result["correlation_pair_indices"].tolist() == [[0, 0], [0, 1]]
    torch.testing.assert_close(result["xi_p"][0, 0], torch.full((3,), 23.0))
    torch.testing.assert_close(result["xi_m"][0, 0], torch.full((3,), 21.0))
    torch.testing.assert_close(result["xi"][0, 0], torch.full((3,), 24.0))
    torch.testing.assert_close(result["xi"][0, 1], torch.full((3,), 38.0))


def test_calculate_batch_correlations_returns_empty_shear_tensors(monkeypatch) -> None:

    # TODO: Implement this
    
    assert result["xi_p"].shape == (2, 0, 5)
    assert result["xi_m"].shape == (2, 0, 5)
    assert result["shear_pair_indices"].shape == (0, 2)


def test_shard_path_supports_patterns_and_plain_tar_paths(tmp_path) -> None:
    assert _shard_path(tmp_path / "corrs-%03d.tar", 2) == tmp_path / "corrs-002.tar"
    assert _shard_path(tmp_path / "corrs.tar", 0) == tmp_path / "corrs.tar"
    assert _shard_path(tmp_path / "corrs.tar", 2) == tmp_path / "corrs-000002.tar"
