from __future__ import annotations

import pytest

h5py = pytest.importorskip("h5py")
torch = pytest.importorskip("torch")
pytest.importorskip("cuhpx")
calccls_module = pytest.importorskip("euclid_multiprobe_deeplss_training.calccls")
_append_spectra = calccls_module._append_spectra


def test_append_spectra_creates_and_extends_probe_datasets(tmp_path) -> None:
    output_path = tmp_path / "spectra.h5"

    with h5py.File(output_path, "w") as output_file:
        _append_spectra(output_file, (torch.ones(2, 4), torch.full((2, 2, 4), 2.0)))
        _append_spectra(output_file, (torch.full((1, 4), 3.0), torch.full((1, 2, 4), 4.0)))

    with h5py.File(output_path, "r") as output_file:
        assert output_file["cls_0"].shape == (3, 4)
        assert output_file["cls_1"].shape == (3, 2, 4)
        assert output_file["cls_0"][-1].tolist() == [3.0] * 4
        assert output_file["cls_1"][-1].tolist() == [[4.0] * 4] * 2
