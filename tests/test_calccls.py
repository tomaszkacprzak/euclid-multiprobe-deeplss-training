from __future__ import annotations

import pytest

h5py = pytest.importorskip("h5py")
torch = pytest.importorskip("torch")
pytest.importorskip("cuhpx")
calccls_module = pytest.importorskip("euclid_multiprobe_deeplss_training.calccls")
_append_spectra = calccls_module._append_spectra
_append_batch = calccls_module._append_batch


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


def test_append_batch_stores_labels_and_inds_with_spectra(tmp_path) -> None:
    output_path = tmp_path / "spectra.h5"

    with h5py.File(output_path, "w") as output_file:
        _append_batch(
            output_file,
            (torch.ones(2, 4),),
            torch.tensor([[0.1, 0.2], [0.3, 0.4]]),
            torch.tensor([10, 11]),
        )
        _append_batch(
            output_file,
            (torch.full((1, 4), 2.0),),
            torch.tensor([[0.5, 0.6]]),
            torch.tensor([12]),
        )

    with h5py.File(output_path, "r") as output_file:
        assert output_file["cls_0"].shape == (3, 4)
        assert output_file["labels"].shape == (3, 2)
        assert output_file["inds"].shape == (3,)
        assert output_file["labels"][:] == pytest.approx([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        assert output_file["inds"][:].tolist() == [10, 11, 12]


def test_append_batch_rejects_mismatched_batch_sizes(tmp_path) -> None:
    with h5py.File(tmp_path / "spectra.h5", "w") as output_file:
        with pytest.raises(ValueError, match="same batch size"):
            _append_batch(output_file, (torch.ones(2, 4),), torch.ones(2, 3), torch.ones(1))
