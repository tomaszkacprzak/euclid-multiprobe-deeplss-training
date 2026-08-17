from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
h5py = pytest.importorskip("h5py")
torch = pytest.importorskip("torch")
pytest.importorskip("cuhpx")
calccls_module = pytest.importorskip("euclid_multiprobe_deeplss_training.calccls")
_append_spectra = calccls_module._append_spectra
_append_batch = calccls_module._append_batch
_cross_output_path = calccls_module._cross_output_path
create_cross_power_spectra_dashboard = calccls_module.create_cross_power_spectra_dashboard
create_power_spectra_dashboard = calccls_module.create_power_spectra_dashboard
_smooth_spectrum = calccls_module._smooth_spectrum


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


def test_cross_output_path_appends_cross_to_auto_spectrum_stem(tmp_path) -> None:
    assert _cross_output_path(tmp_path / "spectra.h5") == tmp_path / "spectra_cross.h5"


def test_power_spectra_dashboard_is_self_contained_and_includes_model_information(tmp_path) -> None:
    spectra_path = tmp_path / "spectra.h5"
    dashboard_path = tmp_path / "spectra.html"
    with h5py.File(spectra_path, "w") as output_file:
        output_file["labels"] = [[0.2, 0.7], [0.4, 0.9]]
        output_file["cls_0"] = [[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]
        output_file["cls_1"] = [[4.0, 5.0, 6.0], [5.0, 6.0, 7.0]]

    result = create_power_spectra_dashboard(
        spectra_path,
        dashboard_path,
        parameter_names=["omega_m", "sigma8"],
        model_information={
            "physics_model": "onthefly_linkappa",
            "shape_noise_std": 0.3,
            "config_forward_model": "forward.yaml",
        },
    )

    document = dashboard_path.read_text(encoding="utf-8")
    assert result == dashboard_path
    assert "Angular Power Spectra Dashboard" in document
    assert "Model information" in document
    assert "onthefly_linkappa" in document
    assert "shape_noise_std" in document
    assert "omega_m" in document and "sigma8" in document
    assert '"label":"S8"' in document
    assert "plotly.js" in document
    assert "plotly_hover" in document
    assert "'line.width': 4" in document
    assert "https://cdn.plot.ly" not in document
    assert "#3b4cc0" in document and "#b40426" in document


def test_smooth_spectrum_uses_length_ten_uniform_kernel() -> None:
    values = torch.arange(20, dtype=torch.float64).numpy()
    expected = pytest.approx(np.convolve(values, np.ones(10) / 10, mode="same"))

    assert _smooth_spectrum(values) == expected


def test_power_spectra_dashboard_requires_parameter_name_for_each_label(tmp_path) -> None:
    spectra_path = tmp_path / "spectra.h5"
    with h5py.File(spectra_path, "w") as output_file:
        output_file["labels"] = [[0.2, 0.7]]
        output_file["cls_0"] = [[1.0, 2.0]]

    with pytest.raises(ValueError, match="1 parameter names for 2 label columns"):
        create_power_spectra_dashboard(
            spectra_path,
            tmp_path / "spectra.html",
            parameter_names=["omega_m"],
            model_information={},
        )


def test_cross_power_spectra_dashboard_has_probe_sliders_and_color_dropdown(tmp_path) -> None:
    spectra_path = tmp_path / "spectra_cross.h5"
    dashboard_path = tmp_path / "spectra_cross.html"
    with h5py.File(spectra_path, "w") as output_file:
        output_file["labels"] = [[0.2, 0.7], [0.4, 0.9]]
        # Three probe pairs, ordered (0, 0), (0, 1), (1, 1).
        output_file["cls_0"] = np.arange(2 * 4 * 3, dtype=float).reshape(2, 4, 3)

    result = create_cross_power_spectra_dashboard(
        spectra_path,
        dashboard_path,
        parameter_names=["omega_m", "sigma8"],
        model_information={"physics_model": "onthefly_linkappa"},
    )

    document = dashboard_path.read_text(encoding="utf-8")
    assert result == dashboard_path
    assert "Cross Angular Power Spectra Dashboard" in document
    assert 'id="map1-probe" type="range" min="0" max="1"' in document
    assert 'id="map2-probe" type="range" min="0" max="1"' in document
    assert "map1Slider.addEventListener('input', selectProbePair)" in document
    assert "map2Slider.addEventListener('input', selectProbePair)" in document
    assert '"map1":0,"map2":1' in document
    assert '"label":"omega_m"' in document
    assert '"label":"sigma8"' in document
    assert '"label":"S8"' in document
    assert "plotly.js" in document
    assert "https://cdn.plot.ly" not in document


def test_cross_power_spectra_dashboard_rejects_non_triangular_pair_count(tmp_path) -> None:
    spectra_path = tmp_path / "spectra_cross.h5"
    with h5py.File(spectra_path, "w") as output_file:
        output_file["labels"] = [[0.2, 0.7]]
        output_file["cls_0"] = np.ones((1, 4, 2))

    with pytest.raises(ValueError, match="pair dimension .* is not triangular"):
        create_cross_power_spectra_dashboard(
            spectra_path,
            tmp_path / "spectra_cross.html",
            parameter_names=["omega_m", "sigma8"],
            model_information={},
        )
