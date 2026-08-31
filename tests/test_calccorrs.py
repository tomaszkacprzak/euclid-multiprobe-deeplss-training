from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
h5py = pytest.importorskip("h5py")
pytest.importorskip("torch")
pytest.importorskip("healpy")
pytest.importorskip("webdataset")
calccorrs_module = pytest.importorskip("euclid_multiprobe_deeplss_training.calccorrs")
create_correlations_dashboard = calccorrs_module.create_correlations_dashboard
get_footprint_indices = calccorrs_module.get_footprint_indices


def test_get_footprint_indices_deduplicates_downsampled_pixels() -> None:
    assert get_footprint_indices(np.array([0, 1, 3, 4, 7]), 2, 1).tolist() == [0, 1]


def test_correlations_dashboard_loads_all_batches_and_unique_channel_pairs(tmp_path) -> None:
    correlations_path = tmp_path / "corrs.h5"
    dashboard_path = tmp_path / "corrs.html"
    with h5py.File(correlations_path, "w") as output_file:
        for batch_index in range(2):
            group = output_file.create_group(f"batch{batch_index:04d}")
            group["separations"] = [0.1, 0.2, 0.3]
            group["correlations"] = np.arange(2 * 3 * 3 * 3, dtype=float).reshape(2, 3, 3, 3) + batch_index * 100
            group["labels"] = [[0.2 + batch_index, 0.7], [0.4 + batch_index, 0.9]]

    result = create_correlations_dashboard(
        correlations_path,
        dashboard_path,
        parameter_names=["omega_m", "sigma8"],
        model_information={"physics_model": "example"},
    )

    document = dashboard_path.read_text(encoding="utf-8")
    assert result == dashboard_path
    assert "Correlations Dashboard" in document
    assert "Channels 0 × 0" in document
    assert "Channels 0 × 2" in document
    assert "Channels 2 × 2" in document
    assert "Channels 1 × 0" not in document
    assert "Example 3" in document
    assert '"label":"omega_m"' in document
    assert '"label":"sigma8"' in document
    assert "plotly.js" in document
    assert '<script src="https://cdn.plot.ly' not in document


def test_correlations_dashboard_requires_parameter_name_for_each_label(tmp_path) -> None:
    correlations_path = tmp_path / "corrs.h5"
    with h5py.File(correlations_path, "w") as output_file:
        group = output_file.create_group("batch0000")
        group["separations"] = [0.1, 0.2]
        group["correlations"] = np.ones((1, 2, 2, 2))
        group["labels"] = [[0.2, 0.7]]

    with pytest.raises(ValueError, match="1 parameter names for 2 label columns"):
        create_correlations_dashboard(
            correlations_path,
            tmp_path / "corrs.html",
            parameter_names=["omega_m"],
        )
