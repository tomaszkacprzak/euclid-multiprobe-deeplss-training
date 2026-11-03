from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
h5py = pytest.importorskip("h5py")
torch = pytest.importorskip("torch")
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

    document = dashboard_path.read_text(encoding="utf-8").replace(r"\u00d7", "×")
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


@pytest.mark.parametrize("binning", ["standard", "angular"])
@pytest.mark.parametrize("device_source", ["calccorrs", "training", "automatic"])
@pytest.mark.parametrize("complex_input", [False, True])
@pytest.mark.parametrize("input_kind", ["mapping", "config"])
def test_calccorrs_uses_current_nestcorr_cache_and_cpu_device(
    tmp_path, monkeypatch, binning, device_source, complex_input, input_kind
) -> None:
    from nestcorr.generate_pairs import generate_pairs
    from nestcorr.nestcorr import NestcorrFastFootprint

    from euclid_multiprobe_deeplss_training import dataloaders

    pairs_filename = generate_pairs(1, tmp_path / "pairs.h5", binning=binning)
    maps = torch.arange(2 * 48 * 2, dtype=torch.float32).reshape(2, 48, 2) / 100
    if complex_input:
        imaginary = maps.flip(1).clone()
        imaginary[..., 0] = 0
        maps = torch.complex(maps, imaginary)
    labels = torch.tensor([[0.2], [0.7]])
    indices = torch.tensor([[1], [2]], dtype=torch.int32)
    calls = {}

    class PhysicsModel(torch.nn.Module):
        num_channels = 2
        channel_spins = [0, 2]
        params = ["omega_m"]

        def __init__(self, _config, **kwargs):
            super().__init__()
            calls["physics_device"] = kwargs["device"]

        def get_weight_maps(self, values):
            return values.real + 1

        def preprocess_for_correlations(self, values):
            if torch.is_complex(values):
                return values
            # Scalar channels remain real; spin-2 channels use Q+iU.
            imaginary = values.flip(1).clone()
            imaginary[..., 0] = 0
            return torch.complex(values, imaginary)

    def pipeline(**kwargs):
        calls["pipeline"] = kwargs
        return [(maps, labels, indices), (maps[:1], labels[:1], indices[:1])]

    monkeypatch.setattr(calccorrs_module, "load_pixel_file", lambda _config: (np.arange(48), None, None, None))
    monkeypatch.setattr(calccorrs_module, "load_physics_model_class", lambda _name: PhysicsModel)
    monkeypatch.setattr(dataloaders, "OntheflyPipeline", pipeline)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    config = {
        "forward_model": {"analysis": {"n_side": 2, "n_side_down": 1}},
        "training": {
            "physics_model": "fake",
            "records_pattern": "unused-*.tar",
            "batch_size": 2,
            "num_workers": 0,
        },
        "calccorrs": {"pairs_filename": str(pairs_filename)},
    }
    if device_source == "calccorrs":
        config["calccorrs"]["device"] = "cpu"
        config["training"]["device"] = "cuda"  # The diagnostic overrides training.
    elif device_source == "training":
        config["training"]["device"] = "cpu"
    if input_kind == "config":
        config = calccorrs_module.Config.from_mapping(config)

    paths = calccorrs_module.calccorrs(
        config, output_dir=tmp_path / "output", num_batches_per_file=2, file_index=3, dataset_split="validation"
    )

    reference = NestcorrFastFootprint(
        1, [0, 2], R_footprint=0, footprint_indices=np.arange(12),
        pairs_filename=pairs_filename, symmetrize=True, matmul_precision="high",
    )
    physics = PhysicsModel({}, device=torch.device("cpu"))
    expected = reference(
        physics.preprocess_for_correlations(maps).movedim(-1, 1),
        physics.get_weight_maps(maps).movedim(-1, 1),
    )
    assert calls["physics_device"] == torch.device("cpu")
    assert calls["pipeline"]["device"] == torch.device("cpu")
    assert calls["pipeline"]["validation"] is True
    assert paths == [tmp_path / "output" / "corrs_validation_000003.h5"]
    with h5py.File(paths[0], "r") as output:
        assert set(output) == {"batch0000", "batch0001"}
        for batch_name, count in (("batch0000", 2), ("batch0001", 1)):
            group = output[batch_name]
            assert group["correlations"].shape == (count, 2, 2, reference.num_output_bins)
            np.testing.assert_allclose(group["correlations"][:], expected[:count].numpy(), equal_nan=True)
            np.testing.assert_allclose(group["separations"][:], reference.theta.numpy(), equal_nan=True)
            np.testing.assert_array_equal(group["labels"][:], labels[:count].numpy())
            np.testing.assert_array_equal(group["inds"][:], indices[:count].numpy())
    assert paths[0].with_suffix(".html").is_file()
