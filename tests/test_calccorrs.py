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


@pytest.mark.parametrize("cache_case", ["standard", "angular", "selected"])
@pytest.mark.parametrize("device_source", ["argument", "calccorrs", "training", "automatic"])
@pytest.mark.parametrize("complex_input", [False, True])
@pytest.mark.parametrize("input_kind", ["mapping", "config"])
def test_calccorrs_uses_current_nestcorr_cache_and_cpu_device(
    tmp_path, monkeypatch, cache_case, device_source, complex_input, input_kind
) -> None:
    _check_calccorrs_with_cache(tmp_path, monkeypatch, cache_case, device_source, complex_input, input_kind)


def test_calccorrs_cli_preserves_selected_angular_bins_and_overrides(tmp_path, monkeypatch) -> None:
    _check_calccorrs_with_cache(tmp_path, monkeypatch, "selected", "argument", True, "cli")


@pytest.mark.parametrize("config_form", ["contiguous", "repeated", "comma", "mixed"])
def test_calccorrs_cli_merges_multiple_configuration_files(tmp_path, monkeypatch, config_form) -> None:
    _check_calccorrs_with_cache(tmp_path, monkeypatch, "selected", "calccorrs", True, f"cli_multi_{config_form}")


def test_calccorrs_accepts_default_standard_cache(tmp_path, monkeypatch) -> None:
    _check_calccorrs_with_cache(tmp_path, monkeypatch, "standard_nd1", "automatic", False, "mapping")


def _check_calccorrs_with_cache(tmp_path, monkeypatch, cache_case, device_source, complex_input, input_kind) -> None:
    from nestcorr.generate_pairs import generate_pairs
    from nestcorr.nestcorr import NestcorrFastFootprint
    from nestcorr.select_pairs import select_pairs

    from euclid_multiprobe_deeplss_training import dataloaders

    binning = "standard" if cache_case.startswith("standard") else "angular"
    neighbour_distance = 1 if cache_case == "standard_nd1" else 2
    pairs_filename = generate_pairs(
        1, tmp_path / "pairs.h5", nd=neighbour_distance, binning=binning, bins_per_octave=2 if binning == "standard" else 4
    )
    if cache_case == "selected":
        with h5py.File(tmp_path / "pixels.h5", "w") as pixels:
            pixels["data_vec"] = [0, 1]
        pairs_filename = select_pairs(1, pairs_filename, tmp_path / "pixels.h5", tmp_path / "selected.h5")
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
            calls["physics_config"] = _config
            calls["physics_kwargs"] = kwargs

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
    overrides = {}
    if device_source == "argument":
        overrides = {"device": torch.device("cpu")}
        config["calccorrs"]["device"] = "cuda"
    elif device_source == "calccorrs":
        config["calccorrs"]["device"] = "cpu"
        config["training"]["device"] = "cuda"  # The diagnostic overrides training.
    elif device_source == "training":
        config["training"]["device"] = "cpu"
    if input_kind == "config":
        config = calccorrs_module.Config.from_mapping(config)

    if input_kind.startswith("cli"):
        import yaml

        from euclid_multiprobe_deeplss_training.cli import main

        if input_kind.startswith("cli_multi_"):
            # Wrong initial geometry/device/cache must all be replaced before
            # initialization, while unrelated nested settings remain available.
            config["forward_model"]["analysis"]["n_side"] = 4
            config["forward_model"]["survey"] = {"shared": {"retained": "base", "overridden": "base"}}
            config["training"]["batch_size"] = 8
            config["training"]["physics_model_args"] = {"nested_options": {"retained": "base", "overridden": "base"}}
            config["calccorrs"] = {"pairs_filename": str(tmp_path / "base-missing.h5"), "device": "cuda"}
            survey = {
                "forward_model": {
                    "analysis": {"n_side": 2},
                    "survey": {"shared": {"overridden": "survey", "survey_only": True}},
                },
                "training": {
                    "batch_size": 2,
                    "physics_model_args": {"nested_options": {"overridden": "survey", "survey_only": True}},
                },
                "calccorrs": {"pairs_filename": str(tmp_path / "survey-missing.h5"), "device": "cuda"},
            }
            local = {
                "forward_model": {"survey": {"shared": {"overridden": "local"}}},
                "training": {"physics_model_args": {"nested_options": {"overridden": "local"}}},
                "calccorrs": {"pairs_filename": str(pairs_filename), "device": "cpu"},
            }
            filenames = []
            for name, contents in (("base", config), ("survey", survey), ("local", local)):
                config_file = tmp_path / f"{name}.yaml"
                config_file.write_text(yaml.safe_dump(contents), encoding="utf-8")
                filenames.append(str(config_file))
            config_form = input_kind.removeprefix("cli_multi_")
            config_arguments = {
                "contiguous": ["--config", *filenames],
                "repeated": [argument for filename in filenames for argument in ("--config", filename)],
                "comma": ["--config=" + ",".join(filenames)],
                "mixed": ["--config", ",".join(filenames[:2]), "--config", filenames[2]],
            }[config_form]
            device_arguments = []
        else:
            config_file = tmp_path / "config.yaml"
            config_file.write_text(yaml.safe_dump(config), encoding="utf-8")
            config_arguments = ["--config", str(config_file)]
            device_arguments = ["--device", "cpu"]
        result = main([
            *config_arguments, "calccorrs", *device_arguments,
            "--output-dir", str(tmp_path / "output"), "--num-batches-per-file", "2", "--file-index", "3",
            "--dataset-split", "validation",
        ])
        assert result == 0
        paths = [tmp_path / "output" / "corrs_validation_000003.h5"]
        if input_kind.startswith("cli_multi_"):
            assert calls["physics_config"]["analysis"] == {"n_side": 2, "n_side_down": 1}
            assert calls["physics_config"]["survey"]["shared"] == {
                "retained": "base", "overridden": "local", "survey_only": True,
            }
            assert calls["physics_kwargs"]["nested_options"] == {
                "retained": "base", "overridden": "local", "survey_only": True,
            }
            assert calls["pipeline"]["batch_size"] == 2
            assert calls["pipeline"]["num_workers"] == 0
            assert calls["pipeline"]["webds_pattern"] == "unused-*.tar"
            if input_kind == "cli_multi_mixed":
                from euclid_multiprobe_deeplss_training import training

                trained = {}

                def fake_train(raw_config, *, device=None):
                    trained.update(config=raw_config, device=device)
                    return {"step": 0}

                monkeypatch.setattr(training, "train", fake_train)
                assert main([*config_arguments, "train", "--device", "cpu"]) == 0
                # Train adds runtime defaults; compare the settings supplied
                # by these files rather than requiring an exact training map.
                assert trained["config"]["forward_model"] == calls["physics_config"]
                assert trained["config"]["calccorrs"] == {"pairs_filename": str(pairs_filename), "device": "cpu"}
                assert trained["config"]["training"]["batch_size"] == calls["pipeline"]["batch_size"]
                assert trained["config"]["training"]["records_pattern"] == calls["pipeline"]["webds_pattern"]
                assert trained["config"]["training"]["physics_model_args"]["nested_options"] == calls["physics_kwargs"]["nested_options"]
    else:
        paths = calccorrs_module.calccorrs(
            config, output_dir=tmp_path / "output", num_batches_per_file=2, file_index=3, dataset_split="validation", **overrides
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
        assert {name for name in output if name.startswith("batch")} == {"batch0000", "batch0001"}
        assert output.attrs["pairs_filename"] == str(pairs_filename.resolve())
        assert output.attrs["fine_order"] == 1
        assert output.attrs["neighbour_distance"] == neighbour_distance
        assert output.attrs["output_binning"] == binning
        assert output.attrs["num_output_bins"] == reference.num_output_bins
        assert output.attrs["separation_units"] == "radian"
        np.testing.assert_allclose(output["separations"][:], reference.theta.numpy(), equal_nan=True)
        np.testing.assert_allclose(output["theta_range"][:], reference.theta_range.numpy(), equal_nan=True)
        if binning == "angular":
            assert output.attrs["bin_order"] == "increasing angular interval"
            assert output.attrs["bins_per_octave"] == 4
            np.testing.assert_array_equal(output["angular_bin_edges"][:], reference.angular_bin_edges.numpy())
        else:
            assert output.attrs["bin_order"] == "descending inherited output order (L, L-1, ..., 0)"
            np.testing.assert_array_equal(output["output_orders"][:], [1, 0])
        if cache_case == "selected":
            assert output.attrs["selected_pixel_count"] == 2
            assert np.isnan(output["batch0000/correlations"][:]).any()
        for batch_name, count in (("batch0000", 2), ("batch0001", 1)):
            group = output[batch_name]
            assert group["correlations"].shape == (count, 2, 2, reference.num_output_bins)
            np.testing.assert_allclose(group["correlations"][:], expected[:count].numpy(), equal_nan=True)
            np.testing.assert_allclose(group["separations"][:], reference.theta.numpy(), equal_nan=True)
            np.testing.assert_array_equal(group["labels"][:], labels[:count].numpy())
            np.testing.assert_array_equal(group["inds"][:], indices[:count].numpy())
    assert paths[0].with_suffix(".html").is_file()
    document = paths[0].with_suffix(".html").read_text(encoding="utf-8")
    assert "correlation_bins" in document
    assert "neighbour_distance" in document
    assert "Separation (radian)" in document


@pytest.mark.parametrize("cache_case", ["unspecified", "missing", "wrong_order"])
def test_calccorrs_rejects_pair_file_before_initializing_survey(tmp_path, monkeypatch, cache_case) -> None:
    from nestcorr.generate_pairs import generate_pairs

    def unexpected_survey_load(_config):
        raise AssertionError("The survey must not load before the pair file is validated.")

    monkeypatch.setattr(calccorrs_module, "load_pixel_file", unexpected_survey_load)
    config = {"forward_model": {"analysis": {"n_side": 4, "n_side_down": 1}}}
    config["calccorrs"] = {}
    if cache_case == "unspecified":
        error_type, message = ValueError, "set calccorrs.pairs_filename"
    elif cache_case == "missing":
        config["calccorrs"]["pairs_filename"] = tmp_path / "missing.h5"
        error_type, message = FileNotFoundError, "nestcorr generate_pairs 2"
    else:
        config["calccorrs"]["pairs_filename"] = generate_pairs(1, tmp_path / "L01.h5")
        error_type, message = RuntimeError, "built for L=1, not L=2"
    with pytest.raises(error_type, match=message):
        calccorrs_module.calccorrs(config, output_dir=tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_calccorrs_from_config_preserves_pair_setting_and_forwards_device_override(tmp_path, monkeypatch) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text("calccorrs:\n  pairs_filename: configured.h5\n  device: cuda\n", encoding="utf-8")
    received = {}

    def fake_calccorrs(config, **kwargs):
        received.update(config=config, **kwargs)
        return [tmp_path / "result.h5"]

    monkeypatch.setattr(calccorrs_module, "calccorrs", fake_calccorrs)
    result = calccorrs_module.calccorrs_from_config(config_file, device="cpu")
    assert result == [tmp_path / "result.h5"]
    assert "pairs_filename" not in received
    assert received["device"] == "cpu"
    assert received["num_batches_per_file"] == 100
    assert received["config"]["calccorrs"]["pairs_filename"] == "configured.h5"
