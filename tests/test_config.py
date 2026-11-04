"""Tests for loading and normalizing application configuration."""

from dataclasses import fields

import pytest

from euclid_multiprobe_deeplss_training.utils.config import Config, load_config


def test_config_provides_a_default_for_every_field() -> None:
    config = Config()

    assert {item.name for item in fields(Config)} == {
        item.name for item in fields(Config) if hasattr(config, item.name)
    }


def test_load_config_recursively_merges_files_in_order(tmp_path) -> None:
    base = tmp_path / "base.yaml"
    override = tmp_path / "override.yaml"
    base.write_text(
        "records_pattern: records/*.tar\nbatch_size: 32\nmodel:\n  encoder_args:\n    width: 64\n    depth: 2\n",
        encoding="utf-8",
    )
    override.write_text(
        "batch_size: 8\nmodel:\n  encoder_args:\n    depth: 4\n",
        encoding="utf-8",
    )

    merged = load_config([base, override])
    config = Config.from_mapping(merged)

    assert config.records_pattern == "records/*.tar"
    assert config.batch_size == 8
    assert config.encoder_args == {"width": 64, "depth": 4}


def test_load_config_accepts_one_file(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("batch_size: 4\n", encoding="utf-8")

    assert load_config(path) == {"batch_size": 4}


@pytest.mark.parametrize("path_form", ["comma_string", "comma_list", "mixed"])
def test_load_config_merges_comma_separated_paths_in_order(tmp_path, path_form) -> None:
    base = tmp_path / "base.yaml"
    experiment = tmp_path / "experiment.yaml"
    local = tmp_path / "local.yaml"
    base.write_text("training:\n  batch_size: 8\n  encoder_args:\n    depth: 2\n    dropout: 0.1\n", encoding="utf-8")
    experiment.write_text("training:\n  encoder_args:\n    depth: 4\ncalccorrs:\n  pairs_filename: standard.h5\n", encoding="utf-8")
    local.write_text("training:\n  batch_size: 2\ncalccorrs:\n  pairs_filename: angular.h5\n", encoding="utf-8")
    if path_form == "comma_string":
        paths = f"{base}, {experiment}, {local}"
    elif path_form == "comma_list":
        paths = [f"{base},{experiment}", str(local)]
    else:
        paths = [base, f"{experiment}, , {local},"]

    assert load_config(paths) == {
        "training": {"batch_size": 2, "encoder_args": {"depth": 4, "dropout": 0.1}},
        "calccorrs": {"pairs_filename": "angular.h5"},
    }


def test_load_config_preserves_literal_path_with_comma(tmp_path) -> None:
    path = tmp_path / "config,local.yaml"
    path.write_text("batch_size: 4\n", encoding="utf-8")

    assert load_config(path) == {"batch_size": 4}


@pytest.mark.parametrize("paths", [[], "", " , , ", ["", ","]])
def test_load_config_requires_at_least_one_path(paths) -> None:
    with pytest.raises(ValueError, match="At least one configuration file is required"):
        load_config(paths)


def test_config_reads_training_and_forward_model_from_master_mapping() -> None:
    config = Config.from_mapping(
        {
            "forward_model": {"analysis": {"n_side": 512}},
            "training": {
                "records_pattern": "records/*.tar",
                "batch_size": 8,
                "physics_model_args": {"shape_noise_std": 0.001},
            },
        }
    )

    assert config.forward_model == {"analysis": {"n_side": 512}}
    assert config.records_pattern == "records/*.tar"
    assert config.batch_size == 8
    assert config.physics_model_args == {"shape_noise_std": 0.001}


def test_training_section_takes_precedence_over_legacy_flat_values() -> None:
    config = Config.from_mapping({"batch_size": 32, "training": {"batch_size": 4}})

    assert config.batch_size == 4


def test_training_section_must_be_a_mapping() -> None:
    import pytest

    with pytest.raises(TypeError, match="training.*must be a mapping"):
        Config.from_mapping({"training": []})
