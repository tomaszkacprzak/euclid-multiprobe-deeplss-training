"""Tests for loading and normalizing application configuration."""

from dataclasses import fields

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
