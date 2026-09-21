from __future__ import annotations

import pytest

from euclid_multiprobe_deeplss_training import webdataset


def test_parse_indices_supports_lists_and_inclusive_ranges() -> None:
    assert webdataset.parse_indices("0,2,4>6") == [0, 2, 4, 5, 6]
    assert webdataset.parse_indices([3, 8]) == [3, 8]


@pytest.mark.parametrize("value", ["", "3>1", "-1"])
def test_parse_indices_rejects_invalid_values(value: str) -> None:
    with pytest.raises(ValueError):
        webdataset.parse_indices(value)


def test_webdataset_from_config_uses_merged_settings_and_cli_overrides(tmp_path, monkeypatch) -> None:
    base = tmp_path / "base.yaml"
    override = tmp_path / "override.yaml"
    base.write_text(
        """
forward_model:
  webdataset:
    input_dir: input
    output_dir: output
    n_cosmos_per_file: 25
training: {}
""",
        encoding="utf-8",
    )
    override.write_text(
        """
forward_model:
  webdataset:
    indices: 2>4
    file_suffix: -test
""",
        encoding="utf-8",
    )
    captured = {}

    def fake_build(forward_model, settings, *, raw_config):
        captured.update(forward_model=forward_model, settings=settings, raw_config=raw_config)
        return 12

    monkeypatch.setattr(webdataset, "build_webdataset", fake_build)

    count = webdataset.webdataset_from_config(
        [base, override], output_dir="overridden-output", max_sleep=0
    )

    assert count == 12
    assert captured["settings"].input_dir.as_posix() == "input"
    assert captured["settings"].output_dir.as_posix() == "overridden-output"
    assert captured["settings"].indices == "2>4"
    assert captured["settings"].file_suffix == "-test"
    assert captured["settings"].max_sleep == 0


def test_webdataset_settings_require_directories() -> None:
    with pytest.raises(ValueError, match="input_dir, output_dir"):
        webdataset.WebDatasetSettings.from_mapping({})
