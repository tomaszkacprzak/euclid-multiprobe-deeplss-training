from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
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

    count = webdataset.webdataset_from_config([base, override], output_dir="overridden-output", max_sleep=0)

    assert count == 12
    assert captured["settings"].input_dir.as_posix() == "input"
    assert captured["settings"].output_dir.as_posix() == "overridden-output"
    assert captured["settings"].indices == "2>4"
    assert captured["settings"].file_suffix == "-test"
    assert captured["settings"].max_sleep == 0


def test_webdataset_settings_require_directories() -> None:
    with pytest.raises(ValueError, match="input_dir, output_dir"):
        webdataset.WebDatasetSettings.from_mapping({})


def test_full_sky_to_patch_reflects_complex64_gamma2(monkeypatch) -> None:
    # Replace healpy with the one operation this unit test needs, avoiding the
    # optional compiled dependency.  The input map has shape ``(12,)``.
    monkeypatch.setitem(sys.modules, "healpy", SimpleNamespace(nside2npix=lambda _n_side: 12))
    full_map = np.array([1 + 2j, 3 + 4j, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.complex64)

    # Two patch index arrays have shape ``(2,)``.  Patch 1 is reflected, so its
    # imaginary gamma2 component must change sign while gamma1 is preserved.
    pixel_indices = (
        np.arange(2),
        {"WL": [[np.array([0, 1]), np.array([1, 0])]]},
        {"WL": [np.array([0, 1])]},
        np.array([1, -1]),
    )
    result = webdataset.full_sky_to_patch(
        full_map,
        {"analysis": {"n_side": 1}},
        pixel_indices,
        i_z=0,
        i_patch=1,
        sample="WL",
    )

    # The flattened result has shape ``(2,)`` and retains complex64 precision.
    np.testing.assert_array_equal(result, np.array([3 - 4j, 1 - 2j], dtype=np.complex64))


@pytest.mark.parametrize(
    ("version", "with_bary", "basename"),
    [
        ("1", False, "projected_probes_maps_nobaryons512.h5"),
        ("1", True, "projected_probes_maps_baryonified512.h5"),
        ("1.1", False, "projected_probes_maps_v11dmo.h5"),
        ("1.1", True, "projected_probes_maps_v11dmb.h5"),
    ],
)
def test_get_full_sky_perm_uses_cosmogrid_filename(version, with_bary, basename) -> None:
    # Verify every supported version/matter combination, including the
    # zero-padded permutation directory in the resulting path.
    config = {"analysis": {"modelling": {"baryonified": with_bary}}}
    result = webdataset.get_full_sky_perm(version, config, "/input/cosmo", 7)

    assert result == f"/input/cosmo/perm_0007/{basename}"


def test_get_filename_webdataset_matches_existing_shard_convention() -> None:
    # The basename records survey/patch tag, simulation set, matter model, and
    # a four-digit shard index; no tensors or arrays are involved here.
    result = webdataset.get_filename_webdataset("/output", 12, "survey_patch03", "grid", with_bary=True)

    assert result == "/output/survey_patch03_grid_dmb_0012.tar"
