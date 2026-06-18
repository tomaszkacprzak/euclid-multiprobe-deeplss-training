from __future__ import annotations

from euclid_multiprobe_deeplss_training.cli import main


def test_info_command_prints_package_name(capsys) -> None:
    exit_code = main(["info"])

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "euclid-multiprobe-deeplss-training" in captured.out
