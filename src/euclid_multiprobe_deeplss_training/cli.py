"""Command line interface for euclid-multiprobe-deeplss-training."""

from __future__ import annotations

import argparse

from euclid_multiprobe_deeplss_training import __version__


def build_parser() -> argparse.ArgumentParser:
    """Build the command line argument parser."""
    parser = argparse.ArgumentParser(
        prog="euclid-deeplss-training",
        description="Run Euclid multiprobe DeepLSS training workflows.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the configuration file.",
    )
    subparsers = parser.add_subparsers(dest="command")

    info_parser = subparsers.add_parser(
        "info",
        help="Print package information.",
    )
    info_parser.set_defaults(func=_run_info)

    parser.set_defaults(func=_run_info)
    return parser


def _run_info(_args: argparse.Namespace) -> int:
    """Print basic package information."""
    print(f"euclid-multiprobe-deeplss-training {__version__}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the command line interface."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
