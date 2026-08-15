"""Command line interface for euclid-multiprobe-deeplss-training."""

from __future__ import annotations

import argparse

from euclid_multiprobe_deeplss_training import __version__
from euclid_multiprobe_deeplss_training.utils import logger

LOGGER = logger.get_logger(__file__)


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
        help="Path to the configuration file.",
    )
    parser.add_argument(
        "--verbosity",
        type=str,
        default="info",
        choices=["debug", "info", "warning", "error", "critical"],
        help="Verbosity level.",
    )
    subparsers = parser.add_subparsers(dest="command")

    info_parser = subparsers.add_parser(
        "info",
        help="Print package information.",
    )
    info_parser.set_defaults(func=_run_info)

    train_parser = subparsers.add_parser(
        "train",
        help="Run training from the configuration file.",
    )
    train_parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="Checkpoint path to resume training from.",
    )
    train_parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Directory where training checkpoints are written.",
    )
    train_parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum number of training steps to run.",
    )
    train_parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device to train on, such as 'cpu' or 'cuda'.",
    )
    train_parser.add_argument(
        "--wandb-mode",
        type=str,
        default=None,
        choices=("online", "offline", "disabled"),
        help="Weights & Biases mode for this training run.",
    )
    train_parser.add_argument(
        "--tag",
        type=str,
        default='test-run',
        help="Tag for this training run.",
    )
    train_parser.set_defaults(func=_run_train)

    datastats_parser = subparsers.add_parser(
        "datastats",
        help="Print per-channel statistics for input dataset batches.",
    )
    datastats_parser.set_defaults(func=_run_datastats)

    modelprofile_parser = subparsers.add_parser(
        "modelprofile",
        help="Profile nested-transformer forward passes on input dataset batches.",
    )
    modelprofile_parser.set_defaults(func=_run_modelprofile)

    calccls_parser = subparsers.add_parser(
        "calccls",
        help="Calculate auto power spectra for one epoch of training data.",
    )
    calccls_parser.add_argument(
        "--output-path",
        type=str,
        default="cls.h5",
        help="HDF5 file to write spectra to (default: cls.h5).",
    )
    calccls_parser.add_argument(
        "--num-examples",
        type=int,
        default=100,
        help="Number of batches to calculate spectra for.",
    )
    calccls_parser.set_defaults(func=_run_calccls)

    parser.set_defaults(func=_run_info)
    return parser


def _run_info(_args: argparse.Namespace) -> int:
    """Print basic package information."""
    print(f"euclid-multiprobe-deeplss-training {__version__}")
    return 0


def _run_train(args: argparse.Namespace) -> int:
    """Run training from the parsed command line arguments."""
    if args.config is None:
        raise ValueError("The train command requires --config.")

    from euclid_multiprobe_deeplss_training.training import train_from_config

    train_from_config(
        args.config,
        resume_from_checkpoint=args.resume_from_checkpoint,
        checkpoint_dir=args.checkpoint_dir,
        max_steps=args.max_steps,
        device=args.device,
        wandb_mode=args.wandb_mode,
        tag=args.tag,
    )
    return 0


def _run_datastats(args: argparse.Namespace) -> int:
    """Print dataset input-map statistics from the parsed command line arguments."""
    if args.config is None:
        raise ValueError("The datastats command requires --config.")

    from euclid_multiprobe_deeplss_training.datastats import datastats_from_config

    datastats_from_config(args.config)
    return 0


def _run_modelprofile(args: argparse.Namespace) -> int:
    """Profile transformer forward passes from the parsed command line arguments."""
    if args.config is None:
        raise ValueError("The modelprofile command requires --config.")

    from euclid_multiprobe_deeplss_training.modelprofile import modelprofile_from_config

    modelprofile_from_config(args.config)
    return 0


def _run_calccls(args: argparse.Namespace) -> int:
    """Calculate training-set angular auto power spectra."""
    if args.config is None:
        raise ValueError("The calccls command requires --config.")

    from euclid_multiprobe_deeplss_training.calccls import calccls_from_config

    calccls_from_config(args.config, output_path=args.output_path)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the command line interface."""

    # Command line arguments
    parser = build_parser()
    args = parser.parse_args(argv)
    
    # Set logger
    logger.set_all_loggers_level(args.verbosity)

    # Run the command
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
