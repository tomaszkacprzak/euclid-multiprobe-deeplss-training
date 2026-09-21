"""Command line interface for euclid-multiprobe-deeplss-training."""

from __future__ import annotations

import argparse
import sys

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
        action="append",
        help="Path to one or more YAML configuration files (later files override earlier files).",
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



    #####################################################################################
    #
    # webdataset
    #
    #####################################################################################

    webdataset_parser = subparsers.add_parser(
        "webdataset",
        help="Run webdataset workflow.",
    )

    webdataset_parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Override the input root directory from the configuration.",
    )
    webdataset_parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override the output root directory from the configuration.",
    )
    webdataset_parser.add_argument(
        "--cosmogrid-version",
        type=str,
        default=None,
        choices=["1.1", "1"],
        help="version of the input CosmoGrid",
    )
    webdataset_parser.add_argument(
        "--file-suffix",
        type=str,
        default=None,
        help="Optional suffix to be appended to the end of the filename, for example to distinguish different runs",
    )
    webdataset_parser.add_argument(
        "--max-sleep",
        type=float,
        default=None,
        help="set the maximal amount of time to sleep before copying to avoid clashes",
    )
    webdataset_parser.add_argument(
        "--indices", 
        type=str, 
        default=None,
        help="Indices to process, formatted as 0,1,2,4 or an inclusive start>stop range.")

    webdataset_parser.add_argument(
        "--n-cosmos-per-file",
        type=int, 
        default=None,
        help="Override the number of cosmologies per output file.")
    webdataset_parser.add_argument(
        "--debug",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable debug mode from the configuration.",
    )
    webdataset_parser.set_defaults(func=_run_webdataset)

    #####################################################################################
    #
    # train
    #
    #####################################################################################

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

    #####################################################################################
    #
    # predict
    #
    #####################################################################################

    predict_parser = subparsers.add_parser(
        "predict",
        help="Run prediction on the full validation set.",
    )
    predict_parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Training checkpoint containing the model weights.",
    )
    predict_parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="HDF5 file to write labels and predictions to.",
    )
    predict_parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Validation batch size (defaults to batch_size in the config).",
    )
    predict_parser.add_argument(
        "--num-examples",
        type=int,
        default=1000,
        help="Number of examples to predict for.",
    )
    predict_parser.add_argument(
        "--device",
        type=str,
        default='cuda',
        help="Torch device to evaluate on, such as 'cpu' or 'cuda'.",
    )
    predict_parser.set_defaults(func=_run_predict)

    #####################################################################################
    #
    # likelihood
    #
    #####################################################################################

    likelihood_parser = subparsers.add_parser(
        "likelihood",
        help="Train a conditional likelihood from a prediction HDF5 file.",
    )
    likelihood_parser.add_argument("--input-file", required=True, help="HDF5 output produced by the predict command.")
    likelihood_parser.add_argument("--output-file", required=True, help="File in which to store the trained likelihood model.")
    likelihood_parser.add_argument("--device", default=None, help="Torch device to train on, such as 'cpu' or 'cuda'.")
    likelihood_parser.set_defaults(func=_run_likelihood)

    #####################################################################################
    #
    # datastats
    #
    #####################################################################################

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

    #####################################################################################
    #
    # calccls
    #
    #####################################################################################

    calccls_parser = subparsers.add_parser(
        "calccls",
        help="Calculate auto and cross power spectra for one epoch of training data.",
    )
    calccls_parser.add_argument(
        "--output-path",
        type=str,
        default="cls.h5",
        help="HDF5 file to write auto spectra to; cross spectra use an _cross suffix (default: cls.h5).",
    )
    calccls_parser.add_argument(
        "--num-examples",
        type=int,
        default=100,
        help="Number of examples to calculate spectra for.",
    )
    calccls_parser.set_defaults(func=_run_calccls)

    #####################################################################################
    #
    # calccorrs
    #
    #####################################################################################

    calccorrs_parser = subparsers.add_parser(
        "calccorrs",
        help="Calculate scalar and shear two-point correlations for training data.",
    )
    calccorrs_parser.add_argument(
        "--output-dir",
        type=str,
        default='corrs',
        help="Directory to write correlations to (default: corrs).",
    )
    calccorrs_parser.add_argument(
        "--file-index",
        type=int,
        default=0,
        help="Index of the WebDataset shard to calculate correlations for.",
    )
    calccorrs_parser.add_argument(
        "--num-batches-per-file",
        type=int,
        default=100,
        help="Number of input batches stored in each WebDataset tar shard.",
    )
    calccorrs_parser.add_argument(
        "--dataset-split",
        type=str,
        default='training',
        choices=['training', 'validation'],
        help="Type of dataset to calculate correlations for.",
    )
    calccorrs_parser.set_defaults(func=_run_calccorrs)

    parser.set_defaults(func=_run_info)
    return parser


def _config_argument(args: argparse.Namespace) -> str | list[str]:
    """Preserve the scalar form for one path and return all paths otherwise."""
    return args.config[0] if len(args.config) == 1 else args.config


def _run_info(_args: argparse.Namespace) -> int:
    """Print basic package information."""
    print(f"euclid-multiprobe-deeplss-training {__version__}")
    return 0

def _run_webdataset(args: argparse.Namespace) -> int:
    """Run webdataset workflow."""
    if args.config is None:
        raise ValueError("The webdataset command requires --config.")

    from euclid_multiprobe_deeplss_training.webdataset import webdataset_from_config

    webdataset_from_config(
        _config_argument(args),
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        indices=args.indices,
        cosmogrid_version=args.cosmogrid_version,
        file_suffix=args.file_suffix,
        max_sleep=args.max_sleep,
        n_cosmos_per_file=args.n_cosmos_per_file,
        debug=args.debug,
    )
    return 0


def _run_train(args: argparse.Namespace) -> int:
    """Run training from the parsed command line arguments."""
    if args.config is None:
        raise ValueError("The train command requires --config.")

    from euclid_multiprobe_deeplss_training.training import train_from_config

    train_from_config(
        _config_argument(args),
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

    datastats_from_config(_config_argument(args))
    return 0


def _run_predict(args: argparse.Namespace) -> int:
    """Generate predictions for the full validation set."""
    if args.config is None:
        raise ValueError("The predict command requires --config.")

    from euclid_multiprobe_deeplss_training.prediction import predict_from_config

    predict_from_config(
        _config_argument(args),
        checkpoint=args.checkpoint,
        output_file=args.output_file,
        batch_size=args.batch_size,
        num_examples=args.num_examples,
        device=args.device,
    )
    return 0


def _run_likelihood(args: argparse.Namespace) -> int:
    """Train a conditional likelihood model from saved predictions."""
    if args.config is None:
        raise ValueError("The likelihood command requires --config.")
    from euclid_multiprobe_deeplss_training.likelihood.likelihood_training import train_likelihood_from_config

    train_likelihood_from_config(
        _config_argument(args), input_file=args.input_file, output_file=args.output_file, device=args.device
    )
    return 0


def _run_modelprofile(args: argparse.Namespace) -> int:
    """Profile transformer forward passes from the parsed command line arguments."""
    if args.config is None:
        raise ValueError("The modelprofile command requires --config.")

    from euclid_multiprobe_deeplss_training.modelprofile import modelprofile_from_config

    modelprofile_from_config(_config_argument(args))
    return 0


def _run_calccls(args: argparse.Namespace) -> int:
    """Calculate training-set angular auto and cross power spectra."""
    if args.config is None:
        raise ValueError("The calccls command requires --config.")

    from euclid_multiprobe_deeplss_training.calccls import calccls_from_config

    kwargs = {"output_path": args.output_path}
    if args.num_examples != 100:
        kwargs["num_examples"] = args.num_examples
    calccls_from_config(_config_argument(args), **kwargs)
    return 0


def _run_calccorrs(args: argparse.Namespace) -> int:
    """Calculate training-set angular two-point correlations."""
    if args.config is None:
        raise ValueError("The calccorrs command requires --config.")

    from euclid_multiprobe_deeplss_training.calccorrs import calccorrs_from_config

    kwargs = {
        "output_dir": args.output_dir,
        "num_batches_per_file": args.num_batches_per_file,
        "file_index": args.file_index,
        "dataset_split": args.dataset_split,
    }
    calccorrs_from_config(_config_argument(args), **kwargs)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the command line interface."""

    # Command line arguments
    parser = build_parser()
    argv = _expand_config_arguments(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(argv)
    
    # Set logger
    logger.set_all_loggers_level(args.verbosity)

    # Run the command
    return args.func(args)


def _expand_config_arguments(argv: list[str]) -> list[str]:
    """Allow ``--config base.yaml override.yaml command`` with subparsers.

    ``argparse`` cannot combine a variable-length option with a following
    subcommand, so contiguous config values are normalized to repeated options
    before parsing. Repeated ``--config`` options continue to work directly.
    """
    commands = {"info", "webdataset", "train", "predict", "likelihood", "datastats", "modelprofile", "calccls", "calccorrs"}
    expanded: list[str] = []
    index = 0
    while index < len(argv):
        value = argv[index]
        if value != "--config":
            expanded.append(value)
            index += 1
            continue
        index += 1
        if index >= len(argv) or argv[index].startswith("-") or argv[index] in commands:
            expanded.append("--config")
            continue
        while index < len(argv) and not argv[index].startswith("-") and argv[index] not in commands:
            expanded.extend(("--config", argv[index]))
            index += 1
    return expanded


if __name__ == "__main__":
    raise SystemExit(main())
