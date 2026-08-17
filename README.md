# euclid-multiprobe-deeplss-training

Euclid multiprobe DeepLSS pipeline training of neural networks.

## Installation

Install the package and development dependencies with uv:

```bash
uv sync --extra dev
```

## Command line interface

The package exposes the `euclid-deeplss-training` console script:

```bash
uv run euclid-deeplss-training info
```

Show the installed version with:

```bash
uv run euclid-deeplss-training --version
```

Generate label/prediction pairs for the complete validation set from a training
checkpoint. The output is an HDF5 file containing `labels` and `predictions`
datasets:

```bash
uv run euclid-deeplss-training --config configs/example.yaml predict \
  --checkpoint checkpoints/checkpoint-latest.pt \
  --output-file predictions.h5 \
  --batch-size 32
```

## Development

Run the test suite with:

```bash
uv run pytest
```

Run linting with:

```bash
uv run ruff check .
```
