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

Create WebDataset shards with the paths and defaults in
`forward_model.webdataset` in the merged configuration. Later configuration
files override earlier ones recursively, and command-line options can override
settings for one invocation:

```bash
uv run euclid-deeplss-training \
  --config configs/example.yaml local-overrides.yaml \
  webdataset --indices '0>9' --max-sleep 0
```

This workflow additionally requires the private `msfm` forward-model package.

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
