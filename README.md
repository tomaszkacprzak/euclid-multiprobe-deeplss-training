# euclid-multiprobe-deeplss-training

Euclid multiprobe DeepLSS pipeline training of neural networks.

## Installation

Install the package in editable mode for local development:

```bash
python -m pip install -e ".[dev]"
```

## Command line interface

The package exposes the `euclid-deeplss-training` console script:

```bash
euclid-deeplss-training info
```

Show the installed version with:

```bash
euclid-deeplss-training --version
```

## Development

Run the test suite with:

```bash
pytest
```

Run linting with:

```bash
ruff check .
```
