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

## Development

Run the test suite with:

```bash
uv run pytest
```

Run linting with:

```bash
uv run ruff check .
```
