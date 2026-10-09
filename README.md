# euclid-multiprobe-deeplss-training

PyTorch tools for training and inference in the Euclid multiprobe DeepLSS
pipeline. The repository packages projected CosmoGrid maps into survey-patch
datasets, applies on-the-fly physics models, trains neural summaries of
large-scale structure, and fits conditional likelihoods to their predictions.

Applications include joint weak-lensing and galaxy-clustering parameter
inference, cosmological and nuisance-parameter estimation, and comparisons of
map-based and two-point-statistic encoders. Survey geometry, redshift bins,
noise, modelling choices and priors are configured in YAML. The repository
consumes simulation products and survey auxiliary files; it does not generate
the underlying cosmological simulations.

## Installation

Python 3.10 or later is required. Install with uv:

```bash
uv sync --extra dev
uv run euclid-deeplss-training info
```

Power-spectrum routes require `cuhpx` and CUDA; correlation routes require
`pyracorr` and use CUDA. `trianglechain` is fetched from the external GitLab
source in `pyproject.toml`. A complete installation requires access to that
source and an installation source for `pyracorr`; `cuhpx` is imported by the
relevant modules but is not declared as a runtime dependency. CPU execution is
available for compatible model routes, not every command.

The supplied configs contain machine-specific dataset and auxiliary-file paths.
Adapt them before running data-dependent commands.

## Workflow

1. `webdataset`: package projected simulation maps into tar shards.
2. `datastats`: inspect map channels and parameter labels.
3. Optionally use `calccls` or `calccorrs` to inspect two-point statistics, and
   `modelprofile` to inspect model cost.
4. `train`: learn a neural summary/parameter predictor.
5. `predict`: apply a checkpoint to validation examples and save label/prediction pairs.
6. `likelihood`: fit a conditional density from those pairs and optionally sample posteriors.

The statistics commands are diagnostic branches, not prerequisites for training:
statistic-based encoders compute their summaries inside the model. The loader
uses a fixed 20% shard split for validation and the remaining 80% for training,
selected from `training.records_pattern`.

After adapting paths and completing the settings, a typical sequence is:

```bash
# Skip packaging if WebDataset shards already exist.
uv run euclid-deeplss-training --config configs/example.yaml \
  webdataset --indices '0>9' --max-sleep 0

# Shared survey settings + experiment settings + local overrides.
uv run euclid-deeplss-training --config configs/example.yaml \
  configs/conf_nside512_EuclidDR1F.yaml configs/conf_nside512_mse_mapnvit.yaml \
  local-overrides.yaml train --tag map-experiment --wandb-mode offline

# Reuse the resolved model and physics configuration.
uv run euclid-deeplss-training --config results/map-experiment/config.yaml \
  predict --checkpoint results/map-experiment/checkpoint-latest.pt \
  --output-file predictions.h5 --num-examples 1000 --device cuda

uv run euclid-deeplss-training --config results/map-experiment/config.yaml \
  likelihood --input-file predictions.h5 --output-file likelihood.pt
```

Use a distinct training tag for each new run: starting without a resume
checkpoint clears existing contents of `checkpoint_dir/tag`.

## Configuration

Every data/model command requires `--config`. Supply YAML filenames **before the
subcommand**. Both forms below work:

```bash
uv run euclid-deeplss-training --config base.yaml experiment.yaml local.yaml train
uv run euclid-deeplss-training --config base.yaml --config experiment.yaml \
  --config local.yaml train
```

Files merge from left to right. Nested mappings merge recursively; later scalar
values, lists and nulls replace earlier values. Unrelated keys survive. A file
can contain only the sections/options it contributes, so survey, experiment,
likelihood and local settings can live in separate files. CLI overrides apply
after merging. Some flags have explicit defaults that also override YAML values.

| Top-level section | Purpose and consumers |
| --- | --- |
| `forward_model` | Survey and physics: `dirs`, `files`, `survey`, `analysis`, `master_seed`, modelling and priors. Used by map workflows and likelihood posterior plots. `forward_model.webdataset` holds shard-generation options. |
| `training` | Records pattern, physics model, encoder, loss, optimization, batching, device, validation, checkpoints and W&B. Also supplies data/model settings for diagnostics and prediction. |
| `likelihood` | Density-model type/arguments, fitting settings and optional MCMC settings for `likelihood`. |
| `calccls` | Optional `lmax` override for the power-spectrum diagnostic. |

The supplied files illustrate this organization:

- `configs/example.yaml`: combined reference settings for the three main sections.
- `configs/conf_nside512_EuclidDR1F.yaml`: shared survey/physics settings.
- `configs/conf_nside512_mse_mapnvit.yaml`: map-transformer training and likelihood settings.
- `configs/conf_nside512_mse_corrswin.yaml`: correlation-transformer training and likelihood settings.
- A user-created local file: machine paths, devices, batch sizes and logging overrides.

For example, `local-overrides.yaml` could contain:

```yaml
forward_model:
  webdataset:
    input_dir: /data/CosmoGridV1
    output_dir: /data/shards
training:
  records_pattern: /data/shards/EuclidDR1F*.tar
  checkpoint_dir: results
  batch_size: 8
  num_workers: 2
  use_ddp: false
  wandb_mode: offline
likelihood:
  batch_size: 128
```

There are no separate top-level `predict`, `datastats` or `modelprofile`
sections: these commands reuse `training`. The experiment files contain
`training.config_forward_model`, but the current loader does **not** use it to
include another file automatically; pass the survey file explicitly.
`configs/example.yaml` is not loaded implicitly either. Some workflows index
required keys directly, so ensure the merged configuration contains their
complete settings. Relative config filenames are resolved from the working
directory; resource paths follow the consuming module's path handling.

## Commands in workflow order

All commands accept `--help`. Global options are `--config`,
`--verbosity {debug,info,warning,error,critical}` (default `info`) and `--version`.
Place global options before the subcommand, for example:

```bash
uv run euclid-deeplss-training --config configs/example.yaml --verbosity debug train --help
```

### 1. `info` — inspect the installed package

**Config:** none. **Additional arguments:** none.
**Output:** package name and version on stdout. Running without a subcommand
also prints this information; `--version` prints the console-script version.

### 2. `webdataset` — prepare map shards

**Config:** `forward_model`: survey/map selection, analysis settings, pixel and
metadata files, and nested `forward_model.webdataset` options. Neither
`training` nor `likelihood` is required.

| Additional argument | Meaning |
| --- | --- |
| `--input-dir PATH` | Override the projected CosmoGrid input directory. |
| `--output-dir PATH` | Override the shard output directory. |
| `--cosmogrid-version {1,1.1}` | Select the input file conventions. |
| `--file-suffix TEXT` | Add a suffix to shard filenames. |
| `--indices SPEC` | Processing indices, e.g. `0,1,4` or inclusive `'0>9'`. |
| `--n-cosmos-per-file N` | Cosmology block size per output file. |
| `--max-sleep SECONDS` | Maximum pre-copy delay to reduce I/O clashes. |
| `--debug` / `--no-debug` | Override debug mode. |

All flags default to unset and override their corresponding nested config keys.
**Output:** tar shards per survey patch and processing index, with
`maps_float32.pth` map tensors, `vec_int32.pth` metadata and `vec_float32.pth`
parameters; a merged `config.yaml` is saved alongside them. Debug output goes
into an output `debug/` subdirectory.

### 3. `datastats` — inspect the training data

**Config:** `forward_model` and `training`: survey/physics and resolution settings,
`records_pattern`, `batch_size` and `num_workers`. The current implementation
uses the linear on-the-fly physics model.
**Additional arguments:** none.
**Output:** map-channel/label statistics and profiler summaries on stdout/logs,
`maps_batch_1.npy` in the working directory, and scheduled traces under
`torch_profiler_logs/`. Device selection is automatic (CUDA if available, else CPU).

### 4. `calccls` — inspect angular power spectra (optional)

**Config:** `forward_model` and `training` for survey geometry, physics model,
training data, resolution and device; optional `calccls.lmax` for the multipole
limit. Requires CuHPX/CUDA transforms.

| Additional argument | Meaning / default |
| --- | --- |
| `--output-path PATH` | Auto-spectrum HDF5 output; default `cls.h5`. |
| `--num-examples N` | Requested example limit; default `100`. |

**Output:** the requested HDF5 file containing `cls_0`, `cls_1`, etc., `labels`
and `inds`; a companion `<stem>_cross.h5` containing all auto/cross pairs; and
HTML dashboards alongside both files.

### 5. `calccorrs` — inspect two-point correlations (optional)

**Config:** `forward_model` and `training`: survey geometry,
`analysis.n_side`/`n_side_down`, physics model, records, batch size and workers.
The current implementation uses CUDA and Pyracorr.

| Additional argument | Meaning / default |
| --- | --- |
| `--output-dir PATH` | Destination directory; default `corrs`. |
| `--file-index N` | Output index and random-seed selector; default `0`. |
| `--num-batches-per-file N` | Maximum input batches written per output file; default `100`. |
| `--dataset-split {training,validation}` | Input split; default `training`. |

**Output:** `corrs_<split>_<index:06d>.h5` with per-batch `separations`,
`correlations`, `labels` and `inds` datasets, and an HTML dashboard. Despite the
CLI help's tar-shard terminology, this command writes HDF5. `--file-index` does
not select a single input tar shard.

### 6. `modelprofile` — inspect model cost (optional)

**Config:** `forward_model` and `training`: records/batching, resolutions,
`encoder_name` and `encoder_args`. The current profiler prepares inputs using
the linear on-the-fly physics model.
**Additional arguments:** none.
**Output:** model specifications, tensor shapes and timing/memory profiler
summaries, with scheduled traces under `torch_profiler_logs/`. It profiles up
to 20 forward batches without fitting or saving a checkpoint. Device selection
is automatic.

### 7. `train` — fit the neural model

**Config:** `forward_model` and `training`. Supply data and physics-model settings,
encoder/arguments, embedding dimension, loss/arguments, optimization, batching,
validation schedule, checkpointing and logging settings. Encoders include map
transformers and power-spectrum/correlation networks, whose dependencies and
device requirements differ.

| Additional argument | Meaning / default |
| --- | --- |
| `--resume-from-checkpoint PATH` | Restore model/optimizer and training progress. |
| `--checkpoint-dir PATH` | Override the base checkpoint directory. |
| `--max-steps N` | Limit steps in this invocation. |
| `--device DEVICE` | Override the training device, e.g. `cpu` or `cuda`. |
| `--wandb-mode {online,offline,disabled}` | Override W&B logging mode. |
| `--tag TEXT` | Run subdirectory/name; CLI default `test-run` overrides a configured tag. |

Other flags default to unset. Multi-process training uses `torchrun` environment
variables when `training.use_ddp` is enabled.
**Output:** loss/evaluation logs and optional W&B records;
`checkpoint_dir/tag/config.yaml`, `checkpoint-latest.pt` and periodic
`checkpoint-step-XXXXXX.pt`. Validation produces evaluation label/prediction
HDF5 files and diagnostic plots according to the configured schedule.

### 8. `predict` — generate validation summaries

**Config:** `forward_model` and `training`, with data, physics, encoder,
embedding and loss settings matching the checkpoint. `likelihood` is not required.

| Additional argument | Meaning / default |
| --- | --- |
| `--checkpoint PATH` | **Required:** checkpoint with model weights. |
| `--output-file PATH` | **Required:** destination HDF5 file. |
| `--batch-size N` | Override `training.batch_size`. |
| `--num-examples N` | Requested validation example count; default `1000`. |
| `--device DEVICE` | Evaluation device; CLI default `cuda`. Use `cpu` for compatible CPU routes. |

**Output:** HDF5 datasets `labels`, `predictions` and `indices`, plus
physics-model and parameter-name attributes. The command consumes the validation
split subject to the example limit; it does not necessarily process the full
validation set despite the short CLI help description.

### 9. `likelihood` — fit a density and optionally sample posteriors

**Config:** `likelihood`: model type (`mdn` or `cnffm`), `model_args`, epochs,
batch size, learning rate, validation fraction, seed and device. Adding
`mcmc_num_samples` enables posterior sampling/plots and also requires `training`
(physics model/arguments) and `forward_model` (parameters/priors). MCMC settings
select Metropolis–Hastings (`mh`) or Hamiltonian Monte Carlo (`hmc`), observations,
sample count, burn-in and proposal/integration settings.

| Additional argument | Meaning / default |
| --- | --- |
| `--input-file PATH` | **Required:** prediction HDF5 containing `labels` and `predictions`. |
| `--output-file PATH` | **Required:** trained likelihood model destination. |
| `--device DEVICE` | Override `likelihood.device`; unset by default. |

**Output:** saved likelihood model, fit diagnostic `<output-stem>.png`, and,
when MCMC is enabled, `<stem>_<sampler>_triangle_chain_<observation>.png` plots.
Input pairs are split into disjoint training/validation sets using
`likelihood.validation_fraction`, independently of the map-shard split.

## Development

```bash
uv run pytest
uv run ruff check .
```

The command reference describes current interfaces and intended outputs.
In the current source, `calccls` and `calccorrs` import `load_pixel_indices`,
which is absent from the configuration module; these routes need that repository
issue resolved before they can run.
