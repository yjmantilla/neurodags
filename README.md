# NeuroDAGs

**An Extensible and Declarative DAG Framework for Reproducible Neuroscience Workflows**

M/EEG studies generate many interdependent intermediate derivatives. Recomputing full pipelines is wasteful; reusing valid intermediates is non-trivial. Large-scale studies require reproducible, extensible, and efficient workflows. NeuroDAGs addresses this with a declarative, graph-based framework for scalable and reusable derivative computation.

**[Conference Poster](https://canva.link/c2b2fm0mk2wotwq)**

## Core Idea

Pipelines are defined as a **directed acyclic graph (DAG)** of computation nodes that output reusable derivatives, executed for each input file.

## Design Principles

- **Reproducible, transparent workflows** defined declaratively in YAML — version-controllable and LLM-friendly.
- **Uniform node abstraction** — preprocessing, features, and any custom nodes are treated identically.
- **Directory-agnostic** — outputs mirror inputs' organization. Derivatives are labeled with a `@DerivativeName` suffix.
- **xarray-centered outputs** — derivatives stored as language-agnostic, metadata-rich, dimension-aware xarray → NetCDF.
- **Graph-based reuse** — if a derivative is already computed and `overwrite=False`, it is skipped automatically.

## Features

- Agnostic to data organization / directory hierarchy
- SLURM / HPC friendly with file-level parallelism via joblib
- Graph-based caching: skip already-computed derivatives
- Extensible node system — add nodes without forking the package
- YAML-based declarative configuration
- Built-in nodes for preprocessing, spectral analysis, entropy, complexity, and data transformations
- Dataframe assembly (wide or long format) from derivative artifacts
- Dry-run mode — inspect planned computations without executing
- Built-in Dash-Plotly explorer for `.fif` and `.nc` files

## Installation

```bash
pip install neurodags
```

With [uv](https://docs.astral.sh/uv/) (recommended):

```bash
uv add neurodags
```

## Quickstart

Run a full synthetic pipeline — no real data required:


```bash
uv run python docs/examples/plot_quickstart_synthetic.py
```

The quickstart generates 1/f-noise EEG, runs preprocessing → spectral analysis → band power, builds a dataframe, and plots relative power per subject.

## Development

```bash
git clone https://github.com/yjmantilla/neurodags
cd neurodags
uv sync --all-extras    # creates .venv and installs all deps incl. dev/test/docs
uv run pre-commit install
```

Key commands (all via `uv run`):

```bash
uv run ruff check src/              # lint  (fix: uv run ruff check src/ --fix)
uv run black --check .              # format check  (fix: uv run black .)
uv run pytest -q                    # run tests
uv run pytest -s -q --no-cov --pdb  # debug a failing test

uv run sphinx-build -b html docs docs/_build/html -W --keep-going  # build docs
rm -rf docs/_build                                                   # clean docs
```

> **No uv?** Install it with `pip install uv` or `curl -Ls https://astral.sh/uv/install.sh | sh`.
> All commands above work with plain `python`/`pip` too — swap `uv run` → activate `.venv`, `uv sync` → `pip install -e .[dev,test,docs]`.

## Project Structure

```
my_project/
├── datasets.yml      # Dataset sources and paths
├── pipeline.yml      # Derivative definitions and execution list
└── custom_nodes.py   # Optional custom node definitions
```

## Quick Example

**`datasets.yml`**
```yaml
my_dataset:
  name: MyDataset
  file_pattern:
    local: data/**/*.vhdr
    hpc: /cluster/BIDS/**/*.vhdr
  derivatives_path:
    local: outputs/
    hpc: /cluster/scratch/out
```

**`pipeline.yml`**
```yaml
datasets: datasets.yml
mount_point: local
new_definitions: custom_nodes.py  # optional

DerivativeDefinitions:
  CleanedEEG:
    nodes:
      - id: 0
        derivative: SourceFile
      - id: 1
        node: basic_preprocessing
        args:
          mne_object: id.0
          resample: 256
          filter_args:
            l_freq: 0.5
            h_freq: 110

  PowerSpectrum:
    for_dataframe: True
    nodes:
      - id: 0
        derivative: CleanedEEG.fif
      - id: 1
        node: mne_spectrum_array
        args:
          meeg: id.0
          method: multitaper

DerivativeList:
  - CleanedEEG
  - PowerSpectrum
```

**Python**
```python
from neurodags.loaders import load_configuration
from neurodags.orchestrators import iterate_derivative_pipeline

config = load_configuration("pipeline.yml")
iterate_derivative_pipeline(config, "CleanedEEG")
iterate_derivative_pipeline(config, "PowerSpectrum")
```

## Custom Nodes

Add nodes without modifying or forking the package:

```python
# custom_nodes.py
from neurodags.nodes import register_node
from neurodags.definitions import Artifact, NodeResult

@register_node
def my_node(data) -> NodeResult:
    result = compute(data)
    return NodeResult(
        artifacts={
            ".nc": Artifact(
                item=result,
                writer=lambda path: result.to_netcdf(path),
            ),
        },
    )
```

Key rules:
1. A node is a function decorated with `@register_node`.
2. It returns a `NodeResult`.
3. A `NodeResult` contains `artifacts` — a dict mapping file extension to `Artifact(item, writer)`.

## Dataframe Assembly

```python
from neurodags.orchestrators import build_derivative_dataframe

df = build_derivative_dataframe("pipeline.yml", output_format="wide")
```

Derivatives marked `for_dataframe: True` are collected automatically. Supports `"wide"` (one row per file) and `"long"` (one row per value) formats.

## Parallel Execution

```yaml
# pipeline.yml
n_jobs: 4           # -1 = all cores, 1 or null = serial
joblib_backend: loky
joblib_prefer: processes
```

Or via Python:

```python
iterate_derivative_pipeline(config, "MyDerivative", n_jobs=4)
```

## Visualization

```bash
python -m neurodags.visualization path/to/file.fif
python -m neurodags.visualization path/to/file.nc
```

Built-in Dash-Plotly explorer with dimension-aware UI — dropdown per axis, plot types: Line, Scatter, Bar, Heatmap.

## Inspection (Dry Run)

```python
iterate_derivative_pipeline(config, "MyDerivative", dry_run=True)
```

Returns a dataframe describing the execution plan without running any nodes. `.error` marker files prevent silent retry of failed runs.

## Derivative Flags

| Flag | Default | Description |
|------|---------|-------------|
| `save` | `True` | Persist artifacts to disk. `False` = compute but don't write. |
| `overwrite` | `False` | Force recompute even if output exists. |
| `for_dataframe` | `False` | Include this derivative in `build_derivative_dataframe`. |

## Custom Node Definitions

Point `new_definitions` to one or more Python files:

```yaml
new_definitions:
  - custom_nodes/my_nodes.py
  - /abs/path/to/other_nodes.py
```

Relative paths are resolved from the pipeline YAML location.

## Documentation

[https://yjmantilla.github.io/neurodags/](https://yjmantilla.github.io/neurodags/)

## HDF5 / NetCDF Note

If you encounter `RuntimeError: NetCDF: HDF error`:

```bash
uv run pip install --no-binary=h5py h5py
# or without uv:
pip install --no-binary=h5py h5py
```

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md).

## License

MIT. See [`LICENSE`](LICENSE).
