# Installation

## Stable Release

With [uv](https://docs.astral.sh/uv/) (recommended):

```bash
uv add neurodags
```

With pip:

```bash
pip install neurodags
```

## Development Install

```bash
git clone https://github.com/yjmantilla/neurodags
cd neurodags
uv sync --all-extras    # creates .venv and installs all deps incl. dev/test/docs
uv run pre-commit install
```

> **No uv?** `pip install uv` or `curl -Ls https://astral.sh/uv/install.sh | sh`.  
> Without uv: `python -m venv .venv && source .venv/bin/activate && pip install -e .[dev,test,docs] && pre-commit install`

## Running Commands

All development commands use `uv run` so you never need to activate the environment:

```bash
uv run ruff check src/              # lint
uv run ruff check src/ --fix        # lint + autofix
uv run black --check .              # format check
uv run black .                      # format
uv run pytest -q                    # run tests
uv run pytest -s -q --no-cov --pdb  # debug a test

uv run sphinx-build -b html docs docs/_build/html -W --keep-going  # build docs
rm -rf docs/_build                                                   # clean docs
```

## HDF5 / NetCDF Note

If you encounter `RuntimeError: NetCDF: HDF error` when writing `.nc` files:

```bash
uv run pip install --no-binary=h5py h5py
```

## Requirements

- Python >= 3.10
- Core dependencies: `mne`, `xarray`, `netCDF4`, `pydantic`, `networkx`, `joblib`, `plotly`, `dash`, `structlog`, `pyyaml`, `fooof`, `antropy`, `neurokit2`
