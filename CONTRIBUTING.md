# Contributing

Thanks for your interest in contributing! This document describes how to set up your environment and make changes.

## Setup

1. Fork and clone the repo
2. Create a virtual environment and install dev dependencies:

```bash
python -m venv .venv
.venv\\Scripts\\activate  # Windows
# source .venv/bin/activate  # macOS/Linux
pip install -U pip
pip install -e .[dev,test,docs]
pre-commit install
```

## Development workflow

- Run linters and type checks: `ruff check . && black --check . && mypy src`
- Run tests: `pytest -q`
- Format code: `black .`
- Fix lint issues: `ruff check --fix .`

## Docs

- Build docs locally: `sphinx-build -b html docs docs/_build/html -W --keep-going`
- Examples for the gallery live in `docs/examples` and must start with `plot_*.py`.

## Commit style

- Use clear, imperative commit messages
- Keep changes small and focused

## Releases

- Bump version in `src/neurodags/_version.py` and `pyproject.toml`
- Create a tag `vX.Y.Z`
- Publish to PyPI (example):

```bash
python -m build
python -m twine upload dist/*
```

## Code of Conduct

Be kind and respectful. We follow the [Contributor Covenant](https://www.contributor-covenant.org/).
