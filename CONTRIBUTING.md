# Contributing

## Setup

Fork and clone the repo, then let uv handle the environment:

```bash
git clone https://github.com/yjmantilla/neurodags
cd neurodags
uv sync --all-extras        # creates .venv, installs dev + test + docs deps
uv run pre-commit install
```

> **No uv?** `pip install uv` or see [uv docs](https://docs.astral.sh/uv/).  
> Without uv: `python -m venv .venv && source .venv/bin/activate && pip install -e .[dev,test,docs] && pre-commit install`

## Development Workflow

```bash
uv run ruff check src/              # lint
uv run ruff check src/ --fix        # lint + autofix
uv run black --check .              # format check
uv run black .                      # format
uv run mypy src                     # type check
uv run pytest -q                    # run tests
uv run pytest -s -q --no-cov --pdb  # debug a test
```

## Docs

Examples in `docs/examples/` must be named `plot_*.py` (sphinx-gallery convention).

```bash
uv run sphinx-build -b html docs docs/_build/html -W --keep-going  # build
rm -rf docs/_build                                                   # clean
```

## Commit Style

- Imperative, concise commit messages
- Keep changes small and focused

## Releases

Version is derived automatically from git tags via `hatch-vcs` — no manual version bump needed.

```bash
git tag vX.Y.Z
git push origin vX.Y.Z

uv run python -m build
uv run twine upload dist/*
```

## Code of Conduct

Be kind and respectful. We follow the [Contributor Covenant](https://www.contributor-covenant.org/).
