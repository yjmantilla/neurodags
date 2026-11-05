"""Every pipeline / datasets YAML shown in the docs must be a valid 0.4.0 configuration.

The fenced ```yaml blocks of ``README.md`` and ``docs/**/*.md`` (the changelog excepted:
it quotes old versions on purpose) are extracted and classified:

- **pipeline**: a mapping with a top-level ``naming_version``, or with both
  ``DerivativeDefinitions`` and ``DerivativeList`` (a complete pipeline). Loaded with the
  real loader, :func:`neurodags.loaders.load_pipeline_configuration` (version check, name
  grammar, references, duplicate keys).
- **pipeline excerpt**: a mapping with ``DerivativeDefinitions`` only. Its names and
  references are checked with :func:`neurodags.loaders.check_pipeline_names`.
- **datasets**: a mapping whose values are all mappings and at least one has a
  ``file_pattern``. Every entry must set ``source_root`` and pass
  :class:`neurodags.definitions.DatasetConfig` (and the ``derivatives_subdir`` check).

Templates are skipped: blocks with ``<placeholder>`` tokens or ``...`` elision lines. The
YAML files in ``example_pipelines/`` are validated the same way.
"""

from __future__ import annotations

import io
import re
import textwrap
from pathlib import Path

import pytest
import yaml

from neurodags.definitions import DatasetConfig
from neurodags.loaders import (
    check_pipeline_names,
    load_configuration,
    load_pipeline_configuration,
)
from neurodags.utils import check_derivatives_subdir

ROOT = Path(__file__).resolve().parents[1]
DOC_FILES = [
    ROOT / "README.md",
    *sorted(
        p
        for p in (ROOT / "docs").rglob("*.md")
        if p.name != "changelog.md" and "_build" not in p.parts
    ),
]
EXAMPLES = ROOT / "example_pipelines"

FENCE_RE = re.compile(
    r"^(?P<indent>[ \t]*)```(?P<lang>[A-Za-z]*)[^\n]*\n(?P<body>.*?)^(?P=indent)```", re.M | re.S
)
PLACEHOLDER_RE = re.compile(r"<[A-Za-z_][A-Za-z0-9_ .-]*>")


def _yaml_blocks():
    for path in DOC_FILES:
        text = path.read_text(encoding="utf-8")
        for match in FENCE_RE.finditer(text):
            if match.group("lang").lower() not in {"yaml", "yml"}:
                continue
            body = textwrap.dedent(match.group("body"))
            line = text[: match.start()].count("\n") + 1
            yield pytest.param(body, id=f"{path.relative_to(ROOT)}:{line}")


def _is_template(body: str) -> bool:
    return bool(PLACEHOLDER_RE.search(body)) or any(
        ln.strip() in {"...", "…"} for ln in body.splitlines()
    )


def _classify(data) -> str | None:
    if not isinstance(data, dict):
        return None
    if "naming_version" in data or {"DerivativeDefinitions", "DerivativeList"} <= set(data):
        return "pipeline"
    if "DerivativeDefinitions" in data:
        return "pipeline-excerpt"
    values = list(data.values())
    if values and all(isinstance(v, dict) for v in values):
        if any("file_pattern" in v for v in values):
            return "datasets"
    return None


def _check_datasets(data: dict) -> None:
    for key, entry in data.items():
        assert entry.get("source_root") is not None, f"dataset {key!r} has no source_root"
        DatasetConfig(**entry)
        check_derivatives_subdir(entry.get("derivatives_subdir"))


def _check_block(body: str) -> str | None:
    data = yaml.safe_load(body)  # every non-template snippet must at least parse
    kind = _classify(data)
    if kind is not None:
        data = load_configuration(io.StringIO(body))  # also rejects duplicate keys
    if kind == "pipeline":
        config = load_pipeline_configuration(io.StringIO(body))
        datasets = config.get("datasets")
        if isinstance(datasets, dict):
            _check_datasets(datasets)
    elif kind == "pipeline-excerpt":
        check_pipeline_names(data)
    elif kind == "datasets":
        _check_datasets(data)
    return kind


@pytest.mark.parametrize("body", list(_yaml_blocks()))
def test_docs_yaml_snippet_is_valid(body):
    if _is_template(body):
        pytest.skip("template / excerpt with placeholders")
    _check_block(body)


def test_docs_yaml_snippets_cover_full_configs():
    """Guard the extractor itself: it must find the full examples it is meant to check."""
    kinds: dict[str, int] = {}
    for param in _yaml_blocks():
        body = param.values[0]
        if _is_template(body):
            continue
        kind = _classify(yaml.safe_load(body))
        kinds[kind] = kinds.get(kind, 0) + 1
    assert kinds.get("pipeline", 0) >= 5
    assert kinds.get("datasets", 0) >= 5


@pytest.mark.parametrize("path", sorted(EXAMPLES.glob("pipeline_*.yml")), ids=lambda p: p.name)
def test_example_pipeline_is_valid(path):
    config = load_pipeline_configuration(str(path))
    datasets = config["datasets"]
    assert isinstance(datasets, str)
    _check_datasets(load_configuration(str(path.parent / datasets)))


@pytest.mark.parametrize("path", sorted(EXAMPLES.glob("datasets_*.yml")), ids=lambda p: p.name)
def test_example_datasets_is_valid(path):
    _check_datasets(load_configuration(str(path)))
