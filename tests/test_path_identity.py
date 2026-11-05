"""One lexical path identity (``utils.lexical_path_key``) for every comparison that guards
outputs: ``/x`` and ``//x`` (POSIX) must not let two sources share one artifact."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from neurodags.orchestrators import run_pipeline
from neurodags.outputs import check_output_collisions
from neurodags.utils import DatasetConfigError

pytestmark = pytest.mark.skipif(os.name == "nt", reason="POSIX leading-slash rule")

_DEFINITIONS = {
    "A": {
        "nodes": [
            {"id": 0, "derivative": "SourceFile"},
            {"id": 1, "node": "dummy", "args": {"param1": "id.0"}},
        ]
    }
}


def _dataset(root: str, out: str | None) -> dict:
    return {
        "name": "ds",
        "source_root": root,
        "file_pattern": root + "/*.txt",
        "derivatives_path": out,
    }


def _pipeline(datasets: dict) -> dict:
    return {
        "naming_version": 1,
        "datasets": datasets,
        "DerivativeDefinitions": _DEFINITIONS,
        "DerivativeList": ["A"],
    }


def _sources(tmp_path: Path) -> tuple[Path, Path]:
    a, b = tmp_path / "a", tmp_path / "b"
    for root in (a, b):
        root.mkdir()
        (root / "x.txt").write_text(root.name)
    return a, b


def test_shared_output_root_aliases_are_refused_before_writing(tmp_path: Path):
    a, b = _sources(tmp_path)
    out = tmp_path / "out"
    datasets = {"a": _dataset(str(a), str(out)), "b": _dataset(str(b), "/" + str(out))}
    with pytest.raises(DatasetConfigError, match="share derivatives_path"):
        run_pipeline(_pipeline(datasets))
    assert not out.exists()


def test_output_next_to_source_aliases_are_refused_before_writing(tmp_path: Path):
    # No derivatives_path: outputs go next to the sources, and /a and //a are one folder.
    a, _ = _sources(tmp_path)
    datasets = {"a": _dataset(str(a), None), "b": _dataset("/" + str(a), None)}
    with pytest.raises(DatasetConfigError, match="same outputs"):
        run_pipeline(_pipeline(datasets))
    assert sorted(p.name for p in a.iterdir()) == ["x.txt"]


def test_check_output_collisions_treats_leading_double_slash_as_root():
    with pytest.raises(DatasetConfigError, match="same outputs"):
        check_output_collisions([("d1: x.vhdr", "/o/x.vhdr"), ("d2: x.vhdr", "//o/x.vhdr")])
