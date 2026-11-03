"""``neurodags run`` exits 1 when a unit failed (release review 2f4894c), after processing
every requested input; failures reach API callers through ``failures=``."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from neurodags.cli import main
from neurodags.definitions import Artifact, NodeResult
from neurodags.nodes import register_node
from neurodags.orchestrators import run_pipeline


@register_node(name="_test_fails_on_bad", override=True)
def _fails_on_bad(path):
    if "bad" in Path(path).name:
        raise ValueError("bad input")
    return NodeResult({".txt": Artifact("ok", lambda p: Path(p).write_text("ok"))})


def _pipeline(tmp_path: Path, names: list[str]) -> Path:
    raw = tmp_path / "raw"
    raw.mkdir()
    for name in names:
        (raw / name).write_text(name)
    config = {
        "naming_version": 1,
        "datasets": {
            "ds": {
                "name": "ds",
                "source_root": "raw",
                "file_pattern": "raw/*.dat",
                "derivatives_path": "out",
            }
        },
        "DerivativeDefinitions": {
            "Check": {
                "nodes": [
                    {"id": 0, "derivative": "SourceFile"},
                    {"id": 1, "node": "_test_fails_on_bad", "args": {"path": "id.0"}},
                ]
            }
        },
        "DerivativeList": ["Check"],
    }
    path = tmp_path / "pipeline.yml"
    path.write_text(yaml.safe_dump(config))
    return path


@pytest.mark.parametrize(
    ("names", "failed"),
    [(["a.dat", "b.dat"], 0), (["a.dat", "bad.dat"], 1), (["bad1.dat", "bad2.dat"], 2)],
)
def test_run_exit_code_reports_failed_units(tmp_path, capsys, names, failed):
    pipe = _pipeline(tmp_path, names)
    assert main(["run", str(pipe)]) == (1 if failed else 0)
    err = capsys.readouterr().err
    if failed:
        assert f"neurodags: error: {failed} unit(s) failed (Check #" in err
    # Every input was processed: the good ones are written, the bad ones have .error.
    out = tmp_path / "out"
    assert len(list(out.glob("*@Check.done"))) == len(names) - failed
    assert len(list(out.glob("*@Check.error"))) == failed


def test_dry_run_is_not_a_failure(tmp_path):
    pipe = _pipeline(tmp_path, ["bad.dat"])
    assert main(["dry-run", str(pipe), "--output", str(tmp_path / "plan.csv")]) == 0


def test_run_pipeline_reports_failures(tmp_path):
    pipe = _pipeline(tmp_path, ["a.dat", "bad.dat"])
    failures: list[dict] = []
    run_pipeline(str(pipe), failures=failures)
    assert [(f["index"], f["derivative"], f["error_type"]) for f in failures] == [
        (1, "Check", "ValueError")
    ]
