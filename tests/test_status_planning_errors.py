"""``status`` / ``dry-run`` must fail when a unit could not be planned.

A per-file planning exception used to be logged and dropped from the dry-run dataframe, so
``status`` reported the remaining (possibly empty) population as ``complete`` with exit 0.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from neurodags.cli import main
from neurodags.tui import _status_sync


def _pipeline(tmp_path: Path, *, arg: str = "id.99", n_files: int = 1) -> Path:
    raw = tmp_path / "raw"
    raw.mkdir()
    for i in range(n_files):
        (raw / f"x{i}.txt").write_text("")
    cfg = {
        "naming_version": 1,
        "datasets": {
            "d": {
                "name": "d",
                "source_root": str(raw),
                "file_pattern": str(raw / "*.txt"),
                "derivatives_path": str(tmp_path / "out"),
            }
        },
        "DerivativeDefinitions": {
            # ``id.99`` names a node that does not exist (only 0 and 1 do).
            "A": {
                "nodes": [
                    {"id": 0, "derivative": "SourceFile"},
                    {"id": 1, "node": "dummy", "args": {"param1": arg}},
                ]
            },
        },
        "DerivativeList": ["A"],
    }
    pipe = tmp_path / "pipeline.yml"
    pipe.write_text(yaml.safe_dump(cfg))
    return pipe


def _status(pipe: Path, capsys, *args: str) -> tuple[int, str, str]:
    rc = main(["--log-level", "CRITICAL", "status", str(pipe), *args])
    out, err = capsys.readouterr()
    return rc, out, err


def test_status_json_with_planning_error_is_not_complete(tmp_path, capsys):
    pipe = _pipeline(tmp_path, n_files=2)
    assert main(["count-inputs", str(pipe)]) == 0
    assert capsys.readouterr().out.strip() == "2"
    rc, out, err = _status(pipe, capsys, "--format", "json", "--include-files")
    report = json.loads(out)
    assert rc == 1
    assert report["complete"] is False
    assert report["n_files"] == 2
    assert report["planning_error_indices"] == [0, 1]
    assert report["grand_total"]["planning-error"] == 2
    assert {f["status"] for f in report["files"]} == {"planning-error"}
    assert all("StopIteration" in f["planning_error"] for f in report["files"])
    assert "could not be planned" in err


def test_status_table_and_indices_fail_on_planning_error(tmp_path, capsys):
    pipe = _pipeline(tmp_path)
    rc, out, _ = _status(pipe, capsys)
    assert rc == 1
    assert "could not be planned" in out
    assert "No files found" not in out
    rc, out, err = _status(pipe, capsys, "--format", "indices")
    assert rc == 1
    assert out.strip() == ""
    assert "could not be planned" in err


def test_status_exit_zero_still_reports_planning_error(tmp_path, capsys):
    pipe = _pipeline(tmp_path)
    rc, out, err = _status(pipe, capsys, "--format", "json", "--exit-zero")
    assert rc == 0
    assert json.loads(out)["complete"] is False
    assert "could not be planned" in err


def test_dry_run_fails_on_planning_error_and_keeps_the_unit(tmp_path, capsys):
    pipe = _pipeline(tmp_path)
    plan_path = tmp_path / "plan.csv"
    rc = main(["--log-level", "CRITICAL", "dry-run", str(pipe), "--output", str(plan_path)])
    err = capsys.readouterr().err
    assert rc == 1
    assert "could not be planned" in err
    plan = pd.read_csv(plan_path)
    assert list(plan["index"]) == [0]
    assert plan["planning_error"].str.contains("StopIteration").all()


def test_valid_pipeline_and_empty_discovery_still_succeed(tmp_path, capsys):
    pipe = _pipeline(tmp_path, arg="id.0")
    rc, out, _ = _status(pipe, capsys, "--format", "json")
    report = json.loads(out)
    assert rc == 1  # nothing computed yet: missing, but every unit was planned
    assert report["planning_error_indices"] == []
    assert report["grand_total"]["missing"] == 1
    assert main(["--log-level", "CRITICAL", "dry-run", str(pipe), "--output", "p.csv"]) == 0
    capsys.readouterr()

    for f in (tmp_path / "raw").iterdir():
        f.unlink()
    rc, out, _ = _status(pipe, capsys, "--format", "json")
    assert rc == 0
    assert json.loads(out)["complete"] is True


def test_tui_status_counts_planning_errors_as_errored(tmp_path):
    pipe = _pipeline(tmp_path)
    rows, grand, errored = _status_sync(str(pipe), None, None)
    assert grand["total"] == 1
    assert grand["errored"] == 1
    assert errored and "planning error" in errored[0][2]


@pytest.fixture(autouse=True)
def _cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
