"""Glob metacharacters in source/derivative paths, and dataframe column collisions."""

from __future__ import annotations

import glob
import json
import shutil
from pathlib import Path
from unittest.mock import patch

import numpy as np
import xarray as xr
import yaml

from neurodags.cli import main
from neurodags.dag import _artifact_candidates_for, collect_derivative_for_dataframe
from neurodags.mermaid import pipeline_to_mermaid
from neurodags.orchestrators import _collect_dataframe_file, run_pipeline

# ---------------------------------------------------------------------------
# Cache lookup with glob metacharacters
# ---------------------------------------------------------------------------


def test_cached_artifact_found_under_bracketed_parent_dir(tmp_path):
    study = tmp_path / "study [2024]"
    study.mkdir()
    artifact = study / "sub-01.vhdr@BandPower.nc"
    artifact.write_text("x")

    assert _artifact_candidates_for((study / "sub-01.vhdr@BandPower").as_posix()) == [
        artifact.as_posix()
    ]


def test_cached_artifact_found_for_bracketed_source_name(tmp_path):
    artifact = tmp_path / "a[1]*?.fif@Fit.nc"
    artifact.write_text("x")
    (tmp_path / "a[1]*?.fif@Fit.error").write_text("boom")

    assert _artifact_candidates_for((tmp_path / "a[1]*?.fif@Fit").as_posix()) == [
        artifact.as_posix()
    ]


def test_bracketed_source_does_not_match_other_source_outputs(tmp_path):
    (tmp_path / "a1.fif@Fit.nc").write_text("x")

    assert _artifact_candidates_for((tmp_path / "a[1].fif@Fit").as_posix()) == []


def test_artifact_candidates_ignore_directories(tmp_path):
    (tmp_path / "sub-01.vhdr@Fit.d").mkdir()
    artifact = tmp_path / "sub-01.vhdr@Fit.nc"
    artifact.write_text("x")

    assert _artifact_candidates_for((tmp_path / "sub-01.vhdr@Fit").as_posix()) == [
        artifact.as_posix()
    ]


def test_pipeline_under_bracketed_dir_is_cached_and_reported_done(dummy_pipeline, tmp_path, capsys):
    study = tmp_path / "study [2024]"
    shutil.copytree(dummy_pipeline["data_dir"], study / "rawdata")
    out_dir = study / "derivatives"
    cfg = {
        **dummy_pipeline["config"],
        "DerivativeList": ["BasicPrep"],
        "datasets": {
            "test_dataset": {
                "name": "TestDataset",
                # source_root is a literal path (no glob), so it is not escaped.
                "source_root": (study / "rawdata").as_posix(),
                # file_pattern is a user glob, so the literal directory is escaped there.
                "file_pattern": glob.escape((study / "rawdata").as_posix()) + "/**/*.vhdr",
                "derivatives_path": out_dir.as_posix(),
            }
        },
    }
    config_path = tmp_path / "pipeline.yml"
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    run_pipeline(str(config_path), derivatives=["BasicPrep"], raise_on_error=True)
    outputs = sorted(out_dir.rglob("*@BasicPrep.fif"))
    assert len(outputs) == 2
    mtimes = {p: p.stat().st_mtime_ns for p in outputs}

    # Second run: everything is cached, nothing is rewritten.
    run_pipeline(str(config_path), derivatives=["BasicPrep"], raise_on_error=True)
    assert {p: p.stat().st_mtime_ns for p in outputs} == mtimes

    capsys.readouterr()
    rc = main(["status", str(config_path), "--format", "json"])
    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload["complete"] is True


# ---------------------------------------------------------------------------
# Dataframe column collisions
# ---------------------------------------------------------------------------


def test_flatten_warns_on_coordinate_encoding_collision():
    da = xr.DataArray(np.array([1.0, 2.0]), dims=["ch"], coords={"ch": ["a-b", "a~b"]})

    with (
        patch("neurodags.dag.log") as log,
        patch("neurodags.dag._resolve_derivative_artifacts", return_value={".nc": da}),
    ):
        row = collect_derivative_for_dataframe(
            {"save": False}, "Power", "input.vhdr", reference_base=Path("unused")
        )

    # Existing output is unchanged: one column, last value wins.
    assert row == {"Power.nc@ch-a~b": 2.0}
    collisions = [c for c in log.warning.call_args_list if "collision" in c.args[0]]
    assert len(collisions) == 1
    entry = collisions[0].kwargs
    assert entry["derivative"] == "Power"
    assert entry["suffix"] == ".nc"
    assert entry["column_suffix"] == "ch-a~b"
    assert [str(v) for _, v in entry["overwritten_coords"]] == ["a-b"]
    assert [str(v) for _, v in entry["kept_coords"]] == ["a~b"]


def test_flatten_without_collision_does_not_warn():
    da = xr.DataArray(np.array([1.0, 2.0]), dims=["ch"], coords={"ch": ["a", "b"]})
    with (
        patch("neurodags.dag.log") as log,
        patch("neurodags.dag._resolve_derivative_artifacts", return_value={".nc": da}),
    ):
        row = collect_derivative_for_dataframe(
            {"save": False}, "Power", "input.vhdr", reference_base=Path("unused")
        )
    assert row == {"Power.nc@ch-a": 1.0, "Power.nc@ch-b": 2.0}
    assert not log.warning.called


def test_row_merge_warns_when_two_derivatives_produce_same_column(tmp_path):
    values = {"A": {"Shared.nc": 1.0, "A.nc": 1.0}, "B": {"Shared.nc": 2.0}}

    def fake_collect(derivative_def, derivative_name, *args, **kwargs):
        return values[derivative_name]

    with (
        patch("neurodags.orchestrators.log") as log,
        patch("neurodags.orchestrators.collect_derivative_for_dataframe", fake_collect),
    ):
        row, _ = _collect_dataframe_file(
            index=0,
            dataset_name="ds",
            file_path=str(tmp_path / "x.vhdr"),
            dataset_config_dict={"name": "ds", "file_pattern": str(tmp_path / "*.vhdr")},
            source_path=str(tmp_path / "x.vhdr"),
            common_root=str(tmp_path),
            mount_point=None,
            selected_derivatives=["A", "B"],
            derivative_definitions={"A": {}, "B": {}},
            preserve_complex_values=False,
            collect_long_rows=False,
            raise_on_error=True,
            n_files=1,
        )

    assert row["Shared.nc"] == 2.0  # behaviour unchanged: later derivative wins
    collisions = [c for c in log.warning.call_args_list if "collision" in c.args[0]]
    assert len(collisions) == 1
    assert collisions[0].kwargs == {"columns": ["Shared.nc"], "previous_owners": ["A"]}


# ---------------------------------------------------------------------------
# Mermaid node ids
# ---------------------------------------------------------------------------


def test_pipeline_mermaid_sanitizes_node_ids():
    cfg = {
        "DerivativeDefinitions": {
            "psd-welch": {"nodes": [{"id": 0, "derivative": "SourceFile"}]},
            "psd_welch": {"nodes": [{"id": 0, "derivative": "SourceFile"}]},
            "Bands": {"nodes": [{"id": 0, "derivative": "psd-welch.nc"}]},
        }
    }
    result = pipeline_to_mermaid(cfg)
    assert 'psd_welch_2["psd-welch"]' in result
    assert 'psd_welch["psd_welch"]' in result
    assert 'Bands["Bands"]' in result
    assert "psd_welch_2 --> Bands" in result
    assert "psd-welch -->" not in result
