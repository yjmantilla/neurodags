"""Glob metacharacters in source/derivative paths (dataframe collisions: test_dataframe_columns)."""

from __future__ import annotations

import glob
import json
import shutil

import yaml

from neurodags.cli import main
from neurodags.mermaid import pipeline_to_mermaid
from neurodags.orchestrators import run_pipeline
from neurodags.outputs import build_manifest, manifest_artifacts, unit_state, write_manifest

# ---------------------------------------------------------------------------
# Cache lookup with glob metacharacters (manifest-based, spec 6.2)
# ---------------------------------------------------------------------------


def _done(prefix: str, name: str, suffixes=(".nc",)) -> dict:
    paths = {}
    for suffix in suffixes:
        with open(prefix + suffix, "w") as fh:
            fh.write("x")
        paths[suffix] = [prefix + suffix]
    write_manifest(prefix, build_manifest(prefix, name, dataset="d", rel="x", artifacts=paths))
    return {k: list(v) for k, v in paths.items()}


def test_cached_artifact_found_under_bracketed_parent_dir(tmp_path):
    study = tmp_path / "study [2024]"
    study.mkdir()
    prefix = str(study / "sub-01.vhdr@BandPower")
    expected = _done(prefix, "BandPower")

    assert manifest_artifacts(prefix, "BandPower") == expected
    assert unit_state(prefix, "BandPower").state == "done"


def test_cached_artifact_found_for_bracketed_source_name(tmp_path):
    prefix = str(tmp_path / "a[1]*?.fif@Fit")
    expected = _done(prefix, "Fit")

    assert manifest_artifacts(prefix, "Fit") == expected


def test_bracketed_source_does_not_match_other_source_outputs(tmp_path):
    _done(str(tmp_path / "a1.fif@Fit"), "Fit")

    assert manifest_artifacts(str(tmp_path / "a[1].fif@Fit"), "Fit") is None
    assert unit_state(str(tmp_path / "a[1].fif@Fit"), "Fit").state == "missing"


def test_unit_state_ignores_directories(tmp_path):
    (tmp_path / "sub-01.vhdr@Fit.d").mkdir()

    assert unit_state(str(tmp_path / "sub-01.vhdr@Fit"), "Fit").state == "missing"


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
