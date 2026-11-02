"""Choosing the mount point outside the pipeline YAML (#66): -m/--mount-point,
NEURODAGS_MOUNT_POINT and the ``mount_point=`` API argument."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pandas as pd
import pytest
import yaml

from neurodags.cli import build_parser, main
from neurodags.datasets import (
    MOUNT_POINT_ENV,
    get_datasets_and_mount_point_from_pipeline_configuration,
    resolve_mount_point,
)
from neurodags.iterators import enumerate_pipeline_inputs
from neurodags.orchestrators import run_pipeline

# The lab tree has 2 recordings, the hpc tree 3, so the count shows which mount was used.
_FILES = {"lab": ["sub-01/a.vhdr", "sub-02/a.vhdr"], "hpc": ["s1/a.vhdr", "s2/a.vhdr", "s3/a.vhdr"]}


def _config(tmp_path: Path, mount_point: str | None = "lab") -> dict:
    roots = {}
    for mount, names in _FILES.items():
        root = tmp_path / mount / "raw"
        for name in names:
            (root / name).parent.mkdir(parents=True, exist_ok=True)
            (root / name).touch()
        roots[mount] = root
    return {
        "naming_version": 1,
        "mount_point": mount_point,
        "datasets": {
            "ds": {
                "name": "ds",
                "source_root": {m: str(r) for m, r in roots.items()},
                "file_pattern": {m: str(r / "**" / "*.vhdr") for m, r in roots.items()},
                "derivatives_path": {m: str(tmp_path / m / "derivatives") for m in roots},
            }
        },
        "DerivativeDefinitions": {
            "Prep": {
                "nodes": [
                    {"id": 0, "derivative": "SourceFile"},
                    {"id": 1, "node": "basic_preprocessing", "args": {"mne_object": "id.0"}},
                ]
            }
        },
        "DerivativeList": ["Prep"],
    }


@pytest.fixture
def pipeline_yml(tmp_path: Path) -> Path:
    path = tmp_path / "pipeline.yml"
    path.write_text(yaml.safe_dump(_config(tmp_path)))
    return path


# ---------------------------------------------------------------------------
# Precedence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("configured", "override", "env", "expected"),
    [
        ("lab", None, None, ("lab", "pipeline")),
        ("lab", None, "hpc", ("hpc", "environment")),
        ("lab", "hpc", "other", ("hpc", "argument")),
        (None, None, "hpc", ("hpc", "environment")),
        ("lab", None, "", ("lab", "pipeline")),  # empty variable counts as unset
        (None, None, None, (None, "unset")),
    ],
)
def test_resolve_mount_point_precedence(monkeypatch, configured, override, env, expected):
    if env is not None:
        monkeypatch.setenv(MOUNT_POINT_ENV, env)
    assert resolve_mount_point(configured, override) == expected


def test_loader_applies_override_and_environment(pipeline_yml, monkeypatch):
    load = get_datasets_and_mount_point_from_pipeline_configuration
    assert load(pipeline_yml)[1] == "lab"
    assert load(pipeline_yml, mount_point="hpc")[1] == "hpc"
    monkeypatch.setenv(MOUNT_POINT_ENV, "hpc")
    assert load(pipeline_yml)[1] == "hpc"
    assert load(pipeline_yml, mount_point="lab")[1] == "lab"


def test_mount_map_without_any_mount_names_every_way_to_set_one(tmp_path):
    config = _config(tmp_path, mount_point=None)
    with pytest.raises(ValueError, match="--mount-point.*NEURODAGS_MOUNT_POINT"):
        enumerate_pipeline_inputs(config)


def test_unknown_mount_is_an_error(pipeline_yml):
    with pytest.raises(Exception, match="nowhere"):
        enumerate_pipeline_inputs(str(pipeline_yml), mount_point="nowhere")


# ---------------------------------------------------------------------------
# Python API
# ---------------------------------------------------------------------------


def test_run_pipeline_mount_point_with_dict_config(pipeline_yml, monkeypatch, tmp_path):
    """The gica use case: a dict config, another host's mount, no edits to the dict."""
    config = yaml.safe_load(pipeline_yml.read_text())
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    plan = run_pipeline(config, mount_point="hpc", base_dir=pipeline_yml.parent, dry_run=True)
    assert sorted(plan["file_path"]) == sorted(
        str(tmp_path / "hpc" / "raw" / name) for name in _FILES["hpc"]
    )
    assert config["mount_point"] == "lab"  # the caller's dict is not modified


def test_run_snapshot_records_mount_and_source(pipeline_yml, monkeypatch, tmp_path):
    from neurodags import orchestrators

    calls = []
    monkeypatch.setattr(orchestrators, "iterate_derivative_pipeline", lambda **kw: calls.append(kw))
    monkeypatch.setenv(MOUNT_POINT_ENV, "hpc")
    run_pipeline(str(pipeline_yml))
    env = json.loads((tmp_path / "hpc" / "derivatives" / "code" / "neurodags_env.json").read_text())
    assert env["mount_point"] == "hpc"
    assert env["mount_point_source"] == "environment"
    assert not (tmp_path / "lab" / "derivatives" / "code").exists()
    assert calls


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("argv", "env", "expected"),
    [
        ([], None, 2),
        (["-m", "hpc"], None, 3),
        (["--mount-point", "hpc"], None, 3),
        ([], "hpc", 3),
        (["-m", "lab"], "hpc", 2),  # the flag wins over the environment
    ],
)
def test_count_inputs_mount_point(pipeline_yml, capsys, monkeypatch, argv, env, expected):
    if env is not None:
        monkeypatch.setenv(MOUNT_POINT_ENV, env)
    assert main(["count-inputs", str(pipeline_yml), *argv]) == 0
    assert capsys.readouterr().out.strip() == str(expected)


def test_inputs_mount_point(pipeline_yml, capsys):
    assert main(["inputs", str(pipeline_yml), "-m", "hpc"]) == 0
    out = capsys.readouterr().out
    assert "s3/a.vhdr" in out
    assert "sub-01" not in out


@pytest.mark.parametrize(
    ("argv", "env", "expected"),
    [
        ([], None, "mount_point: lab (from pipeline)"),
        (["-m", "hpc"], None, "mount_point: hpc (from --mount-point)"),
        ([], "hpc", f"mount_point: hpc (from ${MOUNT_POINT_ENV})"),
    ],
)
def test_validate_reports_mount_source(pipeline_yml, capsys, monkeypatch, argv, env, expected):
    if env is not None:
        monkeypatch.setenv(MOUNT_POINT_ENV, env)
    assert main(["validate", str(pipeline_yml), *argv]) == 0
    out = capsys.readouterr().out
    assert expected in out
    mount = expected.split()[1]
    assert f"source_root: {pipeline_yml.parent / mount / 'raw'}" in out


def test_status_reports_mount(pipeline_yml, capsys):
    main(["status", str(pipeline_yml), "-m", "hpc", "--format", "json"])
    data = json.loads(capsys.readouterr().out)
    assert data["mount_point"] == "hpc"
    assert data["mount_point_source"] == "--mount-point"
    assert data["n_files"] == 3

    main(["status", str(pipeline_yml), "-m", "hpc"])
    assert "mount:  hpc (from --mount-point)" in capsys.readouterr().out


def test_run_and_dataframe_pass_mount_point(pipeline_yml, monkeypatch):
    from neurodags import cli

    seen = {}

    def fake_dataframe(**kwargs):
        seen["df"] = kwargs
        return pd.DataFrame()

    monkeypatch.setattr(cli, "run_pipeline", lambda **kw: seen.setdefault("run", kw))
    monkeypatch.setattr(cli, "build_derivative_dataframe", fake_dataframe)
    assert main(["run", str(pipeline_yml), "-m", "hpc"]) == 0
    assert seen["run"]["mount_point"] == "hpc"
    assert main(["dataframe", str(pipeline_yml), "-m", "hpc", "--output", ""]) == 0
    assert seen["df"]["mount_point"] == "hpc"


def test_every_dataset_command_accepts_mount_point():
    parser = build_parser()
    for command in (
        "run",
        "dry-run",
        "status",
        "validate",
        "count-inputs",
        "inputs",
        "dataframe",
        "slurm-script",
        "tui",
        "migrate",
    ):
        args = parser.parse_args([command, "p.yml", "-m", "hpc"])
        assert args.mount_point == "hpc", command


def test_migrate_keeps_mount_spelling():
    args = build_parser().parse_args(["migrate", "p.yml", "--mount", "hpc"])
    assert args.mount_point == "hpc"


def test_tui_mount_point_sets_environment(pipeline_yml, monkeypatch):
    import neurodags.tui

    launched = []
    monkeypatch.setattr(
        neurodags.tui.NeuroDagsApp, "run", lambda self: launched.append(self._config_path)
    )
    assert main(["tui", str(pipeline_yml), "-m", "hpc"]) == 0
    assert launched
    assert os.environ[MOUNT_POINT_ENV] == "hpc"


# ---------------------------------------------------------------------------
# slurm-script
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pattern", ["per-file", "flat", "chained"])
def test_slurm_script_writes_mount_point(pipeline_yml, capsys, pattern):
    assert main(["slurm-script", str(pipeline_yml), "-m", "hpc", "--pattern", pattern]) == 0
    out = capsys.readouterr().out
    assert f"export {MOUNT_POINT_ENV}=hpc  # -m given to slurm-script" in out
    assert f"neurodags inputs {pipeline_yml} -m hpc -o inputs.tsv" in out
    assert f"neurodags count-inputs {pipeline_yml} -m hpc --inputs inputs.tsv" in out
    # The export follows the environment activation, which could otherwise reset it.
    assert out.index("source activate") < out.index(f"export {MOUNT_POINT_ENV}")


def test_slurm_script_without_mount_point_leaves_it_to_the_job(pipeline_yml, capsys):
    assert main(["slurm-script", str(pipeline_yml)]) == 0
    out = capsys.readouterr().out
    assert MOUNT_POINT_ENV not in out
    assert " -m " not in out


def test_slurm_script_quotes_mount_point(pipeline_yml, tmp_path):
    out = tmp_path / "run.sh"
    assert main(["slurm-script", str(pipeline_yml), "-m", "my mount;rm", "--output", str(out)]) == 0
    script = out.read_text()
    assert f"export {MOUNT_POINT_ENV}='my mount;rm'" in script
    subprocess.run(["bash", "-n", str(out)], check=True)
