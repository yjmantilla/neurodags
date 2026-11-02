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
from neurodags.iterators import (
    enumerate_pipeline_inputs,
    get_all_files_from_pipeline_configuration,
)
from neurodags.orchestrators import run_pipeline
from neurodags.utils import DatasetConfigError, MountPointError

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


def test_unknown_mount_is_a_dataset_config_error(pipeline_yml):
    from neurodags.utils import resolve_dataset_paths

    datasets, mount = get_datasets_and_mount_point_from_pipeline_configuration(
        pipeline_yml, mount_point="nowhere"
    )
    with pytest.raises(DatasetConfigError, match=r"'nowhere' \(available: hpc, lab\)"):
        resolve_dataset_paths(datasets, mount)
    # Still a KeyError, as before, for callers that caught that.
    with pytest.raises(KeyError):
        resolve_dataset_paths(datasets, mount)


def test_mount_point_error_survives_pickling():
    import pickle

    error = pickle.loads(pickle.dumps(MountPointError(["ds: no entry"])))
    assert type(error) is MountPointError
    assert error.problems == ["ds: no entry"]
    assert str(error) == "invalid dataset configuration:\n  - ds: no entry"


@pytest.mark.parametrize(
    "enumerate_",
    [enumerate_pipeline_inputs, get_all_files_from_pipeline_configuration],
)
def test_enumeration_apis_check_the_dataset_paths(tmp_path, enumerate_):
    config = _config(tmp_path)
    with pytest.raises(MountPointError, match=r"'nowhere' \(available: lab, hpc\)"):
        enumerate_(config, mount_point="nowhere")
    # A map lacking the mount in derivatives_path only is refused too, as by run_pipeline.
    del config["datasets"]["ds"]["derivatives_path"]["hpc"]
    with pytest.raises(MountPointError, match=r"derivatives_path: no entry for mount point 'hpc'"):
        enumerate_(config, mount_point="hpc")


@pytest.mark.parametrize(
    "command", ["run", "dry-run", "status", "inputs", "count-inputs", "dataframe"]
)
def test_cli_unknown_mount_is_a_clean_error(pipeline_yml, capsys, monkeypatch, command):
    monkeypatch.chdir(pipeline_yml.parent)  # dry-run writes its CSV to the cwd
    # dataframe returns early without a for_dataframe derivative, before loading the datasets.
    config = yaml.safe_load(pipeline_yml.read_text())
    config["DerivativeDefinitions"]["Prep"]["for_dataframe"] = True
    pipeline_yml.write_text(yaml.safe_dump(config))
    assert main([command, str(pipeline_yml), "-m", "nowhere"]) == 2
    err = capsys.readouterr().err
    assert (
        "neurodags: error: invalid dataset configuration:\n"
        "  - datasets.ds.source_root: no entry for mount point 'nowhere' (available: hpc, lab)\n"
    ) in err
    assert "KeyError" not in err
    assert "NEURODAGS_MOUNT_POINT" not in err


def test_mount_error_does_not_blame_the_variable_for_an_explicit_flag(
    pipeline_yml, capsys, monkeypatch
):
    monkeypatch.setenv(MOUNT_POINT_ENV, "nowhere")
    assert main(["count-inputs", str(pipeline_yml), "-m", "nowhere"]) == 2
    assert "NEURODAGS_MOUNT_POINT" not in capsys.readouterr().err
    assert main(["validate", str(pipeline_yml), "-m", "nowhere"]) == 1
    captured = capsys.readouterr()
    assert "mount_point: nowhere (from -m/--mount-point)" in captured.out
    assert "NEURODAGS_MOUNT_POINT" not in captured.err


def test_cli_stale_environment_mount_names_the_variable(pipeline_yml, capsys, monkeypatch):
    monkeypatch.setenv(MOUNT_POINT_ENV, "stale-host")
    assert main(["count-inputs", str(pipeline_yml)]) == 2
    err = capsys.readouterr().err
    assert "'stale-host' (available: hpc, lab)" in err
    assert f"set by ${MOUNT_POINT_ENV}" in err


@pytest.mark.parametrize("command", ["count-inputs", "migrate", "tui"])
def test_empty_mount_point_flag_is_rejected(command, capsys):
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args([command, "p.yml", "-m", ""])
    assert exc.value.code == 2
    assert "cannot be empty" in capsys.readouterr().err


def test_empty_override_counts_as_unset(monkeypatch):
    assert resolve_mount_point("lab", "") == ("lab", "pipeline")
    monkeypatch.setenv(MOUNT_POINT_ENV, "hpc")
    assert resolve_mount_point("lab", "") == ("hpc", "environment")


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
        (["-m", "hpc"], None, "mount_point: hpc (from -m/--mount-point)"),
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
    assert data["mount_point_source"] == "argument"  # same tokens as neurodags_env.json
    assert data["n_files"] == 3

    main(["status", str(pipeline_yml), "-m", "hpc"])
    assert "mount:  hpc (from -m/--mount-point)" in capsys.readouterr().out


def test_status_shows_mount_when_no_files_are_found(pipeline_yml, capsys, monkeypatch):
    for f in (pipeline_yml.parent / "hpc" / "raw").rglob("*.vhdr"):
        f.unlink()
    monkeypatch.setenv(MOUNT_POINT_ENV, "hpc")
    assert main(["status", str(pipeline_yml)]) == 0
    out = capsys.readouterr().out
    assert f"mount:  hpc (from ${MOUNT_POINT_ENV})" in out
    assert out.index("mount:") < out.index("No files found.")


def test_run_pipeline_logs_mount_and_source(pipeline_yml, monkeypatch):
    from neurodags import orchestrators

    events = []
    monkeypatch.setattr(orchestrators, "iterate_derivative_pipeline", lambda **kw: None)
    monkeypatch.setattr(orchestrators.log, "info", lambda event, **kw: events.append((event, kw)))
    config = yaml.safe_load(pipeline_yml.read_text())
    monkeypatch.setenv(MOUNT_POINT_ENV, "hpc")
    run_pipeline(config, base_dir=pipeline_yml.parent, dry_run=True)
    assert ("Active mount point", {"mount_point": "hpc", "source": "environment"}) in events


def test_migrate_preview_shows_mount_source(pipeline_yml, monkeypatch):
    from neurodags import migrate

    plan = migrate.plan_migration(str(pipeline_yml))
    assert (plan.mount, plan.mount_source) == ("lab", "pipeline")
    monkeypatch.setenv(MOUNT_POINT_ENV, "hpc")
    plan = migrate.plan_migration(str(pipeline_yml))
    assert (plan.mount, plan.mount_source) == ("hpc", "environment")
    assert f"mount:       hpc (from ${MOUNT_POINT_ENV})" in migrate.format_preview(plan)
    assert plan.to_dict()["mount_source"] == "environment"
    plan = migrate.plan_migration(str(pipeline_yml), mount="lab")
    assert (plan.mount, plan.mount_source) == ("lab", "argument")


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


def test_legacy_tui_entry_point_accepts_mount_point(pipeline_yml, monkeypatch):
    import sys

    import neurodags.tui

    monkeypatch.setattr(neurodags.tui.NeuroDagsApp, "run", lambda self: None)
    monkeypatch.setattr(sys, "argv", ["neurodags-tui", str(pipeline_yml), "--mount-point", "hpc"])
    neurodags.tui.main()
    assert os.environ[MOUNT_POINT_ENV] == "hpc"


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


def test_slurm_script_refuses_a_mount_the_datasets_do_not_define(pipeline_yml, tmp_path, capsys):
    out = tmp_path / "run.sh"
    assert main(["slurm-script", str(pipeline_yml), "-m", "nowhere", "--output", str(out)]) == 2
    assert "no entry for mount point 'nowhere' (available: hpc, lab)" in capsys.readouterr().err
    assert not out.exists()


def test_slurm_script_does_not_check_the_paths_of_the_mount(tmp_path, capsys):
    # The cluster's paths need not exist where the scripts are generated.
    config = _config(tmp_path)
    config["datasets"]["ds"]["source_root"]["hpc"] = "/only/on/the/cluster"
    path = tmp_path / "pipeline.yml"
    path.write_text(yaml.safe_dump(config))
    assert main(["slurm-script", str(path), "-m", "hpc"]) == 0


def test_slurm_script_quotes_mount_point(tmp_path):
    config = _config(tmp_path)
    for field in ("source_root", "file_pattern", "derivatives_path"):
        values = config["datasets"]["ds"][field]
        values["my mount;rm"] = values.pop("hpc")
    pipeline_yml = tmp_path / "pipeline.yml"
    pipeline_yml.write_text(yaml.safe_dump(config))
    out = tmp_path / "run.sh"
    assert main(["slurm-script", str(pipeline_yml), "-m", "my mount;rm", "--output", str(out)]) == 0
    script = out.read_text()
    assert f"export {MOUNT_POINT_ENV}='my mount;rm'" in script
    subprocess.run(["bash", "-n", str(out)], check=True)
