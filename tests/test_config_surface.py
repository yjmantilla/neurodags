"""0.4.0 configuration surface (spec §1, §4.4): naming_version, source_root,
derivatives_subdir, YAML-relative dataset paths, base_dir and the shared
derivatives_path check."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from neurodags.cli import _slurm_chained, _slurm_flat, _slurm_per_file, main
from neurodags.datasets import get_datasets_and_mount_point_from_pipeline_configuration
from neurodags.definitions import DatasetConfig
from neurodags.iterators import get_all_files_from_pipeline_configuration
from neurodags.loaders import (
    ConfigVersionError,
    check_naming_version,
    load_pipeline_configuration,
)
from neurodags.orchestrators import (
    _dataset_job_payload,
    build_derivative_dataframe,
    iterate_derivative_pipeline,
    run_pipeline,
)
from neurodags.utils import (
    DatasetConfigError,
    check_derivatives_subdir,
    effective_derivatives_base,
    inspect_dataset_paths,
    relative_to_source_root,
    resolve_config_path,
    resolve_dataset_paths,
    suggest_source_root,
)

DEFINITIONS = {
    "Fit": {
        "for_dataframe": True,
        "nodes": [
            {"id": 0, "derivative": "SourceFile"},
            {"id": 1, "node": "dummy", "args": {"param1": "id.0"}},
        ],
    }
}


def _make_inputs(root: Path, names=("sub-01/x0.txt", "sub-02/x1.txt")) -> Path:
    for name in names:
        f = root / name
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(name)
    return root


def _pipeline(datasets, **extra) -> dict:
    return {
        "naming_version": 1,
        "datasets": datasets,
        "mount_point": None,
        "DerivativeDefinitions": DEFINITIONS,
        "DerivativeList": ["Fit"],
        **extra,
    }


def _outputs(root: Path) -> list[str]:
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*@Fit*"))


# ---------------------------------------------------------------------------
# naming_version (§1.1)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("config", "match"),
    [
        ({}, "has no top-level 'naming_version'"),
        ({"naming_version": 2}, "declares naming_version: 2"),
        ({"naming_version": "1"}, "declares naming_version: '1'"),
        ({"naming_version": True}, "declares naming_version: True"),
        ({"naming_version": None}, "declares naming_version: None"),
    ],
)
def test_naming_version_refused(config, match):
    with pytest.raises(ConfigVersionError, match=match) as excinfo:
        check_naming_version(config)
    assert "neurodags migrate" in str(excinfo.value)


def test_naming_version_accepted():
    check_naming_version({"naming_version": 1})
    assert load_pipeline_configuration({"naming_version": 1, "x": 2}) == {
        "naming_version": 1,
        "x": 2,
    }


def test_naming_version_error_names_the_file(tmp_path):
    pipe = tmp_path / "pipeline.yml"
    pipe.write_text("datasets: {}\n")
    with pytest.raises(ConfigVersionError, match=str(pipe)):
        load_pipeline_configuration(pipe)


@pytest.mark.parametrize(
    "call",
    [
        lambda cfg: run_pipeline(cfg),
        lambda cfg: run_pipeline(cfg, dry_run=True),
        lambda cfg: iterate_derivative_pipeline(cfg, "Fit"),
        lambda cfg: build_derivative_dataframe(cfg),
        lambda cfg: get_datasets_and_mount_point_from_pipeline_configuration(cfg),
        lambda cfg: get_all_files_from_pipeline_configuration(cfg),
    ],
)
def test_every_api_entry_point_refuses_unversioned_config(tmp_path, call):
    _make_inputs(tmp_path / "in")
    cfg = _pipeline(
        {
            "d": {
                "name": "d",
                "source_root": str(tmp_path / "in"),
                "file_pattern": str(tmp_path / "in" / "**" / "*.txt"),
                "derivatives_path": str(tmp_path / "out"),
            }
        }
    )
    del cfg["naming_version"]
    with pytest.raises(ConfigVersionError, match="neurodags migrate"):
        call(cfg)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    "argv",
    [
        ["validate"],
        ["run"],
        ["dry-run"],
        ["status"],
        ["count-inputs"],
        ["dataframe"],
        ["dag"],
        ["slurm-script"],
        ["tui"],
    ],
)
def test_cli_refuses_unversioned_config_with_clean_error(tmp_path, capsys, argv):
    pipe = tmp_path / "pipeline.yml"
    pipe.write_text("datasets: {}\nDerivativeDefinitions: {A: {}}\nDerivativeList: [A]\n")
    with patch("neurodags.tui.NeuroDagsApp.run") as app_run:
        rc = main([argv[0], str(pipe), *argv[1:]])
    err = capsys.readouterr().err
    assert rc == 2
    app_run.assert_not_called()
    lines = err.strip().splitlines()
    assert len(lines) == 1
    assert lines[0].startswith("neurodags: error: pipeline configuration")
    assert "neurodags migrate" in lines[0]


def test_cli_unversioned_config_subprocess_has_no_traceback(tmp_path):
    pipe = tmp_path / "pipeline.yml"
    pipe.write_text("datasets: {}\n")
    result = subprocess.run(
        [sys.executable, "-m", "neurodags.cli", "run", str(pipe)],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert result.returncode == 2
    assert "Traceback" not in result.stderr
    assert len(result.stderr.strip().splitlines()) == 1
    assert "naming_version" in result.stderr


def test_tui_entry_point_refuses_unversioned_config(tmp_path, capsys):
    from neurodags.tui import main as tui_main

    pipe = tmp_path / "pipeline.yml"
    pipe.write_text("datasets: {}\n")
    with (
        patch.object(sys, "argv", ["neurodags-tui", str(pipe)]),
        patch("neurodags.tui.NeuroDagsApp.run") as app_run,
        pytest.raises(SystemExit) as excinfo,
    ):
        tui_main()
    assert excinfo.value.code == 2
    app_run.assert_not_called()
    assert "neurodags migrate" in capsys.readouterr().err


@pytest.mark.parametrize("make", [_slurm_per_file, _slurm_flat])
def test_slurm_templates_check_version_and_pass_base_dir(make):
    script = make("p/pipeline.yml", ["A"])
    assert "load_pipeline_configuration(config_path)" in script
    assert "base_dir=os.path.dirname(config_path)" in script


def test_slurm_chained_worker_passes_base_dir():
    _, worker = _slurm_chained("p/pipeline.yml", ["A", "B"])
    assert "base_dir=os.path.dirname(config_path)" in worker


# ---------------------------------------------------------------------------
# source_root suggestion and lexical containment (§1.2, §4.1 helper)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("pattern", "expected"),
    [
        ("/data/study/rawdata/**/*.vhdr", "/data/study/rawdata"),
        ("/data/study/rawdata/sub-*/eeg/*.vhdr", "/data/study/rawdata"),
        ("/data/study/sub-0?/x.vhdr", "/data/study"),
        ("/data/study/sub-0[12]/x.vhdr", "/data/study"),
        ("/data/study/sub-01/eeg.vhdr", "/data/study/sub-01"),  # no wildcard: parent
        ("/data//study/./**/*.vhdr", "/data/study/."),
        ("rel/dir/**/*.fif", "rel/dir"),
        ("/*/x.vhdr", None),  # would be the filesystem root
        ("/x.vhdr", None),
        ("**/*.vhdr", None),
        ("", None),
        ("C:\\data\\raw\\**\\*.vhdr", "C:\\data\\raw"),
        ("C:\\*\\x.vhdr", None),
    ],
)
def test_suggest_source_root(pattern, expected):
    assert suggest_source_root(pattern) == expected


def test_relative_to_source_root_inside(tmp_path):
    root = tmp_path / "raw"
    assert relative_to_source_root(root / "sub-01" / "x.vhdr", root) == os.path.join(
        "sub-01", "x.vhdr"
    )
    # lexical normalization, never realpath
    assert relative_to_source_root(f"{root}/./sub-01//x.vhdr", f"{root}/") == os.path.join(
        "sub-01", "x.vhdr"
    )


@pytest.mark.parametrize(
    "path",
    ["raw", "other/x.vhdr", "raw/../other/x.vhdr", "rawdata/x.vhdr"],
)
def test_relative_to_source_root_rejects_outside(tmp_path, path):
    with pytest.raises(ValueError, match="is not inside source_root"):
        relative_to_source_root(tmp_path / path, tmp_path / "raw")


# ---------------------------------------------------------------------------
# derivatives_subdir and the effective output base (§4.4)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, None), ("ds-a", "ds-a"), ("a/b", "a/b"), ("./a//b/", "a/b")],
)
def test_check_derivatives_subdir_valid(value, expected):
    assert check_derivatives_subdir(value) == expected


@pytest.mark.parametrize(
    ("value", "match"),
    [
        ("", "non-empty"),
        (3, "non-empty string"),
        ("/abs", "relative path"),
        ("C:/x", "relative path"),
        ("a\\b", "backslash"),
        ("..", "'\\.\\.'"),
        ("a/../b", "'\\.\\.'"),
        (".", "subfolder"),
        ("code", "reserved"),
        ("code/x", "reserved"),
        (".neurodags", "reserved"),
    ],
)
def test_check_derivatives_subdir_invalid(value, match):
    with pytest.raises(ValueError, match=match):
        check_derivatives_subdir(value)


def test_effective_derivatives_base():
    assert effective_derivatives_base(None, "a") is None
    assert effective_derivatives_base("/out", None) == "/out"
    assert effective_derivatives_base("/out", "a/b") == os.path.join("/out", "a", "b")


def _ds(name, root: Path, out: Path | None, **extra) -> DatasetConfig:
    return DatasetConfig(
        name=name,
        source_root=str(root),
        file_pattern=str(root / "**" / "*.txt"),
        derivatives_path=str(out) if out else None,
        **extra,
    )


def test_shared_derivatives_path_requires_distinct_subdirs(tmp_path):
    out = tmp_path / "out"
    datasets = {"a": _ds("a", tmp_path, out), "b": _ds("b", tmp_path, out / ".")}
    with pytest.raises(DatasetConfigError, match=r"share derivatives_path .* \(unset\)"):
        resolve_dataset_paths(datasets)

    datasets["b"] = _ds("b", tmp_path, out, derivatives_subdir="b")
    resolved = resolve_dataset_paths(datasets)  # unset + "b" differ: fine
    assert resolved["a"]["derivatives_base"] == str(out)
    assert resolved["b"]["derivatives_base"] == str(out / "b")

    datasets["a"] = _ds("a", tmp_path, out, derivatives_subdir="b/")
    with pytest.raises(DatasetConfigError, match=r"same derivatives_subdir \('b'\)"):
        resolve_dataset_paths(datasets)


def test_shared_derivatives_path_ignores_skipped_and_unset_paths(tmp_path):
    out = tmp_path / "out"
    datasets = {
        "a": _ds("a", tmp_path, out),
        "b": _ds("b", tmp_path, out, skip=True),
        "c": _ds("c", tmp_path, None),
        "d": _ds("d", tmp_path, None),
    }
    assert set(resolve_dataset_paths(datasets)) == {"a", "c", "d"}


def test_run_refuses_shared_derivatives_path_before_writing(tmp_path):
    _make_inputs(tmp_path / "in")
    ds = {
        n: {
            "name": n,
            "source_root": str(tmp_path / "in"),
            "file_pattern": str(tmp_path / "in" / "**" / "*.txt"),
            "derivatives_path": str(tmp_path / "out"),
        }
        for n in ("a", "b")
    }
    with pytest.raises(DatasetConfigError, match="share derivatives_path"):
        run_pipeline(_pipeline(ds))
    assert not (tmp_path / "out").exists()


# ---------------------------------------------------------------------------
# source_root required / existing (§1.2)
# ---------------------------------------------------------------------------


def test_missing_source_root_is_reported_with_suggestion(tmp_path):
    datasets = {
        "a": DatasetConfig(name="a", file_pattern=str(tmp_path / "raw" / "**" / "*.vhdr")),
        "b": DatasetConfig(name="b", file_pattern="/*/x.vhdr"),
        "s": DatasetConfig(name="s", file_pattern=str(tmp_path / "s" / "*.vhdr"), skip=True),
    }
    _, problems = inspect_dataset_paths(datasets)
    assert problems == [
        "datasets.a.source_root is required (since 0.4.0): set it to the root folder of the "
        f"dataset's source files; suggested: {tmp_path / 'raw'}",
        "datasets.b.source_root is required (since 0.4.0): set it to the root folder of the "
        "dataset's source files",
        "datasets.s.source_root is required (since 0.4.0): set it to the root folder of the "
        f"dataset's source files; suggested: {tmp_path / 's'}",
    ]
    with pytest.raises(DatasetConfigError, match=r"datasets\.a\.source_root is required"):
        resolve_dataset_paths(datasets)


def test_source_root_must_be_an_existing_directory(tmp_path):
    (tmp_path / "file.txt").write_text("x")
    for root in (tmp_path / "missing", tmp_path / "file.txt"):
        datasets = {"a": DatasetConfig(name="a", source_root=str(root), file_pattern="/x/*.txt")}
        with pytest.raises(DatasetConfigError, match="is not an existing directory"):
            resolve_dataset_paths(datasets)


def test_skipped_dataset_source_root_is_not_resolved(tmp_path, monkeypatch):
    monkeypatch.delenv("ND_TEST_UNSET", raising=False)
    datasets = {
        "a": DatasetConfig(
            name="a", source_root="$ND_TEST_UNSET", file_pattern="/x/*.txt", skip=True
        )
    }
    assert resolve_dataset_paths(datasets) == {}


def test_source_root_mount_map_needs_active_entry(tmp_path):
    datasets, mount = get_datasets_and_mount_point_from_pipeline_configuration(
        {
            "naming_version": 1,
            "mount_point": "hpc",
            "datasets": {
                "a": {
                    "name": "a",
                    "source_root": {"local": str(tmp_path)},
                    "file_pattern": {"local": "/x/*.txt", "hpc": "/y/*.txt"},
                }
            },
        }
    )
    _, problems = inspect_dataset_paths(datasets, mount)
    assert problems == ["datasets.a.source_root: no entry for mount point 'hpc' (available: local)"]
    with pytest.raises(KeyError):
        resolve_dataset_paths(datasets, mount)


def test_validate_prints_dataset_report_and_suggestion(tmp_path, capsys):
    _make_inputs(tmp_path / "raw")
    (tmp_path / "ds.yml").write_text(
        yaml.safe_dump(
            {
                "a": {
                    "name": "a",
                    "file_pattern": "raw/**/*.txt",
                    "derivatives_path": "out",
                    "derivatives_subdir": "a",
                }
            }
        )
    )
    pipe = tmp_path / "pipeline.yml"
    pipe.write_text(yaml.safe_dump(_pipeline("ds.yml")))
    assert main(["validate", str(pipe)]) == 1
    captured = capsys.readouterr()
    assert "  a:\n    mount: None\n    source_root: None\n" in captured.out
    assert f"    file_pattern: {tmp_path / 'raw'}/**/*.txt\n" in captured.out
    assert f"    derivatives_path: {tmp_path / 'out'}\n" in captured.out
    assert "    derivatives_subdir: a\n" in captured.out
    assert f"    outputs under: {tmp_path / 'out' / 'a'}\n" in captured.out
    assert f"suggested: {tmp_path / 'raw'}" in captured.err
    assert "Traceback" not in captured.err


def test_run_cli_reports_dataset_errors_cleanly(tmp_path, capsys):
    _make_inputs(tmp_path / "raw")
    pipe = tmp_path / "pipeline.yml"
    pipe.write_text(yaml.safe_dump(_pipeline({"a": {"name": "a", "file_pattern": "raw/**/*.txt"}})))
    assert main(["run", str(pipe)]) == 2
    err = capsys.readouterr().err
    assert err.startswith("neurodags: error: invalid dataset configuration:")
    assert "datasets.a.source_root is required" in err


# ---------------------------------------------------------------------------
# Relative path resolution (§1.3)
# ---------------------------------------------------------------------------


def _write_yaml_relative_project(root: Path) -> Path:
    """``root/cfg/pipeline.yml`` + ``root/cfg/data/ds.yml`` with paths relative to ds.yml."""
    _make_inputs(root / "cfg" / "data" / "raw")
    (root / "cfg" / "data" / "ds.yml").write_text(
        yaml.safe_dump(
            {
                "d": {
                    "name": "d",
                    "source_root": "raw",
                    "file_pattern": "raw/**/*.txt",
                    "exclude_pattern": "raw/sub-02/*.txt",
                    "derivatives_path": "../out",
                }
            }
        )
    )
    pipe = root / "cfg" / "pipeline.yml"
    pipe.write_text(yaml.safe_dump(_pipeline("data/ds.yml")))
    return pipe


def test_datasets_yaml_paths_resolve_against_that_yaml(tmp_path, monkeypatch):
    pipe = _write_yaml_relative_project(tmp_path)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    datasets, mount = get_datasets_and_mount_point_from_pipeline_configuration(str(pipe))
    resolved = resolve_dataset_paths(datasets, mount)["d"]
    data = tmp_path / "cfg" / "data"
    assert resolved["source_root"] == str(data / "raw")
    assert resolved["file_pattern"] == str(data / "raw" / "**" / "*.txt")
    assert resolved["exclude_pattern"] == str(data / "raw" / "sub-02" / "*.txt")
    assert resolved["derivatives_path"] == str(tmp_path / "cfg" / "out")

    run_pipeline(str(pipe), raise_on_error=True)
    assert _outputs(tmp_path) == ["cfg/out/x0.txt@Fit.message.txt"]
    assert not any(elsewhere.iterdir())


def test_cli_d_datasets_yaml_paths_resolve_against_that_yaml(tmp_path, monkeypatch, capsys):
    pipe = _write_yaml_relative_project(tmp_path)
    other = tmp_path / "other.yml"
    other.write_text(
        yaml.safe_dump(
            {
                "o": {
                    "name": "o",
                    "source_root": "cfg/data/raw",
                    "file_pattern": "cfg/data/raw/**/*.txt",
                    "derivatives_path": "other_out",
                }
            }
        )
    )
    monkeypatch.chdir(tmp_path / "cfg")
    assert main(["validate", str(pipe), "-d", "../other.yml"]) == 0
    out = capsys.readouterr().out
    assert f"    source_root: {tmp_path / 'cfg' / 'data' / 'raw'}\n" in out
    assert f"    derivatives_path: {tmp_path / 'other_out'}\n" in out


def test_inline_datasets_resolve_against_pipeline_yaml(tmp_path, monkeypatch):
    _make_inputs(tmp_path / "proj" / "raw")
    pipe = tmp_path / "proj" / "pipeline.yml"
    pipe.write_text(
        yaml.safe_dump(
            _pipeline(
                {
                    "d": {
                        "name": "d",
                        "source_root": "raw",
                        "file_pattern": "raw/**/*.txt",
                        "derivatives_path": "out",
                    }
                }
            )
        )
    )
    monkeypatch.chdir(tmp_path)
    run_pipeline(str(pipe), raise_on_error=True)
    assert _outputs(tmp_path) == [
        "proj/out/sub-01/x0.txt@Fit.message.txt",
        "proj/out/sub-02/x1.txt@Fit.message.txt",
    ]


def test_expanded_absolute_paths_are_used_as_is(tmp_path, monkeypatch):
    monkeypatch.setenv("ND_TEST_RAW", str(tmp_path / "raw"))
    assert resolve_config_path("$ND_TEST_RAW/x", base_dir="/elsewhere") == str(
        tmp_path / "raw" / "x"
    )
    assert resolve_config_path("x/../y", base_dir=str(tmp_path)) == str(tmp_path / "y")


def test_api_mapping_with_relative_paths_requires_base_dir(tmp_path, monkeypatch):
    _make_inputs(tmp_path / "raw")
    monkeypatch.chdir(tmp_path)
    cfg = _pipeline(
        {
            "d": {
                "name": "d",
                "source_root": "raw",
                "file_pattern": "raw/**/*.txt",
                "derivatives_path": "out",
            }
        }
    )
    with pytest.raises(ValueError, match=r"relative path.*base_dir"):
        run_pipeline(cfg)
    assert not (tmp_path / "out").exists()

    other = tmp_path / "x"
    other.mkdir()
    monkeypatch.chdir(other)  # base_dir, not the cwd, anchors the paths
    run_pipeline(cfg, base_dir=tmp_path, raise_on_error=True)
    assert _outputs(tmp_path) == [
        "out/sub-01/x0.txt@Fit.message.txt",
        "out/sub-02/x1.txt@Fit.message.txt",
    ]

    df = build_derivative_dataframe(cfg, base_dir=tmp_path)
    assert sorted(df["file_path"]) == [
        str(tmp_path / "raw" / "sub-01" / "x0.txt"),
        str(tmp_path / "raw" / "sub-02" / "x1.txt"),
    ]


def test_api_datasets_mapping_uses_base_dir_even_with_pipeline_file(tmp_path):
    _make_inputs(tmp_path / "raw")
    pipe = tmp_path / "cfg" / "pipeline.yml"
    pipe.parent.mkdir()
    pipe.write_text(yaml.safe_dump(_pipeline({})))
    mapping = {"d": {"name": "d", "source_root": "raw", "file_pattern": "raw/**/*.txt"}}

    datasets, mount = get_datasets_and_mount_point_from_pipeline_configuration(
        str(pipe), datasets_input=mapping
    )
    with pytest.raises(ValueError, match="base_dir"):
        resolve_dataset_paths(datasets, mount)

    datasets, mount = get_datasets_and_mount_point_from_pipeline_configuration(
        str(pipe), datasets_input=mapping, base_dir=tmp_path
    )
    assert resolve_dataset_paths(datasets, mount)["d"]["source_root"] == str(tmp_path / "raw")


def test_worker_payload_has_absolute_resolved_paths(tmp_path, monkeypatch):
    pipe = _write_yaml_relative_project(tmp_path)
    monkeypatch.chdir(tmp_path)
    datasets, mount = get_datasets_and_mount_point_from_pipeline_configuration(str(pipe))
    payload = _dataset_job_payload(datasets["d"], mount)
    data = tmp_path / "cfg" / "data"
    assert payload["source_root"] == str(data / "raw")
    assert payload["file_pattern"] == str(data / "raw" / "**" / "*.txt")
    assert payload["exclude_pattern"] == str(data / "raw" / "sub-02" / "*.txt")
    assert payload["derivatives_path"] == str(tmp_path / "cfg" / "out")
    # A worker rebuilds the config without the origin directory: still resolvable.
    rebuilt = DatasetConfig(**payload)
    assert resolve_dataset_paths({"d": rebuilt})["d"]["source_root"] == str(data / "raw")
