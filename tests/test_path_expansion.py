"""Tests for ``~``/``$VAR`` expansion in dataset paths (#28)."""

from __future__ import annotations

import os
from pathlib import Path

import mne
import numpy as np
import pytest
import yaml

from neurodags.cli import main
from neurodags.datasets import get_datasets_and_mount_point_from_pipeline_configuration
from neurodags.definitions import DatasetConfig
from neurodags.iterators import get_all_files_across_datasets
from neurodags.orchestrators import build_derivative_dataframe, run_pipeline
from neurodags.utils import expand_path, get_path, resolve_dataset_paths

# ---------------------------------------------------------------------------
# expand_path / get_path unit tests
# ---------------------------------------------------------------------------


def test_get_path_expands_set_variable(monkeypatch, tmp_path):
    monkeypatch.setenv("ND_TEST_OUT", str(tmp_path / "x"))
    assert get_path("$ND_TEST_OUT/out") == f"{tmp_path / 'x'}/out"


def test_get_path_expands_braced_variable(monkeypatch, tmp_path):
    monkeypatch.setenv("ND_TEST_OUT", str(tmp_path / "x"))
    assert get_path("${ND_TEST_OUT}_suffix/out") == f"{tmp_path / 'x'}_suffix/out"


def test_get_path_expands_leading_tilde():
    assert get_path("~/nd28/derivatives") == str(Path.home() / "nd28" / "derivatives")


def test_get_path_expands_selected_mount_point(monkeypatch, tmp_path):
    monkeypatch.setenv("ND_TEST_OUT", str(tmp_path))
    paths = {"local": "$ND_TEST_OUT/out", "hpc": "/cluster/out"}
    assert get_path(paths, mount_point="local") == f"{tmp_path}/out"


def test_get_path_unset_variable_raises_naming_field_and_var(monkeypatch):
    monkeypatch.delenv("ND_TEST_UNSET", raising=False)
    monkeypatch.delenv("ND_TEST_ALSO_UNSET", raising=False)
    paths = {"local": "$ND_TEST_UNSET/out", "hpc": "$ND_TEST_ALSO_UNSET"}
    with pytest.raises(ValueError, match="ND_TEST_UNSET") as excinfo:
        get_path(paths, mount_point="local", field="derivatives_path")
    assert "derivatives_path" in str(excinfo.value)
    assert "ND_TEST_ALSO_UNSET" not in str(excinfo.value)


def test_get_path_unset_braced_variable_raises(monkeypatch):
    monkeypatch.delenv("ND_TEST_UNSET", raising=False)
    with pytest.raises(ValueError, match="ND_TEST_UNSET"):
        get_path("${ND_TEST_UNSET}/out")


def test_get_path_empty_variable_raises(monkeypatch):
    monkeypatch.setenv("ND_TEST_EMPTY", "")
    with pytest.raises(ValueError, match="empty: ND_TEST_EMPTY"):
        get_path("$ND_TEST_EMPTY/derivatives")


def test_get_path_inactive_mount_point_may_reference_unset_variable(monkeypatch):
    monkeypatch.delenv("ND_TEST_UNSET", raising=False)
    paths = {"local": "./out", "hpc": "$ND_TEST_UNSET/out"}
    assert get_path(paths, mount_point="local") == "./out"


@pytest.mark.parametrize("literal", ["data/a$1", "foo$", "data/$/x", "/data/~/x", "a~b"])
def test_get_path_non_identifier_dollar_and_inner_tilde_pass_through(literal):
    assert get_path(literal) == literal


def test_get_path_dict_without_mount_point_is_returned_unchanged(monkeypatch):
    monkeypatch.delenv("ND_TEST_UNSET", raising=False)
    paths = {"local": "$ND_TEST_UNSET/out"}
    assert get_path(paths, mount_point=None) is paths


def test_expand_path_value_containing_dollar_is_not_reexpanded(monkeypatch):
    monkeypatch.delenv("ND_TEST_UNSET", raising=False)
    monkeypatch.setenv("ND_TEST_OUT", "/odd/$ND_TEST_UNSET")
    assert expand_path("$ND_TEST_OUT/x") == "/odd/$ND_TEST_UNSET/x"


def test_expand_path_unresolvable_home_raises():
    with pytest.raises(ValueError, match="~"):
        expand_path("~nonexistent_user_nd28/x")


@pytest.mark.skipif(os.name != "nt", reason="%VAR% syntax is Windows-only")
def test_expand_path_windows_percent_syntax(monkeypatch):
    monkeypatch.setenv("ND_TEST_OUT", r"C:\scratch")
    assert expand_path(r"%ND_TEST_OUT%\out") == r"C:\scratch\out"


@pytest.mark.skipif(os.name == "nt", reason="POSIX keeps %VAR% literal")
def test_expand_path_percent_is_literal_on_posix():
    assert expand_path("data/%ND_TEST_OUT%/x") == "data/%ND_TEST_OUT%/x"


def test_resolve_dataset_paths_skips_skipped_datasets(monkeypatch, tmp_path):
    monkeypatch.delenv("ND_TEST_UNSET", raising=False)
    monkeypatch.setenv("ND_TEST_OUT", "/scratch/me")
    datasets = {
        "a": DatasetConfig(
            name="a",
            source_root=".",
            file_pattern="in/*.fif",
            derivatives_path="$ND_TEST_OUT/derivatives",
            exclude_pattern="$ND_TEST_OUT/bad*.fif",
        ),
        "b": DatasetConfig(
            name="b",
            source_root="$ND_TEST_UNSET",
            file_pattern="$ND_TEST_UNSET/*.fif",
            derivatives_path=None,
            skip=True,
        ),
    }
    # Relative paths resolve against the directory the dataset was defined in (0.4.0).
    datasets["a"]._base_dir = str(tmp_path)
    resolved = resolve_dataset_paths(datasets)
    assert resolved == {
        "a": {
            "source_root": str(tmp_path),
            "file_pattern": str(tmp_path / "in" / "*.fif"),
            "derivatives_path": "/scratch/me/derivatives",
            "exclude_pattern": "/scratch/me/bad*.fif",
            "derivatives_subdir": None,
            "derivatives_base": "/scratch/me/derivatives",
        }
    }
    datasets["b"].skip = False
    with pytest.raises(ValueError, match=r"datasets\.b\.source_root.*ND_TEST_UNSET"):
        resolve_dataset_paths(datasets)


# ---------------------------------------------------------------------------
# datasets: reference and exclude_pattern
# ---------------------------------------------------------------------------


def test_datasets_reference_is_expanded_before_pipeline_dir_join(monkeypatch, tmp_path):
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "ds.yml").write_text(yaml.safe_dump({"d": {"name": "d", "file_pattern": "in/*.fif"}}))
    pipe_dir = tmp_path / "pipe"
    pipe_dir.mkdir()
    pipe = pipe_dir / "pl.yml"
    pipe.write_text("naming_version: 1\ndatasets: $ND_TEST_PROJ/ds.yml\nmount_point: null\n")
    monkeypatch.setenv("ND_TEST_PROJ", str(proj))

    datasets, _ = get_datasets_and_mount_point_from_pipeline_configuration(str(pipe))
    assert list(datasets) == ["d"]

    monkeypatch.delenv("ND_TEST_PROJ")
    with pytest.raises(ValueError, match="ND_TEST_PROJ"):
        get_datasets_and_mount_point_from_pipeline_configuration(str(pipe))


def test_exclude_pattern_is_expanded(monkeypatch, tmp_path):
    for name in ("keep_1.fif", "bad_1.fif"):
        (tmp_path / name).write_text("")
    monkeypatch.setenv("ND_TEST_DATA", str(tmp_path))
    datasets = {
        "d": DatasetConfig(
            name="d",
            file_pattern="$ND_TEST_DATA/*.fif",
            exclude_pattern="$ND_TEST_DATA/bad_*.fif",
        )
    }
    files_per_dataset, _, _ = get_all_files_across_datasets(datasets)
    assert [Path(f).name for f in files_per_dataset["d"]] == ["keep_1.fif"]


def test_unset_exclude_pattern_variable_raises(monkeypatch, tmp_path):
    monkeypatch.delenv("ND_TEST_UNSET", raising=False)
    datasets = {
        "d": DatasetConfig(
            name="d",
            file_pattern=str(tmp_path / "*.fif"),
            exclude_pattern="$ND_TEST_UNSET/bad_*.fif",
        )
    }
    with pytest.raises(ValueError, match="exclude_pattern"):
        get_all_files_across_datasets(datasets)


# ---------------------------------------------------------------------------
# Regression: real run writes under the expanded dir, never a literal $VAR/~ dir
# ---------------------------------------------------------------------------

PIPELINE = {
    "naming_version": 1,
    "datasets": "ds.yml",
    "mount_point": "local",
    "DerivativeDefinitions": {
        "Meta": {
            "save": True,
            "for_dataframe": False,
            "nodes": [
                {"id": 0, "derivative": "SourceFile"},
                {"id": 1, "node": "extract_meeg_metadata", "args": {"mne_object": "id.0"}},
            ],
        }
    },
    "DerivativeList": ["Meta"],
}


def _write_project(root: Path, derivatives_path: str) -> Path:
    (root / "in").mkdir()
    for i in range(2):
        info = mne.create_info(["a", "b"], 100.0, "eeg")
        raw = mne.io.RawArray(np.random.default_rng(i).standard_normal((2, 1000)) * 1e-5, info)
        raw.save(root / "in" / f"x{i}_raw.fif", verbose="error")
    ds = {
        "d": {
            "name": "d",
            "source_root": {"local": str(root / "in"), "hpc": "$ND_TEST_HPC_ONLY"},
            # Absolute input glob: reused loky workers keep the cwd of the test that
            # spawned them, so a cwd-relative glob would break the n_jobs=2 case.
            "file_pattern": {
                "local": str(root / "in" / "*.fif"),
                "hpc": "$ND_TEST_HPC_ONLY/*.fif",
            },
            "derivatives_path": {"local": derivatives_path, "hpc": "$ND_TEST_HPC_ONLY/out"},
        }
    }
    (root / "ds.yml").write_text(yaml.safe_dump(ds))
    pipe = root / "pl.yml"
    pipe.write_text(yaml.safe_dump(PIPELINE))
    return pipe


def _no_literal_placeholder_dirs(root: Path) -> bool:
    return not any(p.name.startswith(("$", "~")) for p in root.rglob("*"))


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_run_writes_under_expanded_derivatives_path(monkeypatch, tmp_path, n_jobs):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ND_TEST_HPC_ONLY", raising=False)
    out = tmp_path / "scratch"
    monkeypatch.setenv("ND_TEST_OUTDIR", str(out))
    pipe = _write_project(tmp_path, "$ND_TEST_OUTDIR/derivatives")

    run_pipeline(str(pipe), raise_on_error=True, n_jobs=n_jobs)

    derivs = out / "derivatives"
    assert (derivs / "code" / "pl.yml").exists()
    assert sorted(p.name for p in derivs.glob("*@Meta*")) == [
        "x0_raw.fif@Meta.json",
        "x1_raw.fif@Meta.json",
    ]
    assert _no_literal_placeholder_dirs(tmp_path)


@pytest.mark.parametrize("dry_run", [False, True])
def test_run_with_unset_variable_fails_before_any_mkdir(monkeypatch, tmp_path, dry_run):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ND_TEST_OUTDIR", raising=False)
    monkeypatch.delenv("ND_TEST_HPC_ONLY", raising=False)
    pipe = _write_project(tmp_path, "$ND_TEST_OUTDIR/derivatives")
    before = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*"))

    # Default flags (raise_on_error=False): must still fail loudly, not be
    # swallowed into a per-file error record or a snapshot warning.
    with pytest.raises(ValueError, match="ND_TEST_OUTDIR"):
        run_pipeline(str(pipe), dry_run=dry_run)

    assert sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*")) == before
    assert _no_literal_placeholder_dirs(tmp_path)


def test_dataframe_with_unset_variable_fails_before_any_mkdir(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ND_TEST_OUTDIR", raising=False)
    pipe = _write_project(tmp_path, "$ND_TEST_OUTDIR/derivatives")
    config = yaml.safe_load(pipe.read_text())
    config["DerivativeDefinitions"]["Meta"]["for_dataframe"] = True
    pipe.write_text(yaml.safe_dump(config))

    with pytest.raises(ValueError, match="ND_TEST_OUTDIR"):
        build_derivative_dataframe(str(pipe))
    assert _no_literal_placeholder_dirs(tmp_path)


def test_validate_reports_resolved_paths_and_unset_variable(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ND_TEST_HPC_ONLY", raising=False)
    monkeypatch.setenv("ND_TEST_OUTDIR", str(tmp_path / "scratch"))
    pipe = _write_project(tmp_path, "$ND_TEST_OUTDIR/derivatives")

    assert main(["validate", str(pipe)]) == 0
    out = capsys.readouterr().out
    assert f"    derivatives_path: {tmp_path / 'scratch'}/derivatives" in out
    assert f"    source_root: {tmp_path / 'in'}" in out
    assert "    mount: local" in out

    monkeypatch.delenv("ND_TEST_OUTDIR")
    assert main(["validate", str(pipe)]) == 1
    assert "ND_TEST_OUTDIR" in capsys.readouterr().err
    assert _no_literal_placeholder_dirs(tmp_path)
