"""Derivative names are used verbatim on disk and validated when a config is loaded (0.4.0).

Spec sections 2.1-2.2: names follow the grammar, are unique ignoring case, and every
entry point (run, dry-run, status, dataframe, validate, Python API) refuses invalid
names before anything is written.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

import neurodags.dag as dag
import neurodags.utils as utils
from neurodags._legacy import snake_to_camel
from neurodags.cli import main
from neurodags.loaders import check_pipeline_names, load_pipeline_configuration
from neurodags.naming import NamingError
from neurodags.orchestrators import (
    build_derivative_dataframe,
    iterate_derivative_pipeline,
    run_pipeline,
)


def _chain(*first_ref: str) -> dict:
    ref = first_ref[0] if first_ref else "SourceFile"
    return {
        "for_dataframe": True,
        "nodes": [
            {"id": 0, "derivative": ref},
            {"id": 1, "node": "dummy", "args": {"param1": "id.0"}},
        ],
    }


def _write_datasets(root: Path) -> None:
    (root / "in").mkdir()
    (root / "in" / "x0.txt").write_text("x0")
    (root / "ds.yml").write_text(
        yaml.safe_dump(
            {
                "d": {
                    "name": "d",
                    "source_root": str(root / "in"),
                    "file_pattern": str(root / "in" / "*.txt"),
                    "derivatives_path": str(root / "out"),
                }
            }
        )
    )


def _write_project(root: Path, definitions: dict, derivative_list: list) -> Path:
    _write_datasets(root)
    pipe = root / "pl.yml"
    pipe.write_text(
        yaml.safe_dump(
            {
                "naming_version": 1,
                "datasets": "ds.yml",
                "mount_point": None,
                "DerivativeDefinitions": definitions,
                "DerivativeList": derivative_list,
            }
        )
    )
    return pipe


def _tree(root: Path) -> list[Path]:
    return sorted(p.relative_to(root) for p in root.rglob("*"))


CASE_ALIASES = {"BandPower": _chain(), "bandpower": _chain()}


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("fit_w_l06", "FitWL06"),
        ("band_power", "BandPower"),
        ("BandPower", "BandPower"),
        ("EO_Features", "EoFeatures"),
        ("psd-welch", "psd-welch"),
    ],
)
def test_legacy_snake_to_camel_kept_for_migration(name, expected):
    # The 0.3.x on-disk rule survives privately for `neurodags migrate`.
    assert snake_to_camel(name) == expected


def test_removed_runtime_naming_apis():
    for attr in (
        "snake_to_camel",
        "derivative_file_names",
        "check_derivative_names",
        "analyze_bids_filename",
    ):
        assert not hasattr(utils, attr), attr
    assert not hasattr(dag, "_is_step_cached")
    assert not hasattr(dag, "_artifact_candidates_for")


def test_replace_bids_suffix_is_deprecated():
    with pytest.warns(DeprecationWarning, match="0.5"):
        result = utils.replace_bids_suffix("sub-01_eeg.vhdr", "desc-x", ".fif")
    assert result.name == "sub-01_desc-x.fif"


def test_names_are_verbatim_on_disk_and_in_references(tmp_path):
    pipe = _write_project(
        tmp_path,
        {"fit_w_l06": _chain(), "use_fit": _chain("fit_w_l06.message.txt")},
        ["use_fit"],
    )

    run_pipeline(str(pipe), raise_on_error=True)

    names = sorted(p.name for p in (tmp_path / "out").glob("x0.txt@*"))
    assert names == [
        "x0.txt@fit_w_l06.done",
        "x0.txt@fit_w_l06.message.txt",
        "x0.txt@use_fit.done",
        "x0.txt@use_fit.message.txt",
    ]
    manifest = json.loads((tmp_path / "out" / "x0.txt@use_fit.done").read_text())
    assert manifest["derivative"] == "use_fit"
    assert manifest["artifacts"] == {".message.txt": ["x0.txt@use_fit.message.txt"]}

    df = build_derivative_dataframe(str(pipe), include_derivatives=["fit_w_l06", "use_fit"])
    assert len(df) == 1
    assert "fit_w_l06.message.txt" in df.columns
    assert "use_fit.message.txt" in df.columns


def test_reference_by_camelcase_spelling_is_not_an_alias(tmp_path):
    pipe = _write_project(
        tmp_path,
        {"fit_w_l06": _chain(), "use_fit": _chain("FitWL06.message.txt")},
        ["use_fit"],
    )

    with pytest.raises(RuntimeError, match=r"Unknown derivative 'FitWL06\.message\.txt'"):
        run_pipeline(str(pipe), raise_on_error=True)


@pytest.mark.parametrize("dry_run", [False, True])
def test_run_with_case_aliases_fails_before_any_write(tmp_path, dry_run):
    # Only one of the pair is in the run set: all definitions are checked.
    pipe = _write_project(tmp_path, CASE_ALIASES, ["BandPower"])
    before = _tree(tmp_path)

    with pytest.raises(NamingError, match="differ only in case"):
        run_pipeline(str(pipe), dry_run=dry_run)

    assert _tree(tmp_path) == before


def test_iterate_and_dataframe_with_case_aliases_fail(tmp_path):
    pipe = _write_project(tmp_path, CASE_ALIASES, ["BandPower", "bandpower"])
    before = _tree(tmp_path)

    with pytest.raises(NamingError, match="differ only in case"):
        iterate_derivative_pipeline(str(pipe), "BandPower")
    with pytest.raises(NamingError, match="differ only in case"):
        build_derivative_dataframe(str(pipe))

    assert _tree(tmp_path) == before


@pytest.mark.parametrize("command", ["run", "dry-run", "status", "validate", "dataframe"])
def test_cli_with_case_aliases_fails_before_any_write(tmp_path, command, capsys):
    pipe = _write_project(tmp_path, CASE_ALIASES, ["BandPower"])
    before = _tree(tmp_path)

    assert main([command, str(pipe)]) == 2
    err = capsys.readouterr().err
    assert "'BandPower'" in err
    assert "'bandpower'" in err

    assert _tree(tmp_path) == before


def test_non_string_yaml_key_is_rejected(tmp_path):
    _write_datasets(tmp_path)
    pipe = tmp_path / "pl.yml"
    pipe.write_text(
        "naming_version: 1\n"
        "datasets: ds.yml\n"
        "DerivativeDefinitions:\n"
        "  Yes:\n"
        "    nodes:\n"
        "      - {id: 0, derivative: SourceFile}\n"
        "      - {id: 1, node: dummy, args: {param1: id.0}}\n"
    )
    with pytest.raises(NamingError, match="quote"):
        load_pipeline_configuration(str(pipe))


@pytest.mark.parametrize(
    ("definitions", "derivative_list", "fragment"),
    [
        ({"Fit.v2": _chain()}, [], "Fit.v2"),
        ({"SourceFile": _chain()}, [], "reserved"),
        ({"Clean": _chain(), "Clean-1": _chain()}, [], "split"),
        ({"A": _chain("B.c.d@x")}, [], "step id=0"),
        # Reference suffix grammar (spec section 5): checked at load, before any run.
        ({"A": _chain("B..fif")}, [], "'B..fif'"),
        ({"A": _chain("B.fif.")}, [], "'B.fif.'"),
        ({"A": _chain("B._x.fif")}, [], "'B._x.fif'"),
        ({"A": _chain("B.report.done")}, [], "'B.report.done'"),
        ({"A": _chain("SourceFile.fif")}, [], "takes no artifact suffix"),
        ({"A": _chain("sourcefile")}, [], "reserved"),
        ({"A": _chain()}, ["not ok"], "DerivativeList"),
    ],
)
def test_invalid_names_rejected_at_load(tmp_path, definitions, derivative_list, fragment):
    pipe = _write_project(tmp_path, definitions, derivative_list)
    with pytest.raises(NamingError) as excinfo:
        load_pipeline_configuration(str(pipe))
    assert fragment in str(excinfo.value)


def test_invalid_derivative_flag_is_rejected(tmp_path, capsys):
    pipe = _write_project(tmp_path, {"A": _chain()}, ["A"])
    before = _tree(tmp_path)

    assert main(["run", str(pipe), "--derivative", "A.nc"]) == 2
    assert "A.nc" in capsys.readouterr().err
    assert _tree(tmp_path) == before


def test_validate_prints_name_warnings(tmp_path, capsys):
    long_name = "L" * 70
    pipe = _write_project(tmp_path, {long_name: _chain(), "Fit_w-6_w-7": _chain()}, [])

    assert check_pipeline_names(load_pipeline_configuration(str(pipe)))
    assert main(["validate", str(pipe)]) == 0
    err = capsys.readouterr().err
    assert "warning:" in err
    assert "70 characters" in err
    assert "repeats key" in err
