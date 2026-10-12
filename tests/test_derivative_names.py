"""Derivative names vs on-disk names: snake_to_camel mapping and collisions (#42)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from neurodags.cli import main
from neurodags.orchestrators import (
    build_derivative_dataframe,
    iterate_derivative_pipeline,
    run_pipeline,
)
from neurodags.utils import (
    check_derivative_names,
    derivative_file_names,
    snake_to_camel,
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


def _write_project(root: Path, definitions: dict, derivative_list: list[str]) -> Path:
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


COLLIDING = {"band_power": _chain(), "BandPower": _chain()}


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("fit_w_l06", "FitWL06"),
        ("band_power", "BandPower"),
        ("BandPower", "BandPower"),
        ("Fit_A", "FitA"),
        ("EO_Features", "EoFeatures"),
        ("Ep_DetrendedFluctuation", "EpDetrendedfluctuation"),
        ("ICA_cleaned", "IcaCleaned"),
        ("psd-welch", "psd-welch"),
    ],
)
def test_snake_to_camel_mapping_is_stable(name, expected):
    # On-disk names of existing caches must never change.
    assert snake_to_camel(name) == expected


def test_derivative_file_names_and_no_collision():
    names = ["fit_w_l06", "BandPower", "psd-welch"]
    assert derivative_file_names(names) == {
        "fit_w_l06": "FitWL06",
        "BandPower": "BandPower",
        "psd-welch": "psd-welch",
    }
    assert check_derivative_names(names) == derivative_file_names(names)


def test_collision_message_names_both_yaml_names_and_on_disk_name():
    with pytest.raises(ValueError, match="collide on disk") as excinfo:
        check_derivative_names(["band_power", "Other", "BandPower"])
    msg = str(excinfo.value)
    assert "'band_power'" in msg
    assert "'BandPower'" in msg
    assert "@BandPower" in msg
    assert "Other" not in msg


@pytest.mark.parametrize("dry_run", [False, True])
def test_run_with_colliding_names_fails_before_any_write(tmp_path, dry_run):
    # Only one of the pair is in the run set: all definitions are checked.
    pipe = _write_project(tmp_path, COLLIDING, ["band_power"])
    before = _tree(tmp_path)

    # Default flags: must not be swallowed into a per-file error record.
    with pytest.raises(ValueError, match="band_power"):
        run_pipeline(str(pipe), dry_run=dry_run)

    assert _tree(tmp_path) == before


def test_iterate_and_dataframe_with_colliding_names_fail(tmp_path):
    pipe = _write_project(tmp_path, COLLIDING, ["band_power", "BandPower"])
    before = _tree(tmp_path)

    with pytest.raises(ValueError, match="@BandPower"):
        iterate_derivative_pipeline(str(pipe), "band_power")
    with pytest.raises(ValueError, match="@BandPower"):
        build_derivative_dataframe(str(pipe))

    assert _tree(tmp_path) == before


@pytest.mark.parametrize("command", ["run", "dry-run", "status"])
def test_cli_with_colliding_names_fails_before_any_write(tmp_path, command):
    pipe = _write_project(tmp_path, COLLIDING, ["band_power"])
    before = _tree(tmp_path)

    with pytest.raises(ValueError, match="collide on disk"):
        main([command, str(pipe)])

    assert _tree(tmp_path) == before


def test_validate_rejects_colliding_names(tmp_path, capsys):
    pipe = _write_project(tmp_path, COLLIDING, ["band_power"])

    assert main(["validate", str(pipe)]) == 1
    err = capsys.readouterr().err
    assert "'band_power'" in err
    assert "'BandPower'" in err
    assert "@BandPower" in err


def test_validate_prints_on_disk_name_mapping(tmp_path, capsys):
    pipe = _write_project(
        tmp_path,
        {"fit_w_l06": _chain(), "BandPower": _chain(), "psd-welch": _chain()},
        ["fit_w_l06"],
    )

    assert main(["validate", str(pipe)]) == 0
    out = capsys.readouterr().out
    assert "on-disk names differing from YAML names (1):" in out
    assert "  fit_w_l06 -> files named @FitWL06" in out
    # Unchanged names are not listed.
    assert "BandPower ->" not in out
    assert "psd-welch ->" not in out


def test_validate_omits_mapping_when_all_names_unchanged(tmp_path, capsys):
    pipe = _write_project(tmp_path, {"BandPower": _chain()}, ["BandPower"])

    assert main(["validate", str(pipe)]) == 0
    assert "on-disk names" not in capsys.readouterr().out


def test_snake_names_run_and_are_referenced_by_yaml_name(tmp_path):
    # Downstream references use the YAML name (fit_w_l06.message.txt); the files
    # use the CamelCase on-disk name.
    pipe = _write_project(
        tmp_path,
        {"fit_w_l06": _chain(), "use_fit": _chain("fit_w_l06.message.txt")},
        ["use_fit"],
    )

    run_pipeline(str(pipe), raise_on_error=True)

    names = sorted(p.name for p in (tmp_path / "out").glob("x0.txt@*"))
    assert names == ["x0.txt@FitWL06.message.txt", "x0.txt@UseFit.message.txt"]

    df = build_derivative_dataframe(str(pipe), include_derivatives=["fit_w_l06", "use_fit"])
    assert len(df) == 1
    # Dataframe columns use the YAML name, not the on-disk name.
    assert "fit_w_l06.message.txt" in df.columns
    assert "use_fit.message.txt" in df.columns


def test_reference_by_on_disk_name_is_not_a_derivative(tmp_path):
    # The on-disk name is not an alias for the definition: when nothing is
    # cached, referencing it fails as an unknown derivative.
    pipe = _write_project(
        tmp_path,
        {"fit_w_l06": _chain(), "use_fit": _chain("FitWL06.message.txt")},
        ["use_fit"],
    )

    with pytest.raises(RuntimeError, match=r"Unknown derivative 'FitWL06\.message\.txt'"):
        run_pipeline(str(pipe), raise_on_error=True)


@pytest.mark.parametrize(
    ("name", "reason"),
    [
        ("Fit.v2", "contains '.'"),
        ("A@B", "contains '@'"),
        ("a/b", "contains '/'"),
        ("a\\b", "path separator"),
        ("F*", "glob wildcard"),
        ("F?", "glob wildcard"),
        ("F[1]", "glob character"),
        ("SourceFile", "reserved"),
        ("", "empty derivative name"),
    ],
)
def test_layout_breaking_names_are_rejected(name, reason):
    with pytest.raises(ValueError, match="invalid derivative names") as excinfo:
        check_derivative_names(["Fit", name])
    assert reason in str(excinfo.value)


@pytest.mark.parametrize("name", ["psd-welch", "A&B", "fit_w_l06", "Band Power", "source_file"])
def test_other_names_are_accepted(name):
    assert check_derivative_names(["Fit", name])


def test_names_differing_only_in_case_collide():
    # Same file on case-insensitive filesystems (macOS, Windows).
    with pytest.raises(ValueError, match="case-insensitive") as excinfo:
        check_derivative_names(["Bandpower", "band_power"])
    assert "'@BandPower' / '@Bandpower'" in str(excinfo.value)


def test_dotted_name_rejected_before_run(tmp_path):
    # 'Fit.v2' outputs (@Fit.v2.*) would otherwise make 'Fit' look cached.
    pipe = _write_project(tmp_path, {"Fit": _chain(), "Fit.v2": _chain()}, ["Fit"])
    before = _tree(tmp_path)

    with pytest.raises(ValueError, match=r"'Fit\.v2' contains '\.'"):
        run_pipeline(str(pipe))
    assert main(["validate", str(pipe)]) == 1
    assert _tree(tmp_path) == before
