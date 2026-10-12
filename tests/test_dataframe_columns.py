"""Reversible dataframe columns (spec 0.4.0 section 7): encoding, collisions, errors, recipe."""

from __future__ import annotations

import re
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from neurodags.cli import main
from neurodags.dag import collect_derivative_for_dataframe
from neurodags.naming import ColumnCollisionError, NamingError, parse_column
from neurodags.orchestrators import (
    _collect_dataframe_file,
    build_derivative_dataframe,
    iterate_derivative_pipeline,
)

DOCS = Path(__file__).resolve().parents[1] / "docs" / "user_guide"


def _collect(payloads: dict, name: str = "Power") -> dict:
    with patch("neurodags.dag._resolve_derivative_artifacts", return_value=payloads):
        return collect_derivative_for_dataframe(
            {"save": False}, name, "input.vhdr", reference_base=Path("unused")
        )


def _collect_row(tmp_path, values: dict, *, raise_on_error: bool = False, long: bool = False):
    def fake_collect(derivative_def, derivative_name, *args, **kwargs):
        result = values[derivative_name]
        if isinstance(result, Exception):
            raise result
        return result

    with patch("neurodags.orchestrators.collect_derivative_for_dataframe", fake_collect):
        return _collect_dataframe_file(
            index=0,
            dataset_name="ds",
            file_path=str(tmp_path / "x.vhdr"),
            dataset_config_dict={"name": "ds", "file_pattern": str(tmp_path / "*.vhdr")},
            source_path=str(tmp_path / "x.vhdr"),
            common_root=str(tmp_path),
            mount_point=None,
            selected_derivatives=list(values),
            derivative_definitions={key: {} for key in values},
            preserve_complex_values=False,
            collect_long_rows=long,
            raise_on_error=raise_on_error,
            n_files=1,
        )


# ---------------------------------------------------------------------------
# Wide column encoding
# ---------------------------------------------------------------------------


def test_flatten_uses_format_column_with_sorted_dims():
    da = xr.DataArray(
        np.array([[1.0, 2.0]]),
        dims=["channel", "band"],
        coords={"channel": ["Fp1"], "band": ["alpha", "beta"]},
    )
    row = _collect({".nc": da}, name="BandPower")
    assert row == {
        "BandPower.nc@band-alpha_channel-Fp1": 1.0,
        "BandPower.nc@band-beta_channel-Fp1": 2.0,
    }


@pytest.mark.parametrize(
    ("dim", "value", "column"),
    [
        ("band", "8-12", "X.nc@band-8-12"),
        ("freq", -1.5, "X.nc@freq--1.5"),
        ("freq", np.float64(-1.5), "X.nc@freq--1.5"),
        ("channel", "EEG 001", "X.nc@channel-EEG%20001"),
        ("cond", "a_b", "X.nc@cond-a%5Fb"),
        ("ref", "a@b", "X.nc@ref-a%40b"),
        ("x", "1,5", "X.nc@x-1%2C5"),
        ("freq_band", "alpha", "X.nc@freq%5Fband-alpha"),
        ("k", "", "X.nc@k-"),
        ("k", 3, "X.nc@k-3"),
    ],
)
def test_coordinate_encoding_examples(dim, value, column):
    da = xr.DataArray(np.array([7.0]), dims=[dim], coords={dim: [value]})
    row = _collect({".nc": da}, name="X")
    assert list(row) == [column]
    parsed = parse_column(column)
    assert parsed.name == "X"
    assert parsed.suffix == ".nc"
    assert parsed.coords == ((dim, str(value.item() if hasattr(value, "item") else value)),)


def test_dims_without_coords_use_positions():
    da = xr.DataArray(np.arange(4.0).reshape(2, 2), dims=["b", "a"])
    row = _collect({".nc": da}, name="X")
    assert row == {
        "X.nc@a-0_b-0": 0.0,
        "X.nc@a-1_b-0": 1.0,
        "X.nc@a-0_b-1": 2.0,
        "X.nc@a-1_b-1": 3.0,
    }


def test_scalar_and_non_array_artifacts_use_name_and_suffix():
    row = _collect(
        {".nc": xr.DataArray(np.float64(2.5)), ".report.json": {"a": 1}, ".txt": "hi"}, name="Fit"
    )
    assert row == {"Fit.nc": 2.5, "Fit.report.json": {"a": 1}, "Fit.txt": "hi"}


def test_every_generated_column_round_trips():
    da = xr.DataArray(
        np.zeros((2, 2)),
        dims=["freq_band", "dim 1"],
        coords={"freq_band": ["a_b", "a-b"], "dim 1": ["α", "x%y"]},
    )
    row = _collect({".nc": da}, name="Fit_alpha-0p5")
    assert len(row) == 4
    for column in row:
        parsed = parse_column(column)
        assert parsed is not None
        assert parsed.name == "Fit_alpha-0p5"
        assert {d for d, _ in parsed.coords} == {"freq_band", "dim 1"}


def test_metadata_columns_cannot_collide_with_generated_columns(tmp_path):
    row, _ = _collect_row(tmp_path, {"A": {"A.nc": 1.0}, "B": ValueError("boom")})
    metadata = {"index", "dataset", "file_path"}
    generated = set(row) - metadata
    assert generated == {"A.nc", "B.error"}
    assert all("." not in c for c in metadata)
    assert all("." in c for c in generated)
    assert all(parse_column(c) is None for c in metadata)


# ---------------------------------------------------------------------------
# Collisions are errors
# ---------------------------------------------------------------------------


def test_non_unique_coordinate_index_is_an_error():
    da = xr.DataArray(np.array([1.0, 2.0]), dims=["ch"], coords={"ch": ["Fp1", "Fp1"]})
    with pytest.raises(ColumnCollisionError) as excinfo:
        _collect({".nc": da})
    message = str(excinfo.value)
    assert "'Power.nc@ch-Fp1'" in message
    assert message.count("ch='Fp1'") == 2


def test_mixed_type_coordinates_are_an_error():
    da = xr.DataArray(
        np.array([1.0, 2.0]), dims=["k"], coords={"k": np.array([1, "1"], dtype=object)}
    )
    with pytest.raises(ColumnCollisionError, match=r"k=1 and .*k='1'"):
        _collect({".nc": da})


def test_old_lossy_encoding_pair_no_longer_collides():
    # 0.3.x wrote both as "ch-a~b" and kept only the last value.
    da = xr.DataArray(np.array([1.0, 2.0]), dims=["ch"], coords={"ch": ["a-b", "a~b"]})
    assert _collect({".nc": da}) == {"Power.nc@ch-a-b": 1.0, "Power.nc@ch-a~b": 2.0}


def test_collision_across_derivatives_in_row_merge_is_an_error(tmp_path):
    values = {"A": {"Shared.nc": 1.0, "A.nc": 1.0}, "B": {"Shared.nc": 2.0}}
    with pytest.raises(ColumnCollisionError, match=r"'Shared.nc' from derivative 'A' and .*'B'"):
        _collect_row(tmp_path, values, raise_on_error=False)


def test_collision_is_not_swallowed_by_raise_on_error_false(dummy_pipeline):
    cfg = dummy_pipeline["config"]
    iterate_derivative_pipeline(cfg, "BasicPrep", raise_on_error=True)
    iterate_derivative_pipeline(cfg, "Spectrum", raise_on_error=True)
    with patch(
        "neurodags.orchestrators.collect_derivative_for_dataframe",
        side_effect=ColumnCollisionError("Dataframe column collision: simulated"),
    ):
        with pytest.raises(ColumnCollisionError, match="simulated"):
            build_derivative_dataframe(cfg, raise_on_error=False)


def test_cli_reports_collision_cleanly(dummy_pipeline, tmp_path, capsys):
    import yaml

    cfg = dummy_pipeline["config"]
    config_path = tmp_path / "pipeline.yml"
    config_path.write_text(yaml.safe_dump(cfg))
    with patch(
        "neurodags.orchestrators.collect_derivative_for_dataframe",
        side_effect=ColumnCollisionError("Dataframe column collision: simulated"),
    ):
        rc = main(["dataframe", str(config_path)])
    assert rc == 2
    assert "neurodags: error: Dataframe column collision: simulated" in capsys.readouterr().err


def test_invalid_selected_derivative_name_is_rejected(dummy_pipeline):
    cfg = dict(dummy_pipeline["config"])
    definitions = dict(cfg["DerivativeDefinitions"])
    definitions["Bad name"] = {"for_dataframe": True, "nodes": []}
    cfg["DerivativeDefinitions"] = definitions
    with pytest.raises(NamingError, match="Bad name"):
        build_derivative_dataframe(cfg, include_derivatives=["Bad name"])


# ---------------------------------------------------------------------------
# Error columns
# ---------------------------------------------------------------------------


def test_error_column_wide_and_long(tmp_path):
    values = {"A": {"A.nc": 1.0}, "Fit_w-6": ValueError("boom")}
    row, long_rows = _collect_row(tmp_path, values, long=True)
    assert row["Fit_w-6.error"] == "boom"
    assert not any("__error" in c for c in row)
    assert [r["derivative"] for r in long_rows] == ["A.nc", "Fit_w-6.error"]
    assert set(long_rows[0]) == {"index", "dataset", "file_path", "derivative", "value"}
    assert parse_column("Fit_w-6.error").is_error


def test_long_format_columns_and_parse_back(dummy_pipeline):
    df = build_derivative_dataframe(
        dummy_pipeline["config"], output_format="long", raise_on_error=True
    )
    assert list(df.columns) == ["index", "dataset", "file_path", "derivative", "value"]
    assert len(df) > 0
    for derivative in df["derivative"]:
        parsed = parse_column(derivative)
        assert parsed is not None
        assert not parsed.is_error


# ---------------------------------------------------------------------------
# Documentation recipe
# ---------------------------------------------------------------------------


def _docs_recipe() -> str:
    text = (DOCS / "dataframe_assembly.md").read_text(encoding="utf-8")
    match = re.search(
        r"<!-- tested by tests/test_dataframe_columns.py::test_docs_split_recipe -->\s*"
        r"```python\n(.*?)```",
        text,
        re.S,
    )
    assert match, "split recipe not found in dataframe_assembly.md"
    return match.group(1)


def test_docs_split_recipe():
    df = pd.DataFrame(
        {
            "index": [0, 0, 0, 0],
            "dataset": ["demo"] * 4,
            "file_path": ["sub-01.vhdr"] * 4,
            "derivative": [
                "Fit_alpha-0p5_w-6.nc@band-8-12_channel-EEG%20001",
                "Fit_alpha-0p5_w-6.nc@band-alpha_channel-Fp1",
                "BandPower.nc@freq--1.5",
                "Fit_alpha-0p5_w-6.error",
            ],
            "value": [1.0, 2.0, 3.0, "boom"],
        }
    )
    namespace = {"df": df}
    exec(_docs_recipe(), namespace)
    out = namespace["df"]
    assert list(out["name"]) == ["Fit_alpha-0p5_w-6"] * 2 + ["BandPower", "Fit_alpha-0p5_w-6"]
    assert list(out["suffix"]) == [".nc", ".nc", ".nc", ".error"]
    assert list(out["band"][:2]) == ["8-12", "alpha"]
    assert list(out["channel"][:2]) == ["EEG 001", "Fp1"]
    assert out["freq"][2] == "-1.5"
    assert out["w"].isna().tolist() == [False, False, True, False]
    assert out["w"][0] == "6"
    assert out["alpha"][0] == Decimal("0.5")
    assert pd.isna(out["alpha"][2])
