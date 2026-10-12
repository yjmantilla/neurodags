"""End-to-end integration tests for iterate_derivative_pipeline and build_derivative_dataframe."""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from neurodags.orchestrators import build_derivative_dataframe, iterate_derivative_pipeline

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _output_files(out_dir, pattern):
    return list(out_dir.rglob(pattern))


# ---------------------------------------------------------------------------
# Basic execution
# ---------------------------------------------------------------------------


def test_iterate_pipeline_produces_fif_outputs(dummy_pipeline):
    cfg = dummy_pipeline["config"]
    out_dir = dummy_pipeline["out_dir"]

    iterate_derivative_pipeline(cfg, "BasicPrep", raise_on_error=True)

    fif_files = _output_files(out_dir, "*@BasicPrep.fif")
    assert len(fif_files) > 0, "expected .fif derivatives to be written"


def test_iterate_pipeline_produces_nc_outputs(dummy_pipeline):
    cfg = dummy_pipeline["config"]
    out_dir = dummy_pipeline["out_dir"]

    iterate_derivative_pipeline(cfg, "BasicPrep", raise_on_error=True)
    iterate_derivative_pipeline(cfg, "Spectrum", raise_on_error=True)

    nc_files = _output_files(out_dir, "*@Spectrum.nc")
    assert len(nc_files) > 0, "expected .nc derivatives to be written"


def test_iterate_pipeline_one_output_per_input(dummy_pipeline):
    cfg = dummy_pipeline["config"]
    out_dir = dummy_pipeline["out_dir"]
    data_dir = dummy_pipeline["data_dir"]

    source_files = list(data_dir.rglob("*.vhdr"))
    iterate_derivative_pipeline(cfg, "BasicPrep", raise_on_error=True)

    fif_files = _output_files(out_dir, "*@BasicPrep.fif")
    assert len(fif_files) == len(source_files)


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


def test_iterate_pipeline_second_run_skips_cached(dummy_pipeline, tmp_path):
    cfg = dummy_pipeline["config"]
    out_dir = dummy_pipeline["out_dir"]

    iterate_derivative_pipeline(cfg, "BasicPrep", raise_on_error=True)
    fif_files_first = _output_files(out_dir, "*@BasicPrep.fif")
    mtimes_first = {f: f.stat().st_mtime for f in fif_files_first}

    iterate_derivative_pipeline(cfg, "BasicPrep", raise_on_error=True)
    fif_files_second = _output_files(out_dir, "*@BasicPrep.fif")
    mtimes_second = {f: f.stat().st_mtime for f in fif_files_second}

    for fpath, mtime in mtimes_first.items():
        assert mtimes_second[fpath] == mtime, f"{fpath.name} was rewritten on second run"


def test_iterate_pipeline_overwrite_forces_recompute(dummy_pipeline):
    cfg = dummy_pipeline["config"]
    out_dir = dummy_pipeline["out_dir"]

    iterate_derivative_pipeline(cfg, "BasicPrep", raise_on_error=True)
    fif_files_first = _output_files(out_dir, "*@BasicPrep.fif")
    mtimes_first = {f: f.stat().st_mtime for f in fif_files_first}

    time.sleep(0.05)

    cfg_overwrite = {**cfg}
    cfg_overwrite["DerivativeDefinitions"] = {
        **cfg["DerivativeDefinitions"],
        "BasicPrep": {**cfg["DerivativeDefinitions"]["BasicPrep"], "overwrite": True},
    }
    iterate_derivative_pipeline(cfg_overwrite, "BasicPrep", raise_on_error=True)
    fif_files_second = _output_files(out_dir, "*@BasicPrep.fif")
    mtimes_second = {f: f.stat().st_mtime for f in fif_files_second}

    rewritten = [f for f in fif_files_first if mtimes_second[f] != mtimes_first[f]]
    assert len(rewritten) > 0, "expected at least one file rewritten with overwrite=True"


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------


def test_iterate_pipeline_dry_run_returns_dataframe(dummy_pipeline):
    import pandas as pd

    cfg = dummy_pipeline["config"]
    plan = iterate_derivative_pipeline(cfg, "BasicPrep", dry_run=True)
    assert isinstance(plan, pd.DataFrame)
    assert len(plan) > 0


def test_iterate_pipeline_dry_run_writes_no_files(dummy_pipeline):
    cfg = dummy_pipeline["config"]
    out_dir = dummy_pipeline["out_dir"]

    iterate_derivative_pipeline(cfg, "BasicPrep", dry_run=True)

    fif_files = _output_files(out_dir, "*@BasicPrep.fif")
    assert len(fif_files) == 0, "dry_run must not write files"


# ---------------------------------------------------------------------------
# only_index / max_files_per_dataset
# ---------------------------------------------------------------------------


def test_iterate_pipeline_only_index_limits_output(dummy_pipeline):
    cfg = dummy_pipeline["config"]
    out_dir = dummy_pipeline["out_dir"]
    data_dir = dummy_pipeline["data_dir"]

    source_files = list(data_dir.rglob("*.vhdr"))
    if len(source_files) < 2:
        pytest.skip("need at least 2 source files for this test")

    iterate_derivative_pipeline(cfg, "BasicPrep", only_index=0, raise_on_error=True)
    fif_files = _output_files(out_dir, "*@BasicPrep.fif")
    assert len(fif_files) == 1


def test_iterate_pipeline_max_files_per_dataset(dummy_pipeline):
    cfg = dummy_pipeline["config"]
    out_dir = dummy_pipeline["out_dir"]
    data_dir = dummy_pipeline["data_dir"]

    source_files = list(data_dir.rglob("*.vhdr"))
    if len(source_files) < 2:
        pytest.skip("need at least 2 source files for this test")

    iterate_derivative_pipeline(cfg, "BasicPrep", max_files_per_dataset=1, raise_on_error=True)
    fif_files = _output_files(out_dir, "*@BasicPrep.fif")
    assert len(fif_files) == 1


# ---------------------------------------------------------------------------
# Dataframe assembly
# ---------------------------------------------------------------------------


def test_build_derivative_dataframe_wide(dummy_pipeline):
    cfg = dummy_pipeline["config"]

    iterate_derivative_pipeline(cfg, "BasicPrep", raise_on_error=True)
    iterate_derivative_pipeline(cfg, "Spectrum", raise_on_error=True)

    df = build_derivative_dataframe(cfg, output_format="wide")
    assert len(df) > 0


def test_build_derivative_dataframe_wide_rows_match_files(dummy_pipeline):
    cfg = dummy_pipeline["config"]
    data_dir = dummy_pipeline["data_dir"]

    iterate_derivative_pipeline(cfg, "BasicPrep", raise_on_error=True)
    iterate_derivative_pipeline(cfg, "Spectrum", raise_on_error=True)

    source_count = len(list(data_dir.rglob("*.vhdr")))
    df = build_derivative_dataframe(cfg, output_format="wide")
    assert len(df) == source_count


def test_build_derivative_dataframe_long(dummy_pipeline):
    cfg = dummy_pipeline["config"]

    iterate_derivative_pipeline(cfg, "BasicPrep", raise_on_error=True)
    iterate_derivative_pipeline(cfg, "Spectrum", raise_on_error=True)

    df = build_derivative_dataframe(cfg, output_format="long")
    assert len(df) > 0


# ---------------------------------------------------------------------------
# Multi-step chain (save=False derivative is computed, not saved)
# ---------------------------------------------------------------------------


def test_save_false_derivative_not_on_disk(dummy_pipeline):
    cfg = dummy_pipeline["config"]
    out_dir = dummy_pipeline["out_dir"]

    iterate_derivative_pipeline(cfg, "BasicPrep", raise_on_error=True)
    iterate_derivative_pipeline(cfg, "Spectrum", raise_on_error=True)
    iterate_derivative_pipeline(cfg, "BandPowerMean", raise_on_error=True)

    band_files = _output_files(out_dir, "*@BandPowerMean.nc")
    assert len(band_files) == 0, "save=False derivative must not write files"


def test_for_dataframe_derivative_appears_in_df(dummy_pipeline):
    cfg = dummy_pipeline["config"]

    iterate_derivative_pipeline(cfg, "BasicPrep", raise_on_error=True)
    iterate_derivative_pipeline(cfg, "Spectrum", raise_on_error=True)

    df = build_derivative_dataframe(cfg, include_derivatives=["BandPowerMean"])
    assert len(df) > 0


def test_build_derivative_dataframe_parallel_matches_serial(dummy_pipeline):
    cfg = dummy_pipeline["config"]

    iterate_derivative_pipeline(cfg, "BasicPrep", raise_on_error=True)
    iterate_derivative_pipeline(cfg, "Spectrum", raise_on_error=True)

    serial = build_derivative_dataframe(cfg, output_format="wide", n_jobs=1)
    parallel = build_derivative_dataframe(cfg, output_format="wide", n_jobs=2)

    assert len(parallel) == len(serial)
    assert set(parallel.columns) == set(serial.columns)
    # file_path sets must match
    assert set(parallel["file_path"]) == set(serial["file_path"])


def test_build_derivative_dataframe_parallel_long_matches_serial(dummy_pipeline):
    cfg = dummy_pipeline["config"]

    iterate_derivative_pipeline(cfg, "BasicPrep", raise_on_error=True)
    iterate_derivative_pipeline(cfg, "Spectrum", raise_on_error=True)

    serial = build_derivative_dataframe(cfg, output_format="long", n_jobs=1)
    parallel = build_derivative_dataframe(cfg, output_format="long", n_jobs=2)

    assert len(parallel) == len(serial)
    assert set(parallel["file_path"]) == set(serial["file_path"])


# ---------------------------------------------------------------------------
# Parallel workers resolve uncached inter-derivative references (issue #18)
# ---------------------------------------------------------------------------


def test_iterate_parallel_uncached_subderivative_chain(dummy_pipeline):
    # Spectrum depends on an uncached BasicPrep.fif. With n_jobs=2 the loky workers
    # start with an empty derivative registry; before the fix the worker raised
    # "Unknown derivative 'BasicPrep.fif'" instead of recomputing it. Nothing is
    # precomputed here, so the worker must resolve BasicPrep via the registry.
    cfg = dummy_pipeline["config"]
    out_dir = dummy_pipeline["out_dir"]
    iterate_derivative_pipeline(cfg, "Spectrum", n_jobs=2, raise_on_error=True)
    assert list(out_dir.rglob("*@Spectrum.nc"))


def test_dataframe_parallel_uncached_subderivative_chain(dummy_pipeline):
    # Same bug via the dataframe path: BandPowerMean -> Spectrum.nc -> BasicPrep.fif,
    # all uncached, collected with n_jobs=2.
    cfg = dummy_pipeline["config"]
    df = build_derivative_dataframe(
        cfg, include_derivatives=["BandPowerMean"], n_jobs=2, raise_on_error=True
    )
    assert len(df) > 0
    assert not [c for c in df.columns if c.endswith(".error")]


# ---------------------------------------------------------------------------
# Error handling in _collect_dataframe_file
# ---------------------------------------------------------------------------


def test_build_derivative_dataframe_derivative_error_captured(dummy_pipeline):
    """Derivative-level error is caught and stored as name.error column, row still returned."""
    cfg = dummy_pipeline["config"]

    iterate_derivative_pipeline(cfg, "BasicPrep", raise_on_error=True)
    iterate_derivative_pipeline(cfg, "Spectrum", raise_on_error=True)

    with patch(
        "neurodags.orchestrators.collect_derivative_for_dataframe",
        side_effect=ValueError("simulated derivative failure"),
    ):
        df = build_derivative_dataframe(cfg, output_format="wide")

    assert len(df) > 0
    assert "BandPowerMean.error" in df.columns
    assert not [c for c in df.columns if "__error" in c]


def test_build_derivative_dataframe_derivative_error_long_captured(dummy_pipeline):
    """Derivative-level error in long mode adds a derivative = "name.error" row."""
    cfg = dummy_pipeline["config"]

    iterate_derivative_pipeline(cfg, "BasicPrep", raise_on_error=True)
    iterate_derivative_pipeline(cfg, "Spectrum", raise_on_error=True)

    with patch(
        "neurodags.orchestrators.collect_derivative_for_dataframe",
        side_effect=ValueError("simulated derivative failure"),
    ):
        df = build_derivative_dataframe(cfg, output_format="long")

    assert len(df) > 0
    assert "BandPowerMean.error" in set(df["derivative"])


def test_build_derivative_dataframe_file_error_row_omitted(dummy_pipeline):
    """File-level error (outer except) causes row to be omitted from the result."""
    cfg = dummy_pipeline["config"]

    iterate_derivative_pipeline(cfg, "BasicPrep", raise_on_error=True)
    iterate_derivative_pipeline(cfg, "Spectrum", raise_on_error=True)

    with patch(
        "neurodags.orchestrators._build_reference_base",
        side_effect=RuntimeError("simulated file failure"),
    ):
        df = build_derivative_dataframe(cfg, output_format="wide")

    assert len(df) == 0


def test_build_derivative_dataframe_long_empty_when_no_values(dummy_pipeline):
    """Long format with no derivative values returns empty DataFrame with correct columns."""
    cfg = dummy_pipeline["config"]

    with patch(
        "neurodags.orchestrators.collect_derivative_for_dataframe",
        return_value={},
    ):
        df = build_derivative_dataframe(cfg, output_format="long")

    assert len(df) == 0
    assert "derivative" in df.columns


def test_build_derivative_dataframe_derivative_error_raise_on_error(dummy_pipeline):
    """raise_on_error=True re-raises derivative-level exceptions."""
    cfg = dummy_pipeline["config"]

    iterate_derivative_pipeline(cfg, "BasicPrep", raise_on_error=True)
    iterate_derivative_pipeline(cfg, "Spectrum", raise_on_error=True)

    with pytest.raises(ValueError, match="simulated derivative failure"):
        with patch(
            "neurodags.orchestrators.collect_derivative_for_dataframe",
            side_effect=ValueError("simulated derivative failure"),
        ):
            build_derivative_dataframe(cfg, output_format="wide", raise_on_error=True)


def test_build_derivative_dataframe_file_error_raise_on_error(dummy_pipeline):
    """raise_on_error=True re-raises file-level exceptions."""
    cfg = dummy_pipeline["config"]

    iterate_derivative_pipeline(cfg, "BasicPrep", raise_on_error=True)
    iterate_derivative_pipeline(cfg, "Spectrum", raise_on_error=True)

    with pytest.raises(RuntimeError, match="simulated file failure"):
        with patch(
            "neurodags.orchestrators._build_reference_base",
            side_effect=RuntimeError("simulated file failure"),
        ):
            build_derivative_dataframe(cfg, output_format="wide", raise_on_error=True)
