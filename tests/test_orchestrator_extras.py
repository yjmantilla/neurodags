"""Extra coverage tests for orchestrators.py — branches not covered by integration tests."""
from __future__ import annotations

import pandas as pd
import pytest

from neurodags.definitions import Artifact, NodeResult
from neurodags.orchestrators import (
    _resolve_reference_base,
    build_derivative_dataframe,
    iterate_derivative_pipeline,
)


# ---------------------------------------------------------------------------
# _resolve_reference_base — no derivatives_path branch
# ---------------------------------------------------------------------------

def test_resolve_reference_base_no_derivatives_path(tmp_path):
    from neurodags.definitions import DatasetConfig

    ds_config = DatasetConfig(
        name="test",
        file_pattern=str(tmp_path / "**/*.vhdr"),
        derivatives_path=None,
    )
    file_path = str(tmp_path / "sub-01" / "file.vhdr")
    ref_str, ref_path = _resolve_reference_base(file_path, ds_config, None, None)
    assert ref_str == file_path


# ---------------------------------------------------------------------------
# iterate_derivative_pipeline — derivative type checks
# ---------------------------------------------------------------------------

def test_iterate_pipeline_unknown_derivative_raises(dummy_pipeline):
    cfg = dummy_pipeline["config"]
    with pytest.raises(KeyError, match="Unknown derivative or node"):
        iterate_derivative_pipeline(cfg, "NonExistentDerivative")


def test_iterate_pipeline_invalid_type_raises(dummy_pipeline):
    cfg = dummy_pipeline["config"]
    with pytest.raises(TypeError, match="derivative must be"):
        iterate_derivative_pipeline(cfg, 42)


def test_iterate_pipeline_callable_derivative(dummy_pipeline):
    from neurodags.nodes.preprocessing import basic_preprocessing

    cfg = dummy_pipeline["config"]
    iterate_derivative_pipeline(cfg, basic_preprocessing, max_files_per_dataset=1, raise_on_error=True)


# ---------------------------------------------------------------------------
# iterate_derivative_pipeline — n_jobs handling
# ---------------------------------------------------------------------------

def test_iterate_pipeline_n_jobs_zero_maps_to_serial(dummy_pipeline):
    cfg = dummy_pipeline["config"]
    iterate_derivative_pipeline(cfg, "BasicPrep", n_jobs=0, max_files_per_dataset=1, raise_on_error=True)


def test_iterate_pipeline_n_jobs_parallel(dummy_pipeline):
    cfg = dummy_pipeline["config"]
    iterate_derivative_pipeline(
        cfg, "BasicPrep", n_jobs=2, max_files_per_dataset=1, raise_on_error=True
    )


# ---------------------------------------------------------------------------
# iterate_derivative_pipeline — only_index with missing indices (warning path)
# ---------------------------------------------------------------------------

def test_iterate_pipeline_only_index_missing_warns(dummy_pipeline):
    cfg = dummy_pipeline["config"]
    iterate_derivative_pipeline(cfg, "BasicPrep", only_index=[9999], raise_on_error=False)


# ---------------------------------------------------------------------------
# iterate_derivative_pipeline — raise_on_error with a failing node
# ---------------------------------------------------------------------------

def test_iterate_pipeline_raise_on_error(dummy_pipeline):
    cfg = dummy_pipeline["config"]

    def bad_node(file_path):
        raise RuntimeError("intentional failure")

    with pytest.raises(RuntimeError):
        iterate_derivative_pipeline(cfg, bad_node, raise_on_error=True, max_files_per_dataset=1)


# ---------------------------------------------------------------------------
# iterate_derivative_pipeline — empty jobs (no files found)
# ---------------------------------------------------------------------------

def test_iterate_pipeline_empty_dataset(tmp_path):
    cfg = {
        "datasets": {
            "empty": {
                "name": "Empty",
                "file_pattern": str(tmp_path / "**/*.vhdr"),
                "derivatives_path": str(tmp_path / "deriv"),
            }
        },
        "DerivativeDefinitions": {
            "BasicPrep": {
                "overwrite": False,
                "nodes": [{"id": 0, "derivative": "SourceFile"}],
            }
        },
        "DerivativeList": ["BasicPrep"],
    }

    def my_node(file_path):
        pass

    result = iterate_derivative_pipeline(cfg, my_node)
    assert result is None


def test_iterate_pipeline_empty_dataset_dry_run(tmp_path):
    cfg = {
        "datasets": {
            "empty": {
                "name": "Empty",
                "file_pattern": str(tmp_path / "**/*.vhdr"),
                "derivatives_path": str(tmp_path / "deriv"),
            }
        },
        "DerivativeDefinitions": {},
        "DerivativeList": [],
    }

    def my_node(file_path):
        pass

    result = iterate_derivative_pipeline(cfg, my_node, dry_run=True)
    assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# build_derivative_dataframe — no DerivativeDefinitions
# ---------------------------------------------------------------------------

def test_build_dataframe_no_derivative_definitions(dummy_pipeline):
    cfg = {k: v for k, v in dummy_pipeline["config"].items() if k != "DerivativeDefinitions"}
    result = build_derivative_dataframe(cfg)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0


# ---------------------------------------------------------------------------
# build_derivative_dataframe — no eligible derivatives
# ---------------------------------------------------------------------------

def test_build_dataframe_no_eligible_derivatives(dummy_pipeline):
    cfg = {
        **dummy_pipeline["config"],
        "DerivativeDefinitions": {
            "NoSave": {"for_dataframe": False, "nodes": [{"id": 0, "derivative": "SourceFile"}]}
        },
        "DerivativeList": ["NoSave"],
    }
    result = build_derivative_dataframe(cfg)
    assert isinstance(result, pd.DataFrame)
    assert "index" in result.columns


# ---------------------------------------------------------------------------
# build_derivative_dataframe — include_derivatives with missing derivative
# ---------------------------------------------------------------------------

def test_build_dataframe_include_nonexistent_derivative(dummy_pipeline):
    cfg = dummy_pipeline["config"]
    iterate_derivative_pipeline(cfg, "BasicPrep", raise_on_error=True)
    iterate_derivative_pipeline(cfg, "Spectrum", raise_on_error=True)

    result = build_derivative_dataframe(cfg, include_derivatives=["NonExistent"])
    assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# build_derivative_dataframe — only_index with missing index (warning path)
# ---------------------------------------------------------------------------

def test_build_dataframe_only_index_missing(dummy_pipeline):
    cfg = dummy_pipeline["config"]
    iterate_derivative_pipeline(cfg, "BasicPrep", raise_on_error=True)
    iterate_derivative_pipeline(cfg, "Spectrum", raise_on_error=True)

    result = build_derivative_dataframe(cfg, only_index=[9999])
    assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# iterate_derivative_pipeline — new_definitions (str path)
# ---------------------------------------------------------------------------

def test_iterate_pipeline_new_definitions_str(dummy_pipeline, tmp_path):
    from neurodags.nodes.preprocessing import basic_preprocessing

    node_file = tmp_path / "extra_nodes.py"
    node_file.write_text("MY_EXTRA = True\n")

    cfg = {
        **dummy_pipeline["config"],
        "new_definitions": str(node_file),
    }
    iterate_derivative_pipeline(cfg, basic_preprocessing, max_files_per_dataset=1, raise_on_error=True)


def test_iterate_pipeline_new_definitions_list(dummy_pipeline, tmp_path):
    from neurodags.nodes.preprocessing import basic_preprocessing

    node_file = tmp_path / "extra_nodes2.py"
    node_file.write_text("MY_EXTRA2 = True\n")

    cfg = {
        **dummy_pipeline["config"],
        "new_definitions": [str(node_file)],
    }
    iterate_derivative_pipeline(cfg, basic_preprocessing, max_files_per_dataset=1, raise_on_error=True)


def test_iterate_pipeline_new_definitions_invalid_type_raises(dummy_pipeline):
    from neurodags.nodes.preprocessing import basic_preprocessing

    cfg = {
        **dummy_pipeline["config"],
        "new_definitions": 42,
    }
    with pytest.raises(TypeError, match="new_definitions must be"):
        iterate_derivative_pipeline(cfg, basic_preprocessing, max_files_per_dataset=1)
