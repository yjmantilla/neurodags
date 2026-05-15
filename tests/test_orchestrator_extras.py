"""Extra coverage tests for orchestrators.py — branches not covered by integration tests."""
from __future__ import annotations

import pandas as pd
import pytest

from neurodags.definitions import Artifact, NodeResult
from neurodags.orchestrators import (
    _resolve_reference_base,
    _sanitize_reference_base,
    build_derivative_dataframe,
    iterate_derivative_pipeline,
)


# ---------------------------------------------------------------------------
# _sanitize_reference_base — unit tests
# ---------------------------------------------------------------------------

def test_sanitize_reference_base_replaces_at_in_filename():
    result = _sanitize_reference_base("/derivatives/sub-01.vhdr@BasicPrep.fif")
    assert result == "/derivatives/sub-01.vhdr&BasicPrep.fif"


def test_sanitize_reference_base_multiple_at_signs():
    # double-@ (already-chained path): both replaced
    result = _sanitize_reference_base("/derivatives/sub-01.vhdr@First@Second.fif")
    assert result == "/derivatives/sub-01.vhdr&First&Second.fif"


def test_sanitize_reference_base_no_at_unchanged():
    path = "/derivatives/sub-01.vhdr"
    assert _sanitize_reference_base(path) == path


def test_sanitize_reference_base_preserves_directory_components():
    # @ in directory path must NOT be touched
    result = _sanitize_reference_base("/path/with@dir/sub-01@BasicPrep.fif")
    assert result == "/path/with@dir/sub-01&BasicPrep.fif"


def test_sanitize_reference_base_no_directory():
    assert _sanitize_reference_base("sub-01@BasicPrep.fif") == "sub-01&BasicPrep.fif"


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


def test_resolve_reference_base_sanitizes_at_no_derivatives_path(tmp_path):
    """Source filename with @ (neurodags derivative used as input) is sanitized."""
    from neurodags.definitions import DatasetConfig

    ds_config = DatasetConfig(
        name="test",
        file_pattern=str(tmp_path / "**/*.fif"),
        derivatives_path=None,
    )
    # Simulate a neurodags derivative file used as a new source
    file_path = str(tmp_path / "sub-01" / "sub-01.vhdr@BasicPrep.fif")
    ref_str, ref_path = _resolve_reference_base(file_path, ds_config, None, None)
    assert "@" not in ref_str
    assert "&" in ref_str
    assert ref_str == str(tmp_path / "sub-01" / "sub-01.vhdr&BasicPrep.fif")
    assert ref_path == tmp_path / "sub-01" / "sub-01.vhdr&BasicPrep.fif"


def test_resolve_reference_base_sanitizes_at_with_derivatives_path(tmp_path):
    """@ in source filename is sanitized when a separate derivatives_path is set."""
    from neurodags.definitions import DatasetConfig

    raw_dir = tmp_path / "rawdata"
    deriv_dir = tmp_path / "derivatives"
    deriv_dir.mkdir()

    ds_config = DatasetConfig(
        name="test",
        file_pattern=str(raw_dir / "**/*.fif"),
        derivatives_path=str(deriv_dir),
    )
    file_path = str(raw_dir / "sub-01" / "sub-01.vhdr@BasicPrep.fif")
    ref_str, ref_path = _resolve_reference_base(file_path, ds_config, str(raw_dir), None)
    assert "@" not in ref_str
    assert "&" in ref_str
    # derivative goes into deriv_dir with sanitized filename
    assert ref_str.startswith(str(deriv_dir))
    assert "sub-01.vhdr&BasicPrep.fif" in ref_str


# ---------------------------------------------------------------------------
# iterate_derivative_pipeline — derivative type checks
# ---------------------------------------------------------------------------

def test_iterate_pipeline_unknown_derivative_raises(dummy_pipeline):
    cfg = dummy_pipeline["config"]
    with pytest.raises(KeyError, match="Unknown derivative or node"):
        iterate_derivative_pipeline(cfg, "NonExistentDerivative")


def test_iterate_pipeline_derivative_not_in_derivative_list_raises(dummy_pipeline):
    cfg = {
        **dummy_pipeline["config"],
        "DerivativeList": ["BasicPrep"],
    }
    with pytest.raises(KeyError, match="not enabled in DerivativeList"):
        iterate_derivative_pipeline(cfg, "Spectrum")


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


def test_build_dataframe_missing_for_dataframe_defaults_to_excluded(dummy_pipeline):
    cfg = {
        **dummy_pipeline["config"],
        "DerivativeDefinitions": {
            "Implicit": {"nodes": [{"id": 0, "derivative": "SourceFile"}]},
        },
        "DerivativeList": ["Implicit"],
    }
    result = build_derivative_dataframe(cfg)
    assert isinstance(result, pd.DataFrame)
    assert "index" in result.columns
    assert len(result) == 0


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


def test_iterate_pipeline_new_definitions_relative_to_pipeline_yaml(dummy_pipeline, tmp_path):
    """new_definitions relative path resolved against pipeline yaml, not cwd."""
    import yaml
    from neurodags.nodes.preprocessing import basic_preprocessing

    pipeline_dir = tmp_path / "pipeline_dir"
    pipeline_dir.mkdir()
    node_file = pipeline_dir / "extra_nodes.py"
    node_file.write_text("MY_EXTRA = True\n")

    cfg = {
        **dummy_pipeline["config"],
        "new_definitions": "extra_nodes.py",  # relative path
    }
    pipeline_yaml = pipeline_dir / "pipeline.yaml"
    pipeline_yaml.write_text(yaml.dump(cfg))

    # Must resolve relative to pipeline_dir, not cwd
    iterate_derivative_pipeline(str(pipeline_yaml), basic_preprocessing, max_files_per_dataset=1, raise_on_error=True)


def test_get_datasets_relative_to_pipeline_path(tmp_path):
    from neurodags.datasets import get_datasets_and_mount_point_from_pipeline_configuration

    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    datasets_path = config_dir / "datasets.yml"
    datasets_path.write_text(
        "demo:\n"
        "  name: Demo\n"
        f"  file_pattern: {tmp_path}/**/*.vhdr\n"
        f"  derivatives_path: {tmp_path}/derivatives\n"
    )
    pipeline_path = config_dir / "pipeline.yml"
    pipeline_path.write_text("datasets: datasets.yml\nmount_point: null\n")

    datasets, mount_point = get_datasets_and_mount_point_from_pipeline_configuration(pipeline_path)
    assert mount_point is None
    assert "demo" in datasets
