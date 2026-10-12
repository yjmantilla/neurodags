"""Extra coverage tests for orchestrators.py — branches not covered by integration tests."""

from __future__ import annotations

import pandas as pd
import pytest

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
        source_root=str(tmp_path),
        file_pattern=str(tmp_path / "**/*.vhdr"),
        derivatives_path=None,
    )
    file_path = str(tmp_path / "sub-01" / "file.vhdr")
    ref_str, _ref_path = _resolve_reference_base(file_path, ds_config, None, None)
    assert ref_str == file_path


def test_resolve_reference_base_sanitizes_at_no_derivatives_path(tmp_path):
    """Source filename with @ (neurodags derivative used as input) is sanitized."""
    from neurodags.definitions import DatasetConfig

    ds_config = DatasetConfig(
        name="test",
        source_root=str(tmp_path),
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
        source_root=str(raw_dir),
        file_pattern=str(raw_dir / "**/*.fif"),
        derivatives_path=str(deriv_dir),
    )
    file_path = str(raw_dir / "sub-01" / "sub-01.vhdr@BasicPrep.fif")
    ref_str, _ref_path = _resolve_reference_base(file_path, ds_config, str(raw_dir), None)
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
    iterate_derivative_pipeline(
        cfg, basic_preprocessing, max_files_per_dataset=1, raise_on_error=True
    )


# ---------------------------------------------------------------------------
# iterate_derivative_pipeline — n_jobs handling
# ---------------------------------------------------------------------------


def test_iterate_pipeline_n_jobs_zero_maps_to_serial(dummy_pipeline):
    cfg = dummy_pipeline["config"]
    iterate_derivative_pipeline(
        cfg, "BasicPrep", n_jobs=0, max_files_per_dataset=1, raise_on_error=True
    )


def test_iterate_pipeline_n_jobs_parallel(dummy_pipeline):
    cfg = dummy_pipeline["config"]
    iterate_derivative_pipeline(
        cfg, "BasicPrep", n_jobs=2, max_files_per_dataset=1, raise_on_error=True
    )


# ---------------------------------------------------------------------------
# iterate_derivative_pipeline — only_index with unknown indices is an error (0.4.0)
# ---------------------------------------------------------------------------


def test_iterate_pipeline_only_index_unknown_raises(dummy_pipeline):
    cfg = dummy_pipeline["config"]
    with pytest.raises(ValueError, match="unknown: 9999"):
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
        "naming_version": 1,
        "datasets": {
            "empty": {
                "name": "Empty",
                "source_root": str(tmp_path),
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
        "naming_version": 1,
        "datasets": {
            "empty": {
                "name": "Empty",
                "source_root": str(tmp_path),
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
# build_derivative_dataframe — only_index with unknown index is an error (0.4.0)
# ---------------------------------------------------------------------------


def test_build_dataframe_only_index_unknown_raises(dummy_pipeline):
    cfg = dummy_pipeline["config"]
    with pytest.raises(ValueError, match="unknown: 9999"):
        build_derivative_dataframe(cfg, only_index=[9999])


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
    iterate_derivative_pipeline(
        cfg, basic_preprocessing, max_files_per_dataset=1, raise_on_error=True
    )


def test_iterate_pipeline_new_definitions_list(dummy_pipeline, tmp_path):
    from neurodags.nodes.preprocessing import basic_preprocessing

    node_file = tmp_path / "extra_nodes2.py"
    node_file.write_text("MY_EXTRA2 = True\n")

    cfg = {
        **dummy_pipeline["config"],
        "new_definitions": [str(node_file)],
    }
    iterate_derivative_pipeline(
        cfg, basic_preprocessing, max_files_per_dataset=1, raise_on_error=True
    )


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
    iterate_derivative_pipeline(
        str(pipeline_yaml), basic_preprocessing, max_files_per_dataset=1, raise_on_error=True
    )


def test_get_datasets_relative_to_pipeline_path(tmp_path):
    from neurodags.datasets import get_datasets_and_mount_point_from_pipeline_configuration

    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    datasets_path = config_dir / "datasets.yml"
    datasets_path.write_text(
        "demo:\n"
        "  name: Demo\n"
        f"  source_root: {tmp_path}\n"
        f"  file_pattern: {tmp_path}/**/*.vhdr\n"
        f"  derivatives_path: {tmp_path}/derivatives\n"
    )
    pipeline_path = config_dir / "pipeline.yml"
    pipeline_path.write_text("naming_version: 1\ndatasets: datasets.yml\nmount_point: null\n")

    datasets, mount_point = get_datasets_and_mount_point_from_pipeline_configuration(pipeline_path)
    assert mount_point is None
    assert "demo" in datasets


# ---------------------------------------------------------------------------
# Output reference base: inspection is read-only (#33), single-file layout (#34)
# ---------------------------------------------------------------------------

_META_PIPELINE = {
    "naming_version": 1,
    "datasets": "ds.yml",
    "mount_point": None,
    "DerivativeDefinitions": {
        "Meta": {
            "save": True,
            "for_dataframe": True,
            "nodes": [
                {"id": 0, "derivative": "SourceFile"},
                {"id": 1, "node": "extract_meeg_metadata", "args": {"mne_object": "id.0"}},
            ],
        }
    },
    "DerivativeList": ["Meta"],
}


def _write_fif_project(root, relpaths):
    """Write tiny raw FIF files at *relpaths* under ``root/in`` plus pipeline/dataset YAMLs.

    Derivatives go to ``root/out/derivatives`` (which does not exist beforehand).
    """
    import mne
    import numpy as np
    import yaml

    root.mkdir(parents=True, exist_ok=True)
    for i, rel in enumerate(relpaths):
        target = root / "in" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        info = mne.create_info(["a", "b"], 100.0, "eeg")
        raw = mne.io.RawArray(np.random.default_rng(i).standard_normal((2, 500)) * 1e-5, info)
        raw.save(target, verbose="error")
    ds = {
        "d": {
            "name": "d",
            "source_root": str(root),
            "file_pattern": str(root / "in" / "**" / "*_raw.fif"),
            "derivatives_path": str(root / "out" / "derivatives"),
        }
    }
    (root / "ds.yml").write_text(yaml.safe_dump(ds))
    pipe = root / "pl.yml"
    pipe.write_text(yaml.safe_dump(_META_PIPELINE))
    return pipe


def _tree(root):
    return sorted(str(p.relative_to(root)) for p in root.rglob("*"))


@pytest.mark.parametrize(
    "command",
    [["status"], ["status", "--format", "json"], ["dry-run"], ["count-inputs"]],
)
@pytest.mark.parametrize("n_files", [1, 2])
def test_inspection_commands_do_not_create_output_dirs(
    tmp_path, monkeypatch, capsys, command, n_files
):
    """#33: status / dry-run / count-inputs must not write to the filesystem."""
    from neurodags.cli import main

    proj = tmp_path / "proj"
    pipe = _write_fif_project(proj, ["sub-01/a_raw.fif", "sub-02/b_raw.fif"][:n_files])
    # dry-run saves its CSV in cwd; keep cwd outside the tree under inspection.
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    before = _tree(proj)

    rc = main([command[0], str(pipe), *command[1:]])
    # status exits 1 when derivatives are missing (nothing has been computed).
    assert rc == (1 if command[0] == "status" else 0)

    assert _tree(proj) == before
    assert not (proj / "out").exists()


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_dry_run_api_does_not_create_output_dirs(tmp_path, n_jobs):
    """#33: iterate_derivative_pipeline(dry_run=True) is read-only."""
    pipe = _write_fif_project(tmp_path, ["sub-01/a_raw.fif", "sub-02/b_raw.fif"])
    plan = iterate_derivative_pipeline(str(pipe), "Meta", dry_run=True, n_jobs=n_jobs)
    assert isinstance(plan, pd.DataFrame)
    assert len(plan) == 2
    assert not (tmp_path / "out").exists()


def test_dataframe_creates_dirs_only_when_computing(tmp_path):
    """#33: dataframe collection creates output dirs on demand when it must compute."""
    pipe = _write_fif_project(tmp_path, ["sub-01/ses-1/a_raw.fif", "sub-02/ses-1/b_raw.fif"])
    derivs = tmp_path / "out" / "derivatives"

    # Only precomputed (save=True, for_dataframe=True) derivatives are read and
    # nothing is cached: no computation, so nothing may be created.
    df = build_derivative_dataframe(str(pipe))
    assert len(df) == 2
    assert not (tmp_path / "out").exists()

    # A save=False derivative is computed on the fly; its nested dirs are created
    # only then (nodes may write next to reference_base).
    import yaml

    config = yaml.safe_load(pipe.read_text())
    config["DerivativeDefinitions"]["Meta"]["save"] = False
    pipe.write_text(yaml.safe_dump(config))
    df = build_derivative_dataframe(str(pipe), raise_on_error=True)
    assert len(df) == 2
    assert any(c.startswith("Meta") for c in df.columns)
    assert (derivs / "sub-01" / "ses-1").is_dir()
    assert (derivs / "sub-02" / "ses-1").is_dir()


def test_real_run_still_creates_nested_output_dirs(tmp_path):
    """#33 guard: non-dry runs still create derivatives_path and nested subdirs."""
    from neurodags.orchestrators import run_pipeline

    pipe = _write_fif_project(tmp_path, ["sub-01/ses-1/a_raw.fif", "sub-02/ses-1/b_raw.fif"])
    run_pipeline(str(pipe), raise_on_error=True)
    derivs = tmp_path / "out" / "derivatives"
    assert (derivs / "code" / "pl.yml").exists()
    assert (derivs / "sub-01" / "ses-1" / "a_raw.fif@Meta.json").exists()
    assert (derivs / "sub-02" / "ses-1" / "b_raw.fif@Meta.json").exists()


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_single_file_dataset_outputs_land_inside_derivatives(tmp_path, n_jobs):
    """#34: a one-file dataset writes <derivatives>/<file>@Meta.json, not a sibling."""
    from neurodags.orchestrators import run_pipeline

    pipe = _write_fif_project(tmp_path, ["sub-01/only_raw.fif"])
    run_pipeline(str(pipe), raise_on_error=True, n_jobs=n_jobs)

    out = tmp_path / "out"
    assert sorted(p.name for p in out.iterdir()) == ["derivatives"]
    assert (out / "derivatives" / "only_raw.fif@Meta.json").is_file()

    # Dry-run plan and dataframe find the same cached output.
    plan = iterate_derivative_pipeline(str(pipe), "Meta", dry_run=True)
    final = plan["plan"].iloc[0][0]
    assert final["cached"] is True
    assert final["prefix"] == str(out / "derivatives" / "only_raw.fif@Meta")
    df = build_derivative_dataframe(str(pipe), raise_on_error=True)
    assert len(df) == 1
    assert any(c.startswith("Meta") for c in df.columns)


def test_resolve_reference_base_file_as_common_root(tmp_path):
    """#34 guard: a common_root equal to the file itself still yields a path inside."""
    from neurodags.definitions import DatasetConfig

    deriv = tmp_path / "derivatives"
    ds_config = DatasetConfig(
        name="t",
        source_root=str(tmp_path),
        file_pattern=str(tmp_path / "*.fif"),
        derivatives_path=str(deriv),
    )
    src = str(tmp_path / "in" / "x_raw.fif")
    ref_str, _ = _resolve_reference_base(src, ds_config, src, None)
    assert ref_str == str(deriv / "x_raw.fif")


def test_multi_file_reference_bases_unchanged(tmp_path):
    """Cache compatibility (#34): multi-file reference bases are byte-for-byte unchanged."""
    from neurodags.definitions import DatasetConfig
    from neurodags.iterators import get_all_files_across_datasets

    _write_fif_project(
        tmp_path, ["sub-01/ses-1/a_raw.fif", "sub-02/ses-1/b_raw.fif", "sub-02/ses-2/c_raw.fif"]
    )
    deriv = tmp_path / "out" / "derivatives"
    ds_config = DatasetConfig(
        name="d",
        source_root=str(tmp_path),
        file_pattern=str(tmp_path / "in" / "**" / "*_raw.fif"),
        derivatives_path=str(deriv),
    )
    files_per_dataset, _, roots = get_all_files_across_datasets({"d": ds_config})
    assert roots["d"] == str(tmp_path / "in")
    refs = sorted(
        _resolve_reference_base(f, ds_config, roots["d"], None)[0] for f in files_per_dataset["d"]
    )
    assert refs == [
        f"{deriv}/sub-01/ses-1/a_raw.fif",
        f"{deriv}/sub-02/ses-1/b_raw.fif",
        f"{deriv}/sub-02/ses-2/c_raw.fif",
    ]
