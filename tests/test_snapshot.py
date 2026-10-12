"""Tests for _snapshot_pipeline_config and its integration with run_pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from neurodags.definitions import DatasetConfig
from neurodags.orchestrators import _snapshot_pipeline_config, run_pipeline

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pipeline_yaml(
    tmp_path: Path,
    *,
    extra_config: dict | None = None,
    new_defs_name: str | None = None,
    datasets_name: str = "datasets.yml",
) -> tuple[Path, Path, Path | None]:
    """Write a minimal pipeline YAML + datasets YAML (+ optional new_defs file).

    Returns (pipeline_path, datasets_path, new_defs_path | None).
    """
    data_dir = tmp_path / "rawdata"
    data_dir.mkdir(parents=True, exist_ok=True)
    deriv_dir = tmp_path / "derivatives"
    deriv_dir.mkdir(parents=True, exist_ok=True)

    datasets_cfg = {
        "test_ds": {
            "name": "TestDS",
            "source_root": str(data_dir),
            "file_pattern": str(data_dir / "**/*.vhdr"),
            "derivatives_path": str(deriv_dir),
        }
    }
    datasets_path = tmp_path / datasets_name
    datasets_path.write_text(yaml.dump(datasets_cfg))

    new_defs_path = None
    pipeline_cfg: dict = {"naming_version": 1, "datasets": datasets_name, "mount_point": None}
    if new_defs_name:
        new_defs_path = tmp_path / new_defs_name
        new_defs_path.write_text("# custom nodes\n")
        pipeline_cfg["new_definitions"] = new_defs_name

    if extra_config:
        pipeline_cfg.update(extra_config)

    pipeline_path = tmp_path / "pipeline.yml"
    pipeline_path.write_text(yaml.dump(pipeline_cfg))

    return pipeline_path, datasets_path, new_defs_path


def _load_datasets(pipeline_path: Path) -> dict:
    from neurodags.datasets import get_datasets_and_mount_point_from_pipeline_configuration

    datasets_configs, mount_point = get_datasets_and_mount_point_from_pipeline_configuration(
        pipeline_path
    )
    return datasets_configs, mount_point


# ---------------------------------------------------------------------------
# _snapshot_pipeline_config — code/ directory creation
# ---------------------------------------------------------------------------


def test_snapshot_creates_code_dir(tmp_path):
    pipeline_path, _, _ = _make_pipeline_yaml(tmp_path)
    config = yaml.safe_load(pipeline_path.read_text())
    datasets_configs, mount_point = _load_datasets(pipeline_path)

    _snapshot_pipeline_config(pipeline_path, config, datasets_configs, mount_point)

    code_dir = tmp_path / "derivatives" / "code"
    assert code_dir.is_dir()


# ---------------------------------------------------------------------------
# _snapshot_pipeline_config — pipeline YAML copied
# ---------------------------------------------------------------------------


def test_snapshot_copies_pipeline_yaml(tmp_path):
    pipeline_path, _, _ = _make_pipeline_yaml(tmp_path)
    config = yaml.safe_load(pipeline_path.read_text())
    datasets_configs, mount_point = _load_datasets(pipeline_path)

    _snapshot_pipeline_config(pipeline_path, config, datasets_configs, mount_point)

    copied = tmp_path / "derivatives" / "code" / "pipeline.yml"
    assert copied.exists()
    assert copied.read_text() == pipeline_path.read_text()


# ---------------------------------------------------------------------------
# _snapshot_pipeline_config — datasets YAML copied
# ---------------------------------------------------------------------------


def test_snapshot_copies_datasets_yaml(tmp_path):
    pipeline_path, datasets_path, _ = _make_pipeline_yaml(tmp_path)
    config = yaml.safe_load(pipeline_path.read_text())
    datasets_configs, mount_point = _load_datasets(pipeline_path)

    _snapshot_pipeline_config(pipeline_path, config, datasets_configs, mount_point)

    copied = tmp_path / "derivatives" / "code" / "datasets.yml"
    assert copied.exists()
    assert copied.read_text() == datasets_path.read_text()


# ---------------------------------------------------------------------------
# _snapshot_pipeline_config — new_definitions file copied
# ---------------------------------------------------------------------------


def test_snapshot_copies_new_definitions(tmp_path):
    pipeline_path, _, new_defs_path = _make_pipeline_yaml(tmp_path, new_defs_name="custom_nodes.py")
    config = yaml.safe_load(pipeline_path.read_text())
    datasets_configs, mount_point = _load_datasets(pipeline_path)

    _snapshot_pipeline_config(pipeline_path, config, datasets_configs, mount_point)

    copied = tmp_path / "derivatives" / "code" / "custom_nodes.py"
    assert copied.exists()
    assert copied.read_text() == new_defs_path.read_text()


def test_snapshot_new_definitions_list(tmp_path):
    nd1 = tmp_path / "nodes_a.py"
    nd1.write_text("# nodes a\n")
    nd2 = tmp_path / "nodes_b.py"
    nd2.write_text("# nodes b\n")

    datasets_cfg = {
        "ds": {
            "name": "DS",
            "source_root": str(tmp_path),
            "file_pattern": str(tmp_path / "**/*.vhdr"),
            "derivatives_path": str(tmp_path / "deriv"),
        }
    }
    (tmp_path / "deriv").mkdir()
    datasets_path = tmp_path / "datasets.yml"
    datasets_path.write_text(yaml.dump(datasets_cfg))

    pipeline_cfg = {
        "naming_version": 1,
        "datasets": "datasets.yml",
        "mount_point": None,
        "new_definitions": ["nodes_a.py", "nodes_b.py"],
    }
    pipeline_path = tmp_path / "pipeline.yml"
    pipeline_path.write_text(yaml.dump(pipeline_cfg))

    config = yaml.safe_load(pipeline_path.read_text())
    datasets_configs, mount_point = _load_datasets(pipeline_path)

    _snapshot_pipeline_config(pipeline_path, config, datasets_configs, mount_point)

    code_dir = tmp_path / "deriv" / "code"
    assert (code_dir / "nodes_a.py").exists()
    assert (code_dir / "nodes_b.py").exists()


# ---------------------------------------------------------------------------
# _snapshot_pipeline_config — datasets_configuration override is copied
# ---------------------------------------------------------------------------


def test_snapshot_copies_datasets_override(tmp_path):
    pipeline_path, _, _ = _make_pipeline_yaml(tmp_path)
    config = yaml.safe_load(pipeline_path.read_text())
    datasets_configs, mount_point = _load_datasets(pipeline_path)

    override_path = tmp_path / "override_datasets.yml"
    override_path.write_text(
        yaml.dump(
            {
                "test_ds": {
                    "name": "TestDS",
                    "source_root": str(tmp_path),
                    "file_pattern": str(tmp_path / "**/*.vhdr"),
                    "derivatives_path": str(tmp_path / "derivatives"),
                }
            }
        )
    )

    _snapshot_pipeline_config(
        pipeline_path,
        config,
        datasets_configs,
        mount_point,
        datasets_configuration=str(override_path),
    )

    copied = tmp_path / "derivatives" / "code" / "override_datasets.yml"
    assert copied.exists()


# ---------------------------------------------------------------------------
# _snapshot_pipeline_config — neurodags_env.json written with required keys
# ---------------------------------------------------------------------------


def test_snapshot_writes_env_json(tmp_path):
    pipeline_path, _, _ = _make_pipeline_yaml(tmp_path)
    config = yaml.safe_load(pipeline_path.read_text())
    datasets_configs, mount_point = _load_datasets(pipeline_path)

    _snapshot_pipeline_config(pipeline_path, config, datasets_configs, mount_point)

    env_path = tmp_path / "derivatives" / "code" / "neurodags_env.json"
    assert env_path.exists()
    data = json.loads(env_path.read_text())
    assert "snapshot_time" in data


def test_snapshot_env_json_has_version(tmp_path):
    pipeline_path, _, _ = _make_pipeline_yaml(tmp_path)
    config = yaml.safe_load(pipeline_path.read_text())
    datasets_configs, mount_point = _load_datasets(pipeline_path)

    _snapshot_pipeline_config(pipeline_path, config, datasets_configs, mount_point)

    env_path = tmp_path / "derivatives" / "code" / "neurodags_env.json"
    data = json.loads(env_path.read_text())
    # version or git commit must be present (at least one, depending on install type)
    assert "neurodags_version" in data or "neurodags_git_commit" in data


# ---------------------------------------------------------------------------
# _snapshot_pipeline_config — skips datasets without derivatives_path
# ---------------------------------------------------------------------------


def test_snapshot_skips_dataset_without_derivatives_path(tmp_path):
    pipeline_path, _, _ = _make_pipeline_yaml(tmp_path)
    config = yaml.safe_load(pipeline_path.read_text())

    no_deriv_config = DatasetConfig(
        name="NoDeriv",
        source_root=str(tmp_path),
        file_pattern=str(tmp_path / "**/*.vhdr"),
        derivatives_path=None,
    )
    datasets_configs = {"no_deriv": no_deriv_config}

    _snapshot_pipeline_config(pipeline_path, config, datasets_configs, None)

    # Nothing should have been written
    assert not (tmp_path / "code").exists()


# ---------------------------------------------------------------------------
# _snapshot_pipeline_config — deduplicates shared derivatives_path
# ---------------------------------------------------------------------------


def test_snapshot_deduplicates_shared_derivatives_path(tmp_path):
    deriv_dir = tmp_path / "shared_deriv"
    deriv_dir.mkdir()

    ds1 = DatasetConfig(
        name="DS1",
        source_root=str(tmp_path),
        file_pattern=str(tmp_path / "**/*.vhdr"),
        derivatives_path=str(deriv_dir),
    )
    ds2 = DatasetConfig(
        name="DS2",
        source_root=str(tmp_path),
        file_pattern=str(tmp_path / "**/*.vhdr"),
        derivatives_path=str(deriv_dir),
    )
    datasets_configs = {"ds1": ds1, "ds2": ds2}

    pipeline_path, _, _ = _make_pipeline_yaml(tmp_path)
    config = yaml.safe_load(pipeline_path.read_text())

    _snapshot_pipeline_config(pipeline_path, config, datasets_configs, None)

    # Only one copy of the pipeline yaml should exist
    code_dir = deriv_dir / "code"
    assert code_dir.is_dir()
    pipeline_copies = list(code_dir.glob("pipeline.yml"))
    assert len(pipeline_copies) == 1


# ---------------------------------------------------------------------------
# _snapshot_pipeline_config — multiple datasets each get their own code dir
# ---------------------------------------------------------------------------


def test_snapshot_multiple_datasets_get_own_code_dir(tmp_path):
    deriv_a = tmp_path / "deriv_a"
    deriv_a.mkdir()
    deriv_b = tmp_path / "deriv_b"
    deriv_b.mkdir()

    ds_a = DatasetConfig(
        name="DSA",
        source_root=str(tmp_path),
        file_pattern=str(tmp_path / "**/*.vhdr"),
        derivatives_path=str(deriv_a),
    )
    ds_b = DatasetConfig(
        name="DSB",
        source_root=str(tmp_path),
        file_pattern=str(tmp_path / "**/*.vhdr"),
        derivatives_path=str(deriv_b),
    )

    pipeline_path, _, _ = _make_pipeline_yaml(tmp_path)
    config = yaml.safe_load(pipeline_path.read_text())

    _snapshot_pipeline_config(pipeline_path, config, {"a": ds_a, "b": ds_b}, None)

    assert (deriv_a / "code" / "pipeline.yml").exists()
    assert (deriv_b / "code" / "pipeline.yml").exists()


# ---------------------------------------------------------------------------
# _snapshot_pipeline_config — missing new_definitions file silently skipped
# ---------------------------------------------------------------------------


def test_snapshot_missing_new_defs_file_silently_skipped(tmp_path):
    pipeline_path, _, _ = _make_pipeline_yaml(tmp_path)
    config = yaml.safe_load(pipeline_path.read_text())
    config["new_definitions"] = "does_not_exist.py"
    datasets_configs, mount_point = _load_datasets(pipeline_path)

    # Must not raise
    _snapshot_pipeline_config(pipeline_path, config, datasets_configs, mount_point)

    code_dir = tmp_path / "derivatives" / "code"
    assert not (code_dir / "does_not_exist.py").exists()


# ---------------------------------------------------------------------------
# run_pipeline integration — snapshot triggered on real run, not dry run
# ---------------------------------------------------------------------------


def test_run_pipeline_triggers_snapshot(tmp_path, dummy_pipeline):
    """run_pipeline writes code/ next to derivatives when given a YAML path."""
    from neurodags.datasets import generate_dummy_dataset

    data_dir = tmp_path / "rawdata"
    deriv_dir = tmp_path / "deriv"
    deriv_dir.mkdir()

    generate_dummy_dataset(
        data_params={
            "DATASET": "snap_test",
            "PATTERN": "sub-%subject%/sub-%subject%_task-rest",
            "NSUBS": 1,
            "NSESSIONS": 1,
            "NTASKS": 1,
            "NACQS": 1,
            "NRUNS": 1,
            "PREFIXES": {
                "subject": "S",
                "session": "SE",
                "task": "T",
                "acquisition": "A",
                "run": "R",
            },
            "ROOT": str(data_dir),
        },
        generation_args={
            "NCHANNELS": 4,
            "SFREQ": 100.0,
            "STOP": 5.0,
            "NUMEVENTS": 2,
            "random_state": 0,
        },
    )

    datasets_cfg = {
        "snap_ds": {
            "name": "SnapDS",
            "source_root": str(data_dir),
            "file_pattern": str(data_dir / "**/*.vhdr"),
            "derivatives_path": str(deriv_dir),
        }
    }
    pipeline_cfg = {
        "naming_version": 1,
        "datasets": "snap_datasets.yml",
        "mount_point": None,
        "DerivativeDefinitions": {},
        "DerivativeList": [],
    }

    datasets_path = tmp_path / "snap_datasets.yml"
    datasets_path.write_text(yaml.dump(datasets_cfg))
    pipeline_path = tmp_path / "snap_pipeline.yml"
    pipeline_path.write_text(yaml.dump(pipeline_cfg))

    run_pipeline(str(pipeline_path))

    assert (deriv_dir / "code" / "snap_pipeline.yml").exists()
    assert (deriv_dir / "code" / "snap_datasets.yml").exists()
    assert (deriv_dir / "code" / "neurodags_env.json").exists()


def test_run_pipeline_dry_run_does_not_snapshot(tmp_path):
    """dry_run=True must not write code/ snapshot."""
    from neurodags.datasets import generate_dummy_dataset

    data_dir = tmp_path / "rawdata"
    deriv_dir = tmp_path / "deriv"
    deriv_dir.mkdir()

    generate_dummy_dataset(
        data_params={
            "DATASET": "dry_test",
            "PATTERN": "sub-%subject%/sub-%subject%_task-rest",
            "NSUBS": 1,
            "NSESSIONS": 1,
            "NTASKS": 1,
            "NACQS": 1,
            "NRUNS": 1,
            "PREFIXES": {
                "subject": "S",
                "session": "SE",
                "task": "T",
                "acquisition": "A",
                "run": "R",
            },
            "ROOT": str(data_dir),
        },
        generation_args={
            "NCHANNELS": 4,
            "SFREQ": 100.0,
            "STOP": 5.0,
            "NUMEVENTS": 2,
            "random_state": 0,
        },
    )

    datasets_cfg = {
        "dry_ds": {
            "name": "DryDS",
            "source_root": str(data_dir),
            "file_pattern": str(data_dir / "**/*.vhdr"),
            "derivatives_path": str(deriv_dir),
        }
    }
    pipeline_cfg = {
        "naming_version": 1,
        "datasets": "dry_datasets.yml",
        "mount_point": None,
        "DerivativeDefinitions": {},
        "DerivativeList": [],
    }

    datasets_path = tmp_path / "dry_datasets.yml"
    datasets_path.write_text(yaml.dump(datasets_cfg))
    pipeline_path = tmp_path / "dry_pipeline.yml"
    pipeline_path.write_text(yaml.dump(pipeline_cfg))

    run_pipeline(str(pipeline_path), dry_run=True)

    assert not (deriv_dir / "code").exists()
