"""What the parent sends to (reused) loky workers (#35, #36).

loky reuses worker processes across ``Parallel`` calls. A reused worker keeps the
working directory it was started with, so anything cwd-relative in a job payload is
resolved against a stale directory once the parent has ``os.chdir``-ed (notebooks,
tests, scripts calling ``run_pipeline`` more than once). Fresh workers, on the other
hand, start with empty registries, so everything they need (custom nodes from
``new_definitions``) must travel with the job.
"""

from __future__ import annotations

import os
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed

from neurodags.orchestrators import build_derivative_dataframe, run_pipeline

CUSTOM_NODES = """
from neurodags.nodes import get_node, register_node


@register_node(name="{name}", override=True)
def _custom_meta(mne_object):
    return get_node("extract_meeg_metadata")(mne_object)
"""


def _write_project(root: Path, *, node_name: str, save: bool = True) -> Path:
    """A project whose dataset paths and ``new_definitions`` are all relative."""
    (root / "in" / "sub-01").mkdir(parents=True)
    (root / "in" / "sub-02").mkdir(parents=True)
    for i, sub in enumerate(("sub-01", "sub-02")):
        info = mne.create_info(["a", "b"], 100.0, "eeg")
        raw = mne.io.RawArray(np.random.default_rng(i).standard_normal((2, 1000)) * 1e-5, info)
        raw.save(root / "in" / sub / "rec_raw.fif", verbose="error")
    (root / "custom_nodes.py").write_text(CUSTOM_NODES.format(name=node_name))
    ds = {
        "d": {
            "name": "d",
            # Relative to the process cwd, which is what the issue is about.
            "file_pattern": "in/**/*.fif",
            "derivatives_path": "out",
        }
    }
    (root / "ds.yml").write_text(yaml.safe_dump(ds))
    pipeline = {
        "datasets": "ds.yml",
        "new_definitions": "custom_nodes.py",
        "DerivativeDefinitions": {
            "Meta": {
                "save": save,
                "for_dataframe": True,
                "nodes": [
                    {"id": 0, "derivative": "SourceFile"},
                    {"id": 1, "node": node_name, "args": {"mne_object": "id.0"}},
                ],
            }
        },
        "DerivativeList": ["Meta"],
    }
    pipe = root / "pl.yml"
    pipe.write_text(yaml.safe_dump(pipeline))
    return pipe


def _worker_cwds(n_jobs: int = 2) -> set[str]:
    return set(Parallel(n_jobs=n_jobs)(delayed(os.getcwd)() for _ in range(8)))


def _outputs(root: Path) -> list[str]:
    return sorted(p.relative_to(root).as_posix() for p in (root / "out").rglob("*@Meta*"))


EXPECTED_OUTPUTS = ["out/sub-01/rec_raw.fif@Meta.json", "out/sub-02/rec_raw.fif@Meta.json"]


def test_reused_workers_resolve_relative_paths_after_chdir(monkeypatch, tmp_path):
    """#35: relative file_pattern/derivatives_path survive a chdir with reused workers."""
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()

    # Start (or reuse) the loky workers from `first`.
    monkeypatch.chdir(first)
    pipe1 = _write_project(first, node_name="custom_meta_35")
    run_pipeline(str(pipe1), raise_on_error=True, n_jobs=2)
    assert _outputs(first) == EXPECTED_OUTPUTS
    assert str(second) not in _worker_cwds()  # the workers are stale for `second`

    monkeypatch.chdir(second)
    pipe2 = _write_project(second, node_name="custom_meta_35")
    run_pipeline(str(pipe2), raise_on_error=True, n_jobs=2)

    # Same relative layout as a serial run, under `second`, and nothing leaked
    # into `first` (the workers' stale cwd).
    assert _outputs(second) == EXPECTED_OUTPUTS
    assert _outputs(first) == EXPECTED_OUTPUTS

    df = build_derivative_dataframe(str(pipe2), raise_on_error=True, n_jobs=2)
    assert len(df) == 2
    assert not any(c.endswith("__error") for c in df.columns)
    # The reported file_path keeps the discovered (relative) spelling.
    assert sorted(df["file_path"]) == ["in/sub-01/rec_raw.fif", "in/sub-02/rec_raw.fif"]


def test_parallel_layout_matches_serial(monkeypatch, tmp_path):
    """#35: absolutising payload paths must not change the output layout."""
    serial, parallel = tmp_path / "serial", tmp_path / "parallel"
    serial.mkdir()
    parallel.mkdir()

    monkeypatch.chdir(serial)
    run_pipeline(str(_write_project(serial, node_name="custom_meta_35")), n_jobs=1)
    monkeypatch.chdir(parallel)
    run_pipeline(str(_write_project(parallel, node_name="custom_meta_35")), n_jobs=2)

    assert _outputs(serial) == _outputs(parallel) == EXPECTED_OUTPUTS


def test_parallel_dataframe_loads_custom_nodes_for_unsaved_derivative(monkeypatch, tmp_path):
    """#36: workers computing a save=False derivative need the custom nodes."""
    monkeypatch.chdir(tmp_path)
    # A node name no worker of this session has ever seen.
    pipe = _write_project(tmp_path, node_name="custom_meta_36_unsaved", save=False)

    df = build_derivative_dataframe(str(pipe), raise_on_error=True, n_jobs=2)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert not any(c.endswith("__error") for c in df.columns)
    assert not (tmp_path / "out").exists() or _outputs(tmp_path) == []
