"""Regressions for runtime integrity bugs found in the 0.4.0 review (114605e, R1-R4).

- R1: two artifacts of one unit sharing a file (``.EO.fif``'s MNE split part
  ``.EO-1.fif`` and an artifact ``.EO-1.fif``) silently corrupted a recording certified done.
- R2: two processes on the same unit could leave a ``.done`` certifying the other's
  partial write; units are now claimed with ``<prefix>.lock``.
- R3: a ``save: false`` derivative reached as a dependency was read from an old manifest.
- R4: a failed dependency (or argument preparation) left an overwritten parent's old
  ``.done`` and no parent ``.error``.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import mne
import numpy as np
import pytest

from neurodags import dag
from neurodags.dag import run_derivative
from neurodags.definitions import Artifact, NodeResult
from neurodags.derivatives.pipeline import register_derivatives_from_dict
from neurodags.naming import NamingError, validate_artifact_suffixes
from neurodags.nodes import register_node
from neurodags.outputs import (
    UnitLockedError,
    acquire_unit_lock,
    marker_path,
    read_manifest,
    unit_state,
)
from neurodags.writers import save_mne

mne.set_log_level("ERROR")


def _source(tmp_path: Path) -> Path:
    base = tmp_path / "source"
    base.write_text("source")
    return base


def _single(node: str, **args) -> dict:
    return {"nodes": [{"id": 0, "node": node, "args": args}]}


# ---------------------------------------------------------------------------
# R1: artifact suffixes whose files overlap
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "suffixes",
    [
        [".EO.fif", ".EO-1.fif"],
        [".EO-12.fif", ".EO.fif"],
        [".eo.FIF", ".EO-1.fif"],
        [".a.b.fif.gz", ".a.b.fif-3.gz"],
    ],
)
def test_split_part_suffix_pair_is_rejected(suffixes):
    with pytest.raises(NamingError, match="split parts"):
        validate_artifact_suffixes(suffixes)


@pytest.mark.parametrize(
    "suffixes",
    [
        [".EO.fif", ".EO-1.nc"],
        [".EO-1.fif"],
        [".EO.fif", ".EC-1.fif"],
        [".fif", ".x-1.fif"],
        [".EO.fif", ".EO-x.fif"],
    ],
)
def test_non_overlapping_suffixes_are_accepted(suffixes):
    validate_artifact_suffixes(suffixes)


def _raws():
    big = mne.io.RawArray(np.ones((4, 200_000)) * 1e-5, mne.create_info(4, 1000, "eeg"))
    short = mne.io.RawArray(np.ones((4, 1000)) * 9e-5, mne.create_info(4, 1000, "eeg"))
    return big, short


def _register_split_node(order):
    big, short = _raws()
    artifacts = {
        ".EO.fif": Artifact(big, lambda p: save_mne(big, p, split_size="2MB", verbose="error")),
        ".EO-1.fif": Artifact(short, lambda p: save_mne(short, p, verbose="error")),
    }

    @register_node(name="rt_split_overlap", override=True)
    def node():
        return NodeResult({k: artifacts[k] for k in order})

    return big


def test_split_suffix_pair_fails_unit_before_writing(tmp_path):
    """Astra's repro: the pair is refused before any writer runs; no .done."""
    _register_split_node([".EO.fif", ".EO-1.fif"])
    base = _source(tmp_path)
    with pytest.raises(NamingError):
        run_derivative(_single("rt_split_overlap"), "Split", str(base), base)
    prefix = str(base) + "@Split"
    assert unit_state(prefix, "Split").state == "error"
    assert read_manifest(prefix) is None
    assert not os.path.exists(prefix + ".EO.fif")
    assert not os.path.exists(prefix + ".lock")


@pytest.mark.parametrize(
    "order", [[".EO.fif", ".EO-1.fif"], [".EO-1.fif", ".EO.fif"]], ids=["big-first", "short-first"]
)
def test_overlapping_writer_paths_fail_unit(tmp_path, monkeypatch, order):
    """Publish-time guard (validation bypassed): a file claimed twice is never certified."""
    monkeypatch.setattr(dag, "validate_artifact_suffixes", lambda suffixes: None)
    big = _register_split_node(order)
    base = _source(tmp_path)
    with pytest.raises(RuntimeError, match="write the same file"):
        run_derivative(_single("rt_split_overlap"), "Split", str(base), base)
    prefix = str(base) + "@Split"
    assert read_manifest(prefix) is None
    assert unit_state(prefix, "Split").state == "error"
    if order[0] == ".EO.fif":
        # The second writer was refused before it ran: the split recording is intact.
        loaded = mne.io.read_raw_fif(prefix + ".EO.fif", preload=True, verbose="error")
        assert loaded.n_times == big.n_times


def test_writer_reporting_another_artifacts_file_fails_unit(tmp_path):
    @register_node(name="rt_claims_other", override=True)
    def node():
        def first(p):
            other = p[: -len(".a.txt")] + ".b.txt"
            Path(p).write_text("a")
            Path(other).write_text("a's extra file")
            return [p, other]

        return NodeResult(
            {
                ".a.txt": Artifact("a", first),
                ".b.txt": Artifact("b", lambda p: Path(p).write_text("b")),
            }
        )

    base = _source(tmp_path)
    with pytest.raises(RuntimeError, match="write the same file"):
        run_derivative(_single("rt_claims_other"), "Claims", str(base), base)
    prefix = str(base) + "@Claims"
    assert unit_state(prefix, "Claims").state == "error"
    assert Path(prefix + ".b.txt").read_text() == "a's extra file"  # not overwritten


# ---------------------------------------------------------------------------
# R2: per-unit lock
# ---------------------------------------------------------------------------

_FORK = "fork" in mp.get_all_start_methods()


@pytest.mark.skipif(not _FORK, reason="needs the fork start method")
def test_concurrent_run_of_same_unit_cannot_certify_partial_write(tmp_path):
    """Astra's race, forced with events and real processes (no mocks).

    A is inside its first writer when B starts on the same unit. Before the fix B deleted
    the (absent) .done, entered its writer, and after A published, overwrote A's file and
    died: A's .done certified B's partial file. Now B finds the lock and does not touch
    the unit.
    """
    ctx = mp.get_context("fork")
    a_ready, b_entered, release_a, allow_b = (ctx.Event() for _ in range(4))

    @register_node(name="rt_race", override=True)
    def race(who):
        def first(p):
            if who == "A":
                Path(p).write_text("A complete")
                a_ready.set()
                assert release_a.wait(30)
            else:
                b_entered.set()
                assert allow_b.wait(30)
                Path(p).write_text("B PARTIAL")
                os._exit(23)

        def second(p):
            Path(p).write_text(who + " complete")

        return NodeResult({".a.txt": Artifact(who, first), ".b.txt": Artifact(who, second)})

    base = _source(tmp_path)
    prefix = str(base) + "@Race"

    def definition(who):
        return _single("rt_race", who=who)

    def worker(who):
        try:
            run_derivative(definition(who), "Race", str(base), base)
        except UnitLockedError:
            os._exit(42)
        except BaseException:
            os._exit(1)
        os._exit(0)

    a = ctx.Process(target=worker, args=("A",))
    a.start()
    assert a_ready.wait(30)
    b = ctx.Process(target=worker, args=("B",))
    b.start()
    deadline = time.monotonic() + 30
    while b.is_alive() and not b_entered.is_set() and time.monotonic() < deadline:
        time.sleep(0.01)
    release_a.set()
    a.join(30)
    allow_b.set()
    b.join(30)

    assert a.exitcode == 0
    assert not b_entered.is_set(), "B entered the writer of a unit A was writing"
    assert b.exitcode == 42
    state = unit_state(prefix, "Race")
    assert state.state == "done"
    assert Path(prefix + ".a.txt").read_text() == "A complete"
    assert Path(prefix + ".b.txt").read_text() == "A complete"
    assert not os.path.exists(prefix + ".lock")
    assert "cached" in run_derivative(definition("A"), "Race", str(base), base)


@register_node(name="rt_value", override=True)
def _rt_value(value):
    if value == "FAIL":
        raise RuntimeError("rt_value failed")
    return NodeResult({".txt": Artifact(value, lambda p: Path(p).write_text(value))})


def test_held_lock_refuses_unit_and_status_reports_locked(tmp_path):
    base = _source(tmp_path)
    prefix = str(base) + "@Held"
    stale = acquire_unit_lock(prefix, "Held")  # e.g. left by a killed job
    info = json.loads(Path(prefix + ".lock").read_text())
    assert info["pid"] == os.getpid()
    assert info["host"]
    assert info["started"]

    with pytest.raises(UnitLockedError, match="delete the lock file") as excinfo:
        run_derivative(_single("rt_value", value="v"), "Held", str(base), base)
    assert excinfo.value.lock_path == prefix + ".lock"
    assert not os.path.exists(prefix + ".txt")
    assert not os.path.exists(prefix + ".error")  # the unit belongs to the holder
    state = unit_state(prefix, "Held")
    assert state.state == "locked"
    assert state.lock_path == prefix + ".lock"
    assert state.unlisted == ()

    plan = run_derivative(_single("rt_value", value="v"), "Held", str(base), base, dry_run=True)
    final = next(s for s in plan["plan"] if s["kind"] == "derivative_output")
    assert final["state"] == "locked"
    assert final["lock_path"] == prefix + ".lock"

    stale.release()
    run_derivative(_single("rt_value", value="v"), "Held", str(base), base)
    state = unit_state(prefix, "Held")
    assert state.state == "done"
    assert state.unlisted == ()
    assert not os.path.exists(prefix + ".lock")


def test_lock_on_done_unit_does_not_block_cache_hit(tmp_path):
    base = _source(tmp_path)
    prefix = str(base) + "@Done"
    run_derivative(_single("rt_value", value="v"), "Done", str(base), base)
    lock = acquire_unit_lock(prefix, "Done")
    try:
        assert unit_state(prefix, "Done").state == "done"
        assert "cached" in run_derivative(_single("rt_value", value="v"), "Done", str(base), base)
        with pytest.raises(UnitLockedError):
            run_derivative(
                {**_single("rt_value", value="w"), "overwrite": True}, "Done", str(base), base
            )
        assert Path(prefix + ".txt").read_text() == "v"
        assert unit_state(prefix, "Done").state == "done"
    finally:
        lock.release()


def test_lock_released_after_error_and_skip(tmp_path):
    from neurodags.definitions import SkipDerivative

    @register_node(name="rt_skip", override=True)
    def skip():
        raise SkipDerivative("not applicable")

    base = _source(tmp_path)
    with pytest.raises(RuntimeError):
        run_derivative(_single("rt_value", value="FAIL"), "Err", str(base), base)
    assert unit_state(str(base) + "@Err", "Err").state == "error"
    assert not os.path.exists(str(base) + "@Err.lock")

    assert "skipped" in run_derivative(_single("rt_skip"), "Skp", str(base), base)
    assert unit_state(str(base) + "@Skp", "Skp").state == "skipped"
    assert not os.path.exists(str(base) + "@Skp.lock")


def test_lock_suffix_is_reserved():
    with pytest.raises(NamingError, match="reserved"):
        validate_artifact_suffixes([".lock"])
    assert marker_path("x@A", "lock") == "x@A.lock"


def test_locked_child_does_not_mark_parent(tmp_path):
    register_derivatives_from_dict(
        {
            "DerivativeDefinitions": {
                "RtLockChild": _single("rt_value", value="c"),
                "RtLockParent": {
                    "nodes": [
                        {"id": 0, "derivative": "RtLockChild.txt"},
                        {"id": 1, "node": "dummy", "args": {"param1": "id.0"}},
                    ]
                },
            }
        }
    )
    base = _source(tmp_path)
    child_lock = acquire_unit_lock(str(base) + "@RtLockChild", "RtLockChild")
    try:
        parent_def = dag.get_derivative_definition("RtLockParent").definition
        with pytest.raises(UnitLockedError):
            run_derivative(parent_def, "RtLockParent", str(base), base)
        parent = str(base) + "@RtLockParent"
        assert unit_state(parent, "RtLockParent").state == "missing"
        assert not os.path.exists(parent + ".lock")
    finally:
        child_lock.release()


def test_status_cli_reports_locked_units(tmp_path, capsys, monkeypatch):
    import yaml

    from neurodags.cli import main

    monkeypatch.chdir(tmp_path)
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw" / "a.txt").write_text("a")
    ds = {"name": "d", "source_root": "raw", "file_pattern": "raw/**/*.txt"}
    ds["derivatives_path"] = "out"
    (tmp_path / "ds.yml").write_text(yaml.safe_dump({"d": ds}))
    pipeline = tmp_path / "pl.yml"
    pipeline.write_text(
        yaml.safe_dump(
            {
                "naming_version": 1,
                "datasets": "ds.yml",
                "DerivativeDefinitions": {"Val": _single("rt_value", value="v")},
                "DerivativeList": ["Val"],
            }
        )
    )
    out = tmp_path / "out"
    out.mkdir()
    lock = acquire_unit_lock(str(out / "a.txt") + "@Val", "Val")
    try:
        rc = main(["status", str(pipeline), "--format", "json", "--include-files"])
        data = json.loads(capsys.readouterr().out)
        assert rc == 1
        assert data["complete"] is False
        assert data["grand_total"]["locked"] == 1
        assert data["files"][0]["status"] == "locked"
        assert data["files"][0]["lock_path"] == str(out / "a.txt") + "@Val.lock"
        assert data["locked_indices"] == [0]

        rc = main(["status", str(pipeline)])
        text = capsys.readouterr().out
        assert "1 unit(s) locked" in text
        assert str(out / "a.txt") + "@Val.lock" in text
    finally:
        lock.release()


# ---------------------------------------------------------------------------
# R3: save: false dependency never read from cache
# ---------------------------------------------------------------------------


@register_node(name="rt_read", override=True)
def _rt_read(data):
    value = Path(data).read_text() if isinstance(data, str) and Path(data).is_file() else data
    if isinstance(value, NodeResult):
        value = next(iter(value.artifacts.values())).item
    return NodeResult({".txt": Artifact(value, lambda p: Path(p).write_text(value))})


def test_save_false_dependency_ignores_old_manifest(tmp_path):
    """Astra's repro: a formerly saved child turned save: false is recomputed via a parent."""
    base = _source(tmp_path)
    run_derivative(_single("rt_value", value="OLD"), "RtSfChild", str(base), base)
    assert unit_state(str(base) + "@RtSfChild", "RtSfChild").state == "done"

    child = {"save": False, **_single("rt_value", value="NEW")}
    parent = {
        "nodes": [
            {"id": 0, "derivative": "RtSfChild.txt"},
            {"id": 1, "node": "rt_read", "args": {"data": "id.0"}},
        ]
    }
    register_derivatives_from_dict(
        {"DerivativeDefinitions": {"RtSfChild": child, "RtSfParent": parent}}
    )
    assert run_derivative(child, "RtSfChild", str(base), base).artifacts[".txt"].item == "NEW"

    plan = run_derivative(parent, "RtSfParent", str(base), base, dry_run=True)
    dep = next(s for s in plan["plan"] if s["kind"] == "derivative")
    assert dep["cached"] is False

    result = run_derivative(parent, "RtSfParent", str(base), base)
    assert result.artifacts[".txt"].item == "NEW"
    assert Path(str(base) + "@RtSfParent.txt").read_text() == "NEW"


# ---------------------------------------------------------------------------
# R4: failures inside a unit fail the unit
# ---------------------------------------------------------------------------


def _register_parent_child(child: dict, prefix: str = "RtF") -> dict:
    parent = {
        "nodes": [
            {"id": 0, "derivative": f"{prefix}Child.message.txt"},
            {"id": 1, "node": "dummy", "args": {"param1": "id.0"}},
        ]
    }
    register_derivatives_from_dict(
        {"DerivativeDefinitions": {f"{prefix}Child": child, f"{prefix}Parent": parent}}
    )
    return parent


@register_node(name="rt_fail", override=True)
def _rt_fail():
    raise RuntimeError("child failed")


def test_failed_dependency_of_overwritten_parent_marks_parent_error(tmp_path):
    """Astra's repro: the parent's old .done no longer certifies a failed recompute."""
    base = _source(tmp_path)
    parent = _register_parent_child(
        {"nodes": [{"id": 0, "node": "dummy", "args": {"param1": "old"}}]}
    )
    run_derivative(parent, "RtFParent", str(base), base)
    parent_prefix = str(base) + "@RtFParent"
    assert unit_state(parent_prefix, "RtFParent").state == "done"

    _register_parent_child({"overwrite": True, "nodes": [{"id": 0, "node": "rt_fail"}]})
    with pytest.raises(RuntimeError, match="child failed"):
        run_derivative({**parent, "overwrite": True}, "RtFParent", str(base), base)

    assert unit_state(str(base) + "@RtFChild", "RtFChild").state == "error"
    assert unit_state(parent_prefix, "RtFParent").state == "error"
    text = Path(parent_prefix + ".error").read_text()
    assert "derivative='RtFChild.message.txt'" in text
    assert "child failed" in text
    with pytest.raises(RuntimeError, match="child failed"):
        run_derivative(parent, "RtFParent", str(base), base)
    assert not os.path.exists(parent_prefix + ".lock")


def test_failed_dependency_of_fresh_parent_marks_parent_error(tmp_path):
    base = _source(tmp_path)
    parent = _register_parent_child({"nodes": [{"id": 0, "node": "rt_fail"}]}, prefix="RtG")
    with pytest.raises(RuntimeError, match="child failed"):
        run_derivative(parent, "RtGParent", str(base), base)
    assert unit_state(str(base) + "@RtGParent", "RtGParent").state == "error"


def test_save_false_child_failure_marks_parent_error(tmp_path):
    base = _source(tmp_path)
    parent = _register_parent_child(
        {"save": False, "nodes": [{"id": 0, "node": "rt_fail"}]}, prefix="RtH"
    )
    with pytest.raises(RuntimeError, match="child failed"):
        run_derivative(parent, "RtHParent", str(base), base)
    assert unit_state(str(base) + "@RtHParent", "RtHParent").state == "error"
    assert unit_state(str(base) + "@RtHChild", "RtHChild").state == "missing"


def test_argument_preparation_failure_marks_overwritten_parent_error(tmp_path):
    base = _source(tmp_path)
    good = _single("rt_value", value="v")
    run_derivative(good, "RtArgs", str(base), base)
    prefix = str(base) + "@RtArgs"
    assert unit_state(prefix, "RtArgs").state == "done"

    bad = {
        "overwrite": True,
        "nodes": [
            {"id": 0, "derivative": "SourceFile"},
            {"id": 1, "node": "rt_value", "args": {"value": "id.0.missing_field"}},
        ],
    }
    with pytest.raises(ValueError, match="Field access"):
        run_derivative(bad, "RtArgs", str(base), base)
    assert read_manifest(prefix) is None
    state = unit_state(prefix, "RtArgs")
    assert state.state == "error"
    assert "step id=1 node='rt_value'" in Path(prefix + ".error").read_text()
