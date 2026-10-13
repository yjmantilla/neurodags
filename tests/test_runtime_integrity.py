"""Regressions for runtime integrity bugs found in the 0.4.0 review (114605e, R1-R4).

- R1: two artifacts of one unit sharing a file (``.EO.fif``'s MNE split part
  ``.EO-1.fif`` and an artifact ``.EO-1.fif``) silently corrupted a recording certified done.
- R2: two processes on the same unit could leave a ``.done`` certifying the other's
  partial write; artifacts are now written in a staging folder and moved into place with
  ``os.replace``, so readers and ``.done`` only ever see complete files.
- R3: a ``save: false`` derivative reached as a dependency was read from an old manifest.
- R4: a failed dependency (or argument preparation) left an overwritten parent's old
  ``.done`` and no parent ``.error``.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import shutil
import signal
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
from neurodags.outputs import read_manifest, unit_state
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
    assert unit_state(prefix, "Split").temps == ()


@pytest.mark.parametrize(
    "order", [[".EO.fif", ".EO-1.fif"], [".EO-1.fif", ".EO.fif"]], ids=["big-first", "short-first"]
)
def test_overlapping_writer_paths_fail_unit(tmp_path, monkeypatch, order):
    """Publish-time guard (validation bypassed): a file claimed twice is never certified."""
    monkeypatch.setattr(dag, "validate_artifact_suffixes", lambda suffixes: None)
    _register_split_node(order)
    base = _source(tmp_path)
    with pytest.raises(RuntimeError, match="write the same file"):
        run_derivative(_single("rt_split_overlap"), "Split", str(base), base)
    prefix = str(base) + "@Split"
    assert read_manifest(prefix) is None
    assert unit_state(prefix, "Split").state == "error"
    # The overlap is found in the staging folder: nothing reached the final names.
    assert not os.path.exists(prefix + ".EO.fif")
    assert not os.path.exists(prefix + ".EO-1.fif")
    assert unit_state(prefix, "Split").temps == ()


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
    assert not os.path.exists(prefix + ".a.txt")  # nothing published
    assert not os.path.exists(prefix + ".b.txt")


# ---------------------------------------------------------------------------
# R2: atomic publication (staging folder + os.replace)
# ---------------------------------------------------------------------------

_FORK = "fork" in mp.get_all_start_methods()
needs_fork = pytest.mark.skipif(not _FORK, reason="needs the fork start method")


@pytest.fixture
def procs():
    """Forked processes of a test; always reaped, also when the test fails."""
    started: list = []
    yield started
    for proc in started:
        if proc.is_alive():
            proc.kill()
        proc.join(10)


def _start(procs, target, *args):
    proc = mp.get_context("fork").Process(target=target, args=args)
    proc.start()
    procs.append(proc)
    return proc


def _exit_after(fn):
    """Process body: run *fn*, exit 0 on success and 1 on an exception (no pytest unwinding)."""
    try:
        fn()
    except BaseException:
        os._exit(1)
    os._exit(0)


@register_node(name="rt_value", override=True)
def _rt_value(value):
    if value == "FAIL":
        raise RuntimeError("rt_value failed")
    return NodeResult({".txt": Artifact(value, lambda p: Path(p).write_text(value))})


def test_split_recording_is_published_atomically_and_reloads(tmp_path):
    """A real MNE split recording is written in staging and moved into place, still linked."""
    big = mne.io.RawArray(
        np.arange(4 * 200_000, dtype=float).reshape(4, 200_000) * 1e-9,
        mne.create_info(4, 1000, "eeg"),
    )

    @register_node(name="rt_split_one", override=True)
    def node():
        return NodeResult(
            {".fif": Artifact(big, lambda p: save_mne(big, p, split_size="2MB", verbose="error"))}
        )

    base = _source(tmp_path)
    run_derivative(_single("rt_split_one"), "Rec", str(base), base)
    prefix = str(base) + "@Rec"
    state = unit_state(prefix, "Rec")
    assert state.state == "done"
    parts = state.artifacts[".fif"]
    assert len(parts) > 1
    assert parts[0] == prefix + ".fif"
    assert parts[1] == prefix + "-1.fif"
    assert state.temps == ()
    assert [p.name for p in tmp_path.iterdir() if p.name.startswith(".")] == []
    loaded = mne.io.read_raw_fif(prefix + ".fif", preload=True, verbose="error")
    assert loaded.n_times == big.n_times
    np.testing.assert_allclose(loaded.get_data(), big.get_data())


@needs_fork
def test_original_race_schedule_certifies_only_complete_files(tmp_path, procs):
    """Astra's R2 schedule: B enters its writer while A is in its own, A publishes, then B
    writes a partial file and dies. B wrote only in its own staging folder, so A's .done
    certifies A's complete files."""
    ctx = mp.get_context("fork")
    a_ready, b_entered, allow_b = (ctx.Event() for _ in range(3))

    @register_node(name="rt_race", override=True)
    def race(who):
        def first(p):
            if who == "A":
                Path(p).write_text("A complete")
                a_ready.set()
                assert b_entered.wait(30)
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

    def worker(who):
        _exit_after(lambda: run_derivative(_single("rt_race", who=who), "Race", str(base), base))

    a = _start(procs, worker, "A")
    assert a_ready.wait(30)
    b = _start(procs, worker, "B")
    a.join(30)
    allow_b.set()
    b.join(30)

    assert a.exitcode == 0
    assert b.exitcode == 23
    state = unit_state(prefix, "Race")
    assert state.state == "done"
    assert Path(prefix + ".a.txt").read_text() == "A complete"
    assert Path(prefix + ".b.txt").read_text() == "A complete"
    assert len(state.temps) == 1  # B's staging folder, ignored
    assert "cached" in run_derivative(_single("rt_race", who="A"), "Race", str(base), base)


@needs_fork
def test_writer_killed_mid_write_leaves_no_partial_file_and_no_done(tmp_path, procs):
    """Astra (a): a writer dying mid-write never leaves a partial file at the final name."""
    base = _source(tmp_path)
    prefix = str(base) + "@Killed"
    run_derivative(_single("rt_value", value="OLD"), "Killed", str(base), base)

    @register_node(name="rt_dies", override=True)
    def dies():
        def writer(p):
            Path(p).write_text("PARTIAL")
            os._exit(23)

        return NodeResult({".txt": Artifact("x", writer)})

    proc = _start(
        procs,
        lambda: _exit_after(
            lambda: run_derivative(
                {**_single("rt_dies"), "overwrite": True}, "Killed", str(base), base
            )
        ),
    )
    proc.join(30)
    assert proc.exitcode == 23
    assert Path(prefix + ".txt").read_text() == "OLD"  # never partial
    assert read_manifest(prefix) is None
    state = unit_state(prefix, "Killed")
    assert state.state == "incomplete"
    assert len(state.temps) == 1
    assert Path(state.temps[0], "source@Killed.txt").read_text() == "PARTIAL"

    run_derivative(_single("rt_value", value="NEW"), "Killed", str(base), base)
    assert unit_state(prefix, "Killed").state == "done"
    assert Path(prefix + ".txt").read_text() == "NEW"


@needs_fork
@pytest.mark.parametrize("writer_outcome", ["crashes", "publishes"])
def test_cached_consumer_never_reads_partial_overwrite(tmp_path, procs, writer_outcome):
    """Astra's finding 1: a consumer holding a cached dependency path while another process
    overwrites that dependency reads the complete old or the complete new file."""
    ctx = mp.get_context("fork")
    consumer_ready, go_read, partial_written = (ctx.Event() for _ in range(3))
    base = _source(tmp_path)
    child_prefix = str(base) + "@RtCacheChild"

    @register_node(name="rt_cache_writer", override=True)
    def cache_writer(value):
        def writer(p):
            Path(p).write_text("PARTIAL")
            partial_written.set()
            if value == "crashes":
                os._exit(23)
            Path(p).write_text("NEW complete")

        return NodeResult({".txt": Artifact(value, writer)})

    @register_node(name="rt_cache_reader", override=True)
    def cache_reader(data):
        consumer_ready.set()
        assert go_read.wait(30)
        text = Path(data).read_text()
        return NodeResult({".txt": Artifact(text, lambda p: Path(p).write_text(text))})

    child = _single("rt_value", value="COMPLETE")
    parent = {
        "nodes": [
            {"id": 0, "derivative": "RtCacheChild.txt"},
            {"id": 1, "node": "rt_cache_reader", "args": {"data": "id.0"}},
        ]
    }
    register_derivatives_from_dict(
        {"DerivativeDefinitions": {"RtCacheChild": child, "RtCacheParent": parent}}
    )
    run_derivative(child, "RtCacheChild", str(base), base)
    assert unit_state(child_prefix, "RtCacheChild").state == "done"

    consumer = _start(
        procs,
        lambda: _exit_after(lambda: run_derivative(parent, "RtCacheParent", str(base), base)),
    )
    assert consumer_ready.wait(30)  # the consumer holds the cached path
    overwrite = {"overwrite": True, **_single("rt_cache_writer", value=writer_outcome)}
    writer = _start(
        procs,
        lambda: _exit_after(lambda: run_derivative(overwrite, "RtCacheChild", str(base), base)),
    )
    assert partial_written.wait(30)
    writer.join(30)
    go_read.set()
    consumer.join(30)

    assert consumer.exitcode == 0
    parent_text = Path(str(base) + "@RtCacheParent.txt").read_text()
    assert unit_state(str(base) + "@RtCacheParent", "RtCacheParent").state == "done"
    if writer_outcome == "crashes":
        assert writer.exitcode == 23
        assert parent_text == "COMPLETE"
        assert Path(child_prefix + ".txt").read_text() == "COMPLETE"
    else:
        assert writer.exitcode == 0
        assert parent_text == "NEW complete"
        assert unit_state(child_prefix, "RtCacheChild").state == "done"


@needs_fork
@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGKILL], ids=["SIGTERM", "SIGKILL"])
def test_killed_job_does_not_block_next_run(tmp_path, procs, signum):
    """Astra (c): a job killed by the scheduler mid-write leaves nothing that blocks a rerun."""
    ctx = mp.get_context("fork")
    writing, never = ctx.Event(), ctx.Event()

    @register_node(name="rt_hangs", override=True)
    def hangs():
        def writer(p):
            Path(p).write_text("PARTIAL")
            writing.set()
            never.wait(60)

        return NodeResult({".txt": Artifact("x", writer)})

    base = _source(tmp_path)
    prefix = str(base) + "@Hung"
    proc = _start(
        procs,
        lambda: _exit_after(lambda: run_derivative(_single("rt_hangs"), "Hung", str(base), base)),
    )
    assert writing.wait(30)
    os.kill(proc.pid, signum)
    proc.join(30)
    assert proc.exitcode == -signum
    assert not os.path.exists(prefix + ".txt")
    assert unit_state(prefix, "Hung").state == "missing"

    run_derivative(_single("rt_value", value="v"), "Hung", str(base), base)
    state = unit_state(prefix, "Hung")
    assert state.state == "done"
    assert Path(prefix + ".txt").read_text() == "v"
    assert len(state.temps) == 1  # the killed job's staging folder, reported, never read


def test_staging_folder_names_never_parse_as_outputs(tmp_path):
    from neurodags.migrate import _scan_tree
    from neurodags.naming import parse_output_basename
    from neurodags.outputs import is_staging_name, make_staging_dir

    prefix = str(tmp_path / "x&A.fif@B")
    staging = make_staging_dir(prefix)
    name = os.path.basename(staging)
    assert is_staging_name(name)
    assert parse_output_basename(name) is None
    Path(staging, "x&A.fif@B.nc").write_text("partial")
    assert list(_scan_tree(str(tmp_path), recursive=True)) == []
    assert unit_state(prefix, "B").state == "missing"
    assert unit_state(prefix, "B").temps == (staging,)


def _companion_node(name: str, layout: str, text: str) -> None:
    """Register *name*: a ``.txt`` artifact plus a ``plot.txt`` companion (*layout*)."""

    @register_node(name=name, override=True)
    def node():
        def writer(p):
            if layout == "namespaced":
                sub = Path(p + ".figs")  # owned by the unit: starts with its prefix
            else:
                sub = Path(p).parent / "figs"  # shared by every unit of the folder
            sub.mkdir()
            Path(p).write_text(text)
            (sub / "plot.txt").write_text(text)
            Path(p + ".unreported").write_text("scratch")
            return [p, str(sub / "plot.txt")]

        return NodeResult({".txt": Artifact(text, writer)})


def test_writer_outputs_land_relative_to_target_and_unreported_files_are_dropped(tmp_path):
    _companion_node("rt_layout", "namespaced", "main")
    base = _source(tmp_path)
    run_derivative(_single("rt_layout"), "Lay", str(base), base)
    prefix = str(base) + "@Lay"
    assert read_manifest(prefix)["artifacts"] == {
        ".txt": ["source@Lay.txt", "source@Lay.txt.figs/plot.txt"]
    }
    assert (tmp_path / "source@Lay.txt.figs" / "plot.txt").read_text() == "main"
    assert not os.path.exists(prefix + ".txt.unreported")
    state = unit_state(prefix, "Lay")
    assert state.state == "done"
    assert state.temps == ()


def test_companion_outside_the_unit_namespace_fails_unit_and_replaces_nothing(tmp_path):
    """Astra's finding 4: ``figs/plot.txt`` is not owned by the unit."""
    (tmp_path / "figs").mkdir()
    (tmp_path / "figs" / "plot.txt").write_text("someone else's")
    _companion_node("rt_shared", "shared", "mine")
    base = _source(tmp_path)
    with pytest.raises(RuntimeError, match="does not start with the unit's prefix"):
        run_derivative(_single("rt_shared"), "Shared", str(base), base)
    prefix = str(base) + "@Shared"
    assert unit_state(prefix, "Shared").state == "error"
    assert not os.path.exists(prefix + ".txt")
    assert (tmp_path / "figs" / "plot.txt").read_text() == "someone else's"


@pytest.mark.parametrize("layout", ["namespaced", "shared"])
def test_two_derivatives_never_share_a_companion(tmp_path, layout):
    _companion_node("rt_first", layout, "First")
    _companion_node("rt_second", layout, "Second")
    base = _source(tmp_path)
    for node, name in (("rt_first", "First"), ("rt_second", "Second")):
        if layout == "shared":
            with pytest.raises(RuntimeError, match="does not start with the unit's prefix"):
                run_derivative(_single(node), name, str(base), base)
        else:
            run_derivative(_single(node), name, str(base), base)
    for name in ("First", "Second"):
        state = unit_state(f"{base}@{name}", name)
        if layout == "shared":
            assert state.state == "error"
            continue
        assert state.state == "done"
        assert [Path(p).read_text() for p in state.artifacts[".txt"]] == [name, name]
    assert not (tmp_path / "figs").exists()


@pytest.mark.parametrize("layout", ["namespaced", "shared"])
def test_two_sources_never_share_a_companion(tmp_path, layout):
    _companion_node("rt_comp", layout, "x")
    states = []
    for source in ("one", "two"):
        base = tmp_path / source
        base.write_text(source)
        if layout == "shared":
            with pytest.raises(RuntimeError, match="does not start with the unit's prefix"):
                run_derivative(_single("rt_comp"), "Comp", str(base), base)
        else:
            run_derivative(_single("rt_comp"), "Comp", str(base), base)
        states.append(unit_state(f"{base}@Comp", "Comp"))
    if layout == "shared":
        assert [s.state for s in states] == ["error", "error"]
        assert not (tmp_path / "figs").exists()
    else:
        assert [s.state for s in states] == ["done", "done"]
        assert states[0].artifacts[".txt"][1] != states[1].artifacts[".txt"][1]


@pytest.mark.parametrize(
    ("rel", "ok"),
    [
        ("x@A.fif", True),
        ("x@A-1.fif", True),
        ("x@A.EO-2.fif", True),
        ("x@A.fif.figs/p.png", True),
        ("x@A/p.png", True),
        ("x@A", False),
        ("x@A.done", False),
        ("x@A2.fif", False),
        ("x@A-x.fif", False),
        ("x@A.fif@B.nc", False),
        ("figs/p.png", False),
        ("y@A.fif", False),
    ],
)
def test_unit_namespace_rule(rel, ok):
    from neurodags.outputs import _unit_namespace_problem

    assert (_unit_namespace_problem(rel, "x@A") is None) is ok


# ---------------------------------------------------------------------------
# Fingerprints: a .done never certifies files another process replaced
# ---------------------------------------------------------------------------


@register_node(name="rt_pair", override=True)
def _rt_pair(text):
    return NodeResult(
        {
            ".a.txt": Artifact(text, lambda p: Path(p).write_text(text)),
            ".b.txt": Artifact(text, lambda p: Path(p).write_text(text)),
        }
    )


def _pause_replace(after: bool, paused, resume, then_exit: bool = False) -> None:
    """In a forked child: make the first ``os.replace`` pause (before or after it runs)."""
    real = os.replace
    calls = []

    def replace(src, dst):
        calls.append(dst)
        if len(calls) == 1 and not after:
            paused.set()
            assert resume.wait(30)
        real(src, dst)
        if len(calls) == 1 and then_exit:
            os._exit(23)
        if len(calls) == 1 and after:
            paused.set()
            assert resume.wait(30)

    os.replace = replace


@needs_fork
def test_interleaved_publishers_are_not_certified(tmp_path, procs):
    """Astra's schedule 1: A replaces .a.txt and pauses; B publishes both files and its .done;
    A replaces .b.txt and writes its .done. Files are B/A: the unit must not be done."""
    ctx = mp.get_context("fork")
    paused, resume = ctx.Event(), ctx.Event()
    base = _source(tmp_path)
    prefix = str(base) + "@Pair"

    def run_a():
        _pause_replace(True, paused, resume)
        run_derivative(_single("rt_pair", text="A"), "Pair", str(base), base)

    a = _start(procs, lambda: _exit_after(run_a))
    assert paused.wait(30)
    run_derivative(_single("rt_pair", text="BBBB"), "Pair", str(base), base)
    assert unit_state(prefix, "Pair").state == "done"
    resume.set()
    a.join(30)
    assert a.exitcode == 0

    assert Path(prefix + ".a.txt").read_text() == "BBBB"
    assert Path(prefix + ".b.txt").read_text() == "A"
    state = unit_state(prefix, "Pair")
    assert state.state == "incomplete"
    assert "modified after publication" in state.problem
    assert "cached" not in run_derivative(_single("rt_pair", text="A"), "Pair", str(base), base)
    assert unit_state(prefix, "Pair").state == "done"


@needs_fork
def test_publisher_dying_mid_publication_is_not_certified(tmp_path, procs):
    """Astra's schedule 2 with a real split FIF: B stages 9s and pauses before its first
    continuation replace; A publishes a complete recording of 1s; B replaces one part and
    dies. The recording mixes both: the unit must not be done."""
    ctx = mp.get_context("fork")
    paused, resume = ctx.Event(), ctx.Event()

    @register_node(name="rt_rec", override=True)
    def rec(value):
        raw = mne.io.RawArray(np.full((4, 200_000), value * 1e-6), mne.create_info(4, 1000, "eeg"))
        return NodeResult(
            {".fif": Artifact(raw, lambda p: save_mne(raw, p, split_size="2MB", verbose="error"))}
        )

    base = _source(tmp_path)
    prefix = str(base) + "@Rec"

    def run_b():
        _pause_replace(False, paused, resume, then_exit=True)
        run_derivative(_single("rt_rec", value=9), "Rec", str(base), base)

    b = _start(procs, lambda: _exit_after(run_b))
    assert paused.wait(60)
    time.sleep(1.2)  # A's computation starts later than B's: its files are newer
    run_derivative(_single("rt_rec", value=1), "Rec", str(base), base)
    assert unit_state(prefix, "Rec").state == "done"
    resume.set()
    b.join(30)
    assert b.exitcode == 23

    data = mne.io.read_raw_fif(prefix + ".fif", preload=True, verbose="error").get_data()
    assert {round(v * 1e6) for v in np.unique(data)} == {1, 9}  # really mixed
    state = unit_state(prefix, "Rec")
    assert state.state == "incomplete"
    assert "Rec-1.fif" in state.problem


def test_fingerprints_survive_metadata_preserving_copies_only(tmp_path):
    base = _source(tmp_path)
    run_derivative(_single("rt_pair", text="v"), "Fp", str(base), base)
    prefix = str(base) + "@Fp"

    copy = tmp_path / "copy"
    copy.mkdir()
    for path in tmp_path.glob("source@Fp.*"):
        shutil.copy2(path, copy / path.name)  # like rsync -a / cp -a
    assert unit_state(str(copy / "source@Fp"), "Fp").state == "done"

    st = os.stat(prefix + ".a.txt")
    os.utime(prefix + ".a.txt", ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))
    state = unit_state(prefix, "Fp")
    assert state.state == "incomplete"
    assert "modified after publication" in state.problem

    # Manifests written before fingerprints existed stay valid.
    done = Path(prefix + ".done")
    manifest = json.loads(done.read_text())
    del manifest["fingerprints"]
    done.write_text(json.dumps(manifest))
    assert unit_state(prefix, "Fp").state == "done"


# ---------------------------------------------------------------------------
# Migrate never certifies an interrupted 0.4.0 run
# ---------------------------------------------------------------------------


def test_migrate_does_not_certify_interrupted_run(tmp_path, monkeypatch):
    """Astra's finding 3: Ctrl-C after the first final replace leaves NEW/OLD and no .done;
    migrate on the (already naming_version 1) pipeline must not certify it."""
    import yaml

    from neurodags.migrate import apply_migration, plan_migration

    monkeypatch.chdir(tmp_path)
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw" / "source.txt").write_text("s")
    ds = {"name": "d", "source_root": "raw", "file_pattern": "raw/*.txt", "derivatives_path": "out"}
    (tmp_path / "ds.yml").write_text(yaml.safe_dump({"d": ds}))
    pipeline = tmp_path / "pl.yml"
    pipeline.write_text(
        yaml.safe_dump(
            {
                "naming_version": 1,
                "datasets": "ds.yml",
                "DerivativeDefinitions": {"Pair": _single("rt_pair", text="OLD")},
                "DerivativeList": ["Pair"],
            }
        )
    )
    out = tmp_path / "out"
    out.mkdir()
    base = out / "source.txt"
    prefix = str(base) + "@Pair"
    run_derivative(_single("rt_pair", text="OLD"), "Pair", str(base), base)

    from neurodags import outputs

    real = os.replace
    calls = []

    def interrupted(src, dst):
        real(src, dst)
        calls.append(dst)
        if len(calls) == 1:
            raise KeyboardInterrupt

    monkeypatch.setattr(outputs.os, "replace", interrupted)
    with pytest.raises(KeyboardInterrupt):
        run_derivative(
            {**_single("rt_pair", text="NEW"), "overwrite": True}, "Pair", str(base), base
        )
    monkeypatch.setattr(outputs.os, "replace", real)
    assert Path(prefix + ".a.txt").read_text() == "NEW"
    assert Path(prefix + ".b.txt").read_text() == "OLD"
    assert unit_state(prefix, "Pair").state == "incomplete"

    plan = plan_migration(str(pipeline))
    assert plan.mode == "tree-only"
    assert plan.refusals == []
    apply_migration(plan)
    assert unit_state(prefix, "Pair").state == "incomplete"
    assert not os.path.exists(prefix + ".done")
    reasons = [u.manifest_reason for u in plan.units("move", "identity")]
    assert any("recompute with neurodags run" in (r or "") for r in reasons)


def test_writer_reporting_a_file_outside_its_folder_fails_unit(tmp_path):
    elsewhere = tmp_path / "elsewhere.txt"

    @register_node(name="rt_outside", override=True)
    def outside():
        def writer(p):
            Path(p).write_text("main")
            elsewhere.write_text("x")
            return [p, str(elsewhere)]

        return NodeResult({".txt": Artifact("x", writer)})

    base = _source(tmp_path)
    with pytest.raises(RuntimeError, match="published atomically"):
        run_derivative(_single("rt_outside"), "Out", str(base), base)
    prefix = str(base) + "@Out"
    assert unit_state(prefix, "Out").state == "error"
    assert not os.path.exists(prefix + ".txt")
    assert unit_state(prefix, "Out").temps == ()


def test_interrupted_overwrite_of_skipped_unit_is_not_reported_skipped(tmp_path):
    """Astra's finding 3: an overwrite drops the old .skip before recomputing."""
    from neurodags.definitions import SkipDerivative

    @register_node(name="rt_skip", override=True)
    def skip():
        raise SkipDerivative("not applicable")

    base = _source(tmp_path)
    prefix = str(base) + "@Skp"
    assert "skipped" in run_derivative(_single("rt_skip"), "Skp", str(base), base)

    @register_node(name="rt_interrupt", override=True)
    def interrupt():
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_derivative({**_single("rt_interrupt"), "overwrite": True}, "Skp", str(base), base)
    assert unit_state(prefix, "Skp").state == "missing"


def test_status_cli_reports_staging_folders(tmp_path, capsys, monkeypatch):
    import yaml

    from neurodags.cli import main
    from neurodags.outputs import make_staging_dir

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
    (tmp_path / "out").mkdir()
    staging = make_staging_dir(str(tmp_path / "out" / "a.txt") + "@Val")

    rc = main(["status", str(pipeline), "--format", "json", "--include-files"])
    data = json.loads(capsys.readouterr().out)
    assert rc == 1  # missing; staging folders never change a unit's state
    assert data["staging_dirs"] == 1
    assert data["files"][0]["status"] == "missing"
    assert data["files"][0]["staging_dirs"] == [staging]

    assert main(["status", str(pipeline)]) == 1
    assert "1 staging folder(s)" in capsys.readouterr().out


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


@pytest.mark.parametrize(
    ("entry", "expected"),
    [
        ({"op": "move", "old": "/o/a@A.txt", "new": "/n/a@A.txt"}, ["/n/a@A.txt"]),
        ({"op": "mne-done", "new_parts": ["/n/a@A.fif", "/n/a@A-1.fif"]}, None),
        (
            {"op": "mne-staged", "new": "/n/a@A.fif", "parts": [["a@A.fif", 3, "x"]]},
            ["/n/a@A.fif"],
        ),
        (
            {"op": "split-move-start", "parts": [["/o/a@A.fif", "/n/a@A.fif"]]},
            ["/n/a@A.fif"],
        ),
        ({"op": "mne-start", "new": "/n/a@A.fif"}, []),  # nothing published yet
        ({"op": "manifest", "path": "/n/a@A.done"}, []),
    ],
)
def test_migrate_reads_its_own_journal_records(entry, expected):
    from neurodags.migrate import _journal_new_paths

    want = entry.get("new_parts") if expected is None else expected
    assert _journal_new_paths(entry) == want


def test_recording_container_source_publishes_next_to_its_prefix(tmp_path, monkeypatch):
    """A ``.ds`` directory input: the unit prefix is still a file prefix next to it."""
    import yaml

    from neurodags.orchestrators import run_pipeline

    monkeypatch.chdir(tmp_path)
    (tmp_path / "raw" / "sub-01_meg.ds").mkdir(parents=True)
    (tmp_path / "raw" / "sub-01_meg.ds" / "sub-01_meg.meg4").write_text("m")
    ds = {"name": "d", "source_root": "raw", "file_pattern": "raw/*.ds"}
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
    run_pipeline(str(pipeline), raise_on_error=True)
    prefix = str(tmp_path / "out" / "sub-01_meg.ds") + "@Val"
    state = unit_state(prefix, "Val")
    assert state.state == "done"
    assert state.temps == ()
    assert Path(prefix + ".txt").read_text() == "v"
    assert sorted(p.name for p in (tmp_path / "out").iterdir() if p.name != "code") == [
        "sub-01_meg.ds@Val.done",
        "sub-01_meg.ds@Val.txt",
    ]
