"""Output layout and completion manifests (spec 0.4.0 sections 3, 4 and 6)."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from neurodags.cli import main
from neurodags.dag import run_derivative
from neurodags.definitions import Artifact, NodeResult
from neurodags.iterators import discover_inputs
from neurodags.nodes import register_node
from neurodags.orchestrators import build_derivative_dataframe, run_pipeline
from neurodags.outputs import (
    LegacyOutputsError,
    build_manifest,
    manifest_artifacts,
    output_prefix,
    unit_state,
    write_manifest,
    writer_paths,
)
from neurodags.utils import DatasetConfigError
from neurodags.writers import save_mne

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chain(node: str = "dummy", args: dict | None = None, ref: str = "SourceFile") -> dict:
    return {
        "for_dataframe": True,
        "nodes": [
            {"id": 0, "derivative": ref},
            {"id": 1, "node": node, "args": args if args is not None else {"param1": "id.0"}},
        ],
    }


def _inputs(root: Path, rels=("sub-01/x0.txt", "sub-02/x1.txt")) -> None:
    for rel in rels:
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(rel)


def _project(
    root: Path,
    definitions: dict,
    *,
    dataset: dict | None = None,
    mount_point: str | None = None,
    derivative_list: list | None = None,
) -> Path:
    ds = dataset or {
        "name": "d",
        "source_root": "raw",
        "file_pattern": "raw/**/*.txt",
        "derivatives_path": "out",
    }
    (root / "ds.yml").write_text(yaml.safe_dump({"d": ds}))
    pipe = root / "pl.yml"
    pipe.write_text(
        yaml.safe_dump(
            {
                "naming_version": 1,
                "datasets": "ds.yml",
                "mount_point": mount_point,
                "DerivativeDefinitions": definitions,
                "DerivativeList": (
                    derivative_list if derivative_list is not None else list(definitions)
                ),
            }
        )
    )
    return pipe


def _files(root: Path) -> list[str]:
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())


def _final(plan_df, derivative: str, rel_suffix: str) -> dict:
    rows = plan_df[(plan_df["derivative"] == derivative)]
    for _, row in rows.iterrows():
        if row["file_path"].endswith(rel_suffix):
            return next(s for s in row["plan"] if s["kind"] == "derivative_output")
    raise AssertionError(f"no plan row for {derivative} {rel_suffix}")


# ---------------------------------------------------------------------------
# Manifest format and cache-hit rule (spec 6.1, 6.2)
# ---------------------------------------------------------------------------


def test_manifest_format(tmp_path):
    _inputs(tmp_path / "raw")
    pipe = _project(tmp_path, {"Fit_alpha-0p5_w-6": _chain()})
    run_pipeline(str(pipe), raise_on_error=True)

    done = tmp_path / "out" / "sub-01" / "x0.txt@Fit_alpha-0p5_w-6.done"
    manifest = json.loads(done.read_text())
    assert list(manifest) == [
        "manifest_version",
        "naming_version",
        "derivative",
        "source",
        "artifacts",
        "companions",
        "written",
        "origin",
    ]
    assert manifest["manifest_version"] == 1
    assert manifest["naming_version"] == 1
    assert manifest["derivative"] == "Fit_alpha-0p5_w-6"
    assert manifest["source"] == {"dataset": "d", "rel": "sub-01/x0.txt"}
    assert manifest["artifacts"] == {".message.txt": ["x0.txt@Fit_alpha-0p5_w-6.message.txt"]}
    assert manifest["companions"] == []
    assert manifest["written"].endswith("Z")
    assert manifest["origin"] == "run"
    # No temp file is left behind by the atomic write.
    assert not [p for p in done.parent.iterdir() if p.name.endswith(".tmp")]


def test_manifest_keeps_original_source_name(tmp_path):
    _inputs(tmp_path / "raw", ["sub-01/a@b.txt"])
    pipe = _project(tmp_path, {"Fit": _chain()})
    run_pipeline(str(pipe), raise_on_error=True)

    done = tmp_path / "out" / "sub-01" / "a@b.txt@Fit.done"
    assert json.loads(done.read_text())["source"]["rel"] == "sub-01/a@b.txt"


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        (lambda m: m.update(derivative="Other"), "derivative"),
        (lambda m: m.update(naming_version=0), "naming_version"),
        (lambda m: m.update(manifest_version=2), "manifest_version"),
        (lambda m: m["artifacts"].update({".nc": ["../escape.nc"]}), "unsafe"),
        (lambda m: m["artifacts"].update({".nc": ["gone.nc"]}), "missing"),
    ],
)
def test_invalid_manifest_is_not_a_cache_hit(tmp_path, change, expected):
    prefix = str(tmp_path / "x.vhdr@Fit")
    Path(prefix + ".message.txt").write_text("x")
    manifest = build_manifest(
        prefix,
        "Fit",
        dataset="d",
        rel="x.vhdr",
        artifacts={".message.txt": [prefix + ".message.txt"]},
    )
    write_manifest(prefix, manifest)
    assert manifest_artifacts(prefix, "Fit") == {".message.txt": [prefix + ".message.txt"]}

    change(manifest)
    write_manifest(prefix, manifest)
    assert manifest_artifacts(prefix, "Fit") is None
    state = unit_state(prefix, "Fit")
    assert state.state == "incomplete"
    assert expected in state.problem


def test_unparsable_manifest_is_incomplete(tmp_path):
    prefix = str(tmp_path / "x.vhdr@Fit")
    Path(prefix + ".done").write_text("{not json")
    assert unit_state(prefix, "Fit").state == "incomplete"


def test_unlisted_files_are_ignored_and_reported(tmp_path):
    prefix = str(tmp_path / "x.vhdr@Fit")
    Path(prefix + ".nc").write_text("x")
    Path(prefix + ".old.nc").write_text("stale")
    write_manifest(
        prefix,
        build_manifest(
            prefix, "Fit", dataset="d", rel="x.vhdr", artifacts={".nc": [prefix + ".nc"]}
        ),
    )
    state = unit_state(prefix, "Fit")
    assert state.state == "done"
    assert state.artifacts == {".nc": [prefix + ".nc"]}
    assert state.unlisted == (prefix + ".old.nc",)


def test_unit_state_precedence(tmp_path):
    prefix = str(tmp_path / "x.vhdr@Fit")
    assert unit_state(prefix, "Fit").state == "missing"
    Path(prefix + ".nc").write_text("x")
    assert unit_state(prefix, "Fit").state == "incomplete"
    Path(prefix + ".error").write_text("boom")
    assert unit_state(prefix, "Fit").state == "error"
    Path(prefix + ".skip").write_text("n/a")
    assert unit_state(prefix, "Fit").state == "skipped"


def test_output_prefix_validates_name_and_keeps_source_verbatim():
    assert output_prefix("/d/a@b.vhdr", "Fit") == "/d/a@b.vhdr@Fit"
    assert output_prefix("/d/a&b.vhdr", "Fit") == "/d/a&b.vhdr@Fit"
    with pytest.raises(ValueError, match=r"Fit\.v2"):
        output_prefix("/d/a.vhdr", "Fit.v2")


# ---------------------------------------------------------------------------
# Writer contract (spec 6.2)
# ---------------------------------------------------------------------------


def test_writer_paths_normalization(tmp_path):
    target = str(tmp_path / "x@Fit.fif")
    extra = str(tmp_path / "x@Fit-1.fif")
    Path(target).write_text("a")
    Path(extra).write_text("b")
    assert writer_paths(target, None) == [target]
    assert writer_paths(target, 12) == [target]  # e.g. open(path, "w").write(...)
    assert writer_paths(target, Path(target)) == [target]
    assert writer_paths(target, [Path(target), extra, target]) == [target, extra]
    with pytest.raises(RuntimeError, match="did not produce"):
        writer_paths(target, [target, str(tmp_path / "never.fif")])
    with pytest.raises(RuntimeError, match="did not produce"):
        writer_paths(str(tmp_path / "unwritten.fif"), None)


@register_node(name="_test_split_writer", override=True)
def _test_split_writer(n_parts: int = 2) -> NodeResult:
    def writer(path: str) -> list[str]:
        base, ext = os.path.splitext(path)
        written = [path] + [f"{base}-{i}{ext}" for i in range(1, n_parts)]
        for p in written:
            Path(p).write_text(p)
        return written

    return NodeResult(artifacts={".fif": Artifact(item="data", writer=writer)})


def test_fake_split_writer_parts_are_recorded(tmp_path):
    _inputs(tmp_path / "raw", ["sub-01/x0.txt"])
    pipe = _project(tmp_path, {"Clean": _chain("_test_split_writer", {"n_parts": 3})})
    run_pipeline(str(pipe), raise_on_error=True)

    out = tmp_path / "out" / "sub-01"
    manifest = json.loads((out / "x0.txt@Clean.done").read_text())
    assert manifest["artifacts"] == {
        ".fif": ["x0.txt@Clean.fif", "x0.txt@Clean-1.fif", "x0.txt@Clean-2.fif"]
    }
    prefix = str(out / "x0.txt@Clean")
    assert unit_state(prefix, "Clean").state == "done"
    # A missing split part makes the unit incomplete; the next run recomputes it.
    (out / "x0.txt@Clean-2.fif").unlink()
    assert unit_state(prefix, "Clean").state == "incomplete"
    run_pipeline(str(pipe), raise_on_error=True)
    assert unit_state(prefix, "Clean").state == "done"
    assert (out / "x0.txt@Clean-2.fif").is_file()


def test_save_mne_records_real_split_parts(tmp_path):
    mne = pytest.importorskip("mne")
    np = pytest.importorskip("numpy")
    info = mne.create_info(4, 1000.0, "eeg")
    raw = mne.io.RawArray(np.random.default_rng(0).standard_normal((4, 200_000)) * 1e-5, info)
    target = str(tmp_path / "x.vhdr@Clean.fif")

    written = save_mne(raw, target, split_size="2MB", verbose="error")

    assert written[0] == target
    assert written[1:] == [str(tmp_path / f"x.vhdr@Clean-{i}.fif") for i in range(1, len(written))]
    assert len(written) > 1
    assert writer_paths(target, written) == written
    reloaded = mne.io.read_raw_fif(target, verbose="error")
    assert reloaded.n_times == raw.n_times


def test_save_mne_fallback_ignores_stale_parts(tmp_path):
    """An MNE whose save() returns None: parts are found by name and freshness, not glob."""
    target = str(tmp_path / "x@Clean.fif")
    stale = tmp_path / "x@Clean-2.fif"
    stale.write_text("stale")
    os.utime(stale, (0, 0))

    class _OldMne:
        def save(self, path, **kwargs):
            Path(path).write_text("part0")
            Path(path.replace(".fif", "-1.fif")).write_text("part1")

    assert save_mne(_OldMne(), target) == [target, str(tmp_path / "x@Clean-1.fif")]


# ---------------------------------------------------------------------------
# Suffix validation before writers (spec 3)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("keys", [["ok", "bad key"], ["ok", "-x"], ["ok", "OK"]])
def test_invalid_artifact_suffix_fails_unit_and_writes_nothing(tmp_path, keys):
    _inputs(tmp_path / "raw", ["sub-01/x0.txt"])
    pipe = _project(tmp_path, {"Multi": _chain("dummy_multi", {"keys": keys})})
    run_pipeline(str(pipe))

    out = tmp_path / "out" / "sub-01"
    assert sorted(p.name for p in out.iterdir()) == ["x0.txt@Multi.error"]
    assert "Artifact suffix" in (out / "x0.txt@Multi.error").read_text()


# ---------------------------------------------------------------------------
# Write order, partial writes and crashes (spec 6.2)
# ---------------------------------------------------------------------------


@register_node(name="_test_two_artifacts", override=True)
def _test_two_artifacts(fail_second: bool = False) -> NodeResult:
    def first(path: str) -> None:
        Path(path).write_text("first")

    def second(path: str) -> None:
        if fail_second:
            raise KeyboardInterrupt("simulated crash")  # not caught: no .error marker
        Path(path).write_text("second")

    return NodeResult(
        artifacts={
            ".a.txt": Artifact(item="a", writer=first),
            ".b.txt": Artifact(item="b", writer=second),
        }
    )


def test_partial_multi_artifact_write_is_incomplete_and_recomputed(tmp_path):
    _inputs(tmp_path / "raw", ["sub-01/x0.txt"])
    crashing = _project(tmp_path, {"Two": _chain("_test_two_artifacts", {"fail_second": True})})
    with pytest.raises(KeyboardInterrupt):
        run_pipeline(str(crashing))

    out = tmp_path / "out" / "sub-01"
    prefix = str(out / "x0.txt@Two")
    assert sorted(p.name for p in out.iterdir()) == ["x0.txt@Two.a.txt"]
    assert unit_state(prefix, "Two").state == "incomplete"
    assert main(["status", str(crashing), "--format", "json"]) == 1

    fixed = _project(tmp_path, {"Two": _chain("_test_two_artifacts", {"fail_second": False})})
    plan = run_pipeline(str(fixed), dry_run=True)
    assert _final(plan, "Two", "x0.txt")["state"] == "incomplete"
    run_pipeline(str(fixed), raise_on_error=True)
    assert unit_state(prefix, "Two").state == "done"
    assert sorted(p.name for p in out.iterdir()) == [
        "x0.txt@Two.a.txt",
        "x0.txt@Two.b.txt",
        "x0.txt@Two.done",
    ]


def test_crash_before_manifest_leaves_no_done(tmp_path):
    """The old .done is deleted before writing; a crash before os.replace leaves none."""
    _inputs(tmp_path / "raw", ["sub-01/x0.txt"])
    definition = _chain()
    pipe = _project(tmp_path, {"Fit": definition})
    run_pipeline(str(pipe), raise_on_error=True)
    out = tmp_path / "out" / "sub-01"
    prefix = str(out / "x0.txt@Fit")
    assert unit_state(prefix, "Fit").state == "done"

    definition["overwrite"] = True
    pipe = _project(tmp_path, {"Fit": definition})
    with (
        patch("neurodags.outputs.os.replace", side_effect=KeyboardInterrupt("crash")),
        pytest.raises(KeyboardInterrupt),
    ):
        run_pipeline(str(pipe))
    assert not Path(prefix + ".done").exists()
    assert not [p for p in out.iterdir() if p.name.endswith(".tmp")]
    assert unit_state(prefix, "Fit").state == "incomplete"


def test_markers_are_mutually_exclusive(tmp_path):
    ref = tmp_path / "x.vhdr"
    prefix = str(tmp_path / "x.vhdr@Fit")
    Path(prefix + ".skip").write_text("old skip")
    Path(prefix + ".error").write_text("old error")
    run_derivative(
        {"overwrite": True, "nodes": [{"id": 0, "node": "dummy", "args": {}}]},
        "Fit",
        str(ref),
        reference_base=ref,
    )
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "x.vhdr@Fit.done",
        "x.vhdr@Fit.message.txt",
    ]


@register_node(name="_test_fail", override=True)
def _test_fail() -> NodeResult:
    raise ValueError("node failed")


def test_error_marker_replaces_done(tmp_path):
    ref = tmp_path / "x.vhdr"
    prefix = str(tmp_path / "x.vhdr@Fit")
    run_derivative({"nodes": [{"id": 0, "node": "dummy", "args": {}}]}, "Fit", str(ref), ref)
    assert unit_state(prefix, "Fit").state == "done"

    with pytest.raises(ValueError, match="node failed"):
        run_derivative(
            {"overwrite": True, "nodes": [{"id": 0, "node": "_test_fail", "args": {}}]},
            "Fit",
            str(ref),
            ref,
        )
    assert not Path(prefix + ".done").exists()
    assert unit_state(prefix, "Fit").state == "error"


# ---------------------------------------------------------------------------
# Layout stability (spec 0.2, 4.1)
# ---------------------------------------------------------------------------


def test_paths_unchanged_after_adding_files_narrowing_and_limits(tmp_path):
    _inputs(tmp_path / "raw", ["sub-01/ses-1/x0.txt"])
    pipe = _project(tmp_path, {"Fit": _chain()})
    run_pipeline(str(pipe), raise_on_error=True)
    first = tmp_path / "out" / "sub-01" / "ses-1" / "x0.txt@Fit.done"
    assert first.is_file()
    written = json.loads(first.read_text())["written"]

    # More files (a new branch raises the common ancestor): x0's outputs do not move.
    _inputs(tmp_path / "raw", ["sub-02/ses-1/x1.txt", "sub-02/ses-2/x2.txt"])
    run_pipeline(str(pipe), raise_on_error=True)
    assert json.loads(first.read_text())["written"] == written  # cache hit, no recompute
    assert _files(tmp_path / "out") == [
        "code/ds.yml",
        "code/neurodags_env.json",
        "code/pl.yml",
        "sub-01/ses-1/x0.txt@Fit.done",
        "sub-01/ses-1/x0.txt@Fit.message.txt",
        "sub-02/ses-1/x1.txt@Fit.done",
        "sub-02/ses-1/x1.txt@Fit.message.txt",
        "sub-02/ses-2/x2.txt@Fit.done",
        "sub-02/ses-2/x2.txt@Fit.message.txt",
    ]

    # A file limit and a narrowed file_pattern see the same prefixes, all cached.
    plan = run_pipeline(str(pipe), dry_run=True, max_files_per_dataset=1)
    assert _final(plan, "Fit", "x0.txt")["prefix"] == str(first)[: -len(".done")]
    narrowed = _project(
        tmp_path,
        {"Fit": _chain()},
        dataset={
            "name": "d",
            "source_root": "raw",
            "file_pattern": "raw/sub-02/ses-2/*.txt",
            "derivatives_path": "out",
        },
    )
    plan = run_pipeline(str(narrowed), dry_run=True)
    final = _final(plan, "Fit", "x2.txt")
    assert final["prefix"] == str(tmp_path / "out" / "sub-02" / "ses-2" / "x2.txt@Fit")
    assert final["state"] == "done"


def test_tree_copied_to_another_mount_verifies_via_manifest(tmp_path):
    local, hpc = tmp_path / "local", tmp_path / "hpc"
    _inputs(local / "raw")
    dataset = {
        "name": "d",
        "source_root": {"local": str(local / "raw"), "hpc": str(hpc / "raw")},
        "file_pattern": {"local": str(local / "raw/**/*.txt"), "hpc": str(hpc / "raw/**/*.txt")},
        "derivatives_path": {"local": str(local / "out"), "hpc": str(hpc / "out")},
    }
    pipe = _project(tmp_path, {"Fit": _chain()}, dataset=dataset, mount_point="local")
    run_pipeline(str(pipe), raise_on_error=True)

    shutil.copytree(local, hpc)
    shutil.rmtree(local)
    pipe = _project(tmp_path, {"Fit": _chain()}, dataset=dataset, mount_point="hpc")
    plan = run_pipeline(str(pipe), dry_run=True)
    assert {s["state"] for p in plan["plan"] for s in p if s["kind"] == "derivative_output"} == {
        "done"
    }
    df = build_derivative_dataframe(str(pipe), raise_on_error=True)
    assert len(df) == 2
    assert df["Fit.message.txt"].notna().all()


# ---------------------------------------------------------------------------
# Collision preflight (spec 4.3) and chained stages (spec 4.5)
# ---------------------------------------------------------------------------


def test_at_and_ampersand_sources_are_distinct_units(tmp_path, capsys):
    """Source basenames are verbatim: a@b.txt and a&b.txt in one folder do not collide."""
    _inputs(tmp_path / "raw", ["sub-01/a@b.txt", "sub-01/a&b.txt"])
    pipe = _project(tmp_path, {"Fit": _chain()})
    run_pipeline(str(pipe), raise_on_error=True)

    out = tmp_path / "out" / "sub-01"
    for source in ("a@b.txt", "a&b.txt"):
        manifest = json.loads((out / f"{source}@Fit.done").read_text())
        assert manifest["source"]["rel"] == f"sub-01/{source}"
        assert (out / f"{source}@Fit.message.txt").read_text() != ""
        assert unit_state(str(out / f"{source}@Fit"), "Fit").state == "done"

    assert main(["status", str(pipe), "--format", "json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["derivatives"]["Fit"]["done"] == 2
    df = build_derivative_dataframe(str(pipe))
    assert len(df) == 2
    # Both are cache hits on the next run (nothing rewritten).
    before = {p: p.stat().st_mtime_ns for p in out.iterdir()}
    run_pipeline(str(pipe), raise_on_error=True)
    assert {p: p.stat().st_mtime_ns for p in out.iterdir()} == before


def test_sources_mapping_to_same_reference_base_are_refused():
    """Genuine layout collisions (same normalized base) are still refused (spec 4.3)."""
    from neurodags.outputs import check_output_collisions

    check_output_collisions([("d: a@b.vhdr", "/o/a@b.vhdr"), ("d: a&b.vhdr", "/o/a&b.vhdr")])
    with pytest.raises(DatasetConfigError) as excinfo:
        check_output_collisions([("d1: x.vhdr", "/o/x.vhdr"), ("d2: x.vhdr", "/o/./x.vhdr")])
    message = str(excinfo.value)
    assert "'d1: x.vhdr' and 'd2: x.vhdr'" in message
    assert "same outputs" in message
    assert "'&'" not in message


def test_chained_stage_does_not_discover_its_own_outputs(tmp_path, capsys):
    _inputs(tmp_path / "data", ["sub-01/x0.txt"])
    dataset = {
        "name": "d",
        "source_root": "data",
        "file_pattern": "data/**/*.txt",  # also matches the outputs' .message.txt
        "derivatives_path": "data/derivatives",
    }
    pipe = _project(tmp_path, {"Fit": _chain()}, dataset=dataset)
    run_pipeline(str(pipe), raise_on_error=True)
    run_pipeline(str(pipe), raise_on_error=True)

    out = tmp_path / "data" / "derivatives"
    assert _files(out) == [
        "code/ds.yml",
        "code/neurodags_env.json",
        "code/pl.yml",
        "sub-01/x0.txt@Fit.done",
        "sub-01/x0.txt@Fit.message.txt",
    ]

    assert main(["validate", str(pipe)]) == 0
    err = capsys.readouterr().err
    assert "contains this dataset's own outputs" in err

    # An explicit exclude_pattern silences the warning.
    dataset["exclude_pattern"] = "data/derivatives/**/*"
    pipe = _project(tmp_path, {"Fit": _chain()}, dataset=dataset)
    assert main(["validate", str(pipe)]) == 0
    assert "own outputs" not in capsys.readouterr().err


def test_downstream_stage_reads_upstream_tree_but_not_its_code_folder(tmp_path):
    from neurodags.definitions import DatasetConfig

    upstream = tmp_path / "up"
    _inputs(upstream, ["sub-01/x0.txt@Fit.message.txt", "code/pl.txt", ".neurodags/s.txt"])
    up = DatasetConfig(
        name="up",
        source_root=str(tmp_path / "raw"),
        file_pattern=str(tmp_path / "raw" / "*.txt"),
        derivatives_path=str(upstream),
    )
    (tmp_path / "raw").mkdir()
    down = DatasetConfig(
        name="down",
        source_root=str(upstream),
        file_pattern=str(upstream / "**" / "*.txt"),
        derivatives_path=str(tmp_path / "stage2"),
    )
    enumeration = discover_inputs({"up": up, "down": down})
    assert [e.rel for e in enumeration.entries] == ["sub-01/x0.txt@Fit.message.txt"]


# ---------------------------------------------------------------------------
# status reports incomplete units
# ---------------------------------------------------------------------------


def test_status_reports_incomplete(tmp_path, capsys):
    _inputs(tmp_path / "raw")
    pipe = _project(tmp_path, {"Fit": _chain()})
    run_pipeline(str(pipe), raise_on_error=True)
    # 0.3.x-style output without a manifest, plus a stray file next to a done unit.
    (tmp_path / "out" / "sub-02" / "x1.txt@Fit.done").unlink()
    (tmp_path / "out" / "sub-01" / "x0.txt@Fit.old.txt").write_text("stray")

    assert main(["status", str(pipe), "--format", "json", "--include-files"]) == 1
    data = json.loads(capsys.readouterr().out)
    assert data["derivatives"]["Fit"]["done"] == 1
    assert data["derivatives"]["Fit"]["incomplete"] == 1
    assert data["incomplete_indices"] == [1]
    assert data["unlisted_files"] == 1

    assert main(["status", str(pipe), "--format", "indices"]) == 1
    assert capsys.readouterr().out == "1\n"

    assert main(["status", str(pipe)]) == 1
    out = capsys.readouterr().out
    assert "incomplete" in out
    assert "not listed in its .done" in out


# ---------------------------------------------------------------------------
# Chained names: verbatim sources, last '@'
# ---------------------------------------------------------------------------

CHAINED = {
    "name": "d",
    "source_root": "raw",
    "file_pattern": "raw/**/*@Clean.fif",
    "derivatives_path": "out",
}


def test_chained_outputs_keep_upstream_name_and_cache(tmp_path, capsys):
    _inputs(tmp_path / "raw", ["sub-01/x0.txt@Clean.fif", "sub-02/x1.txt@Clean.fif"])
    pipe = _project(tmp_path, {"Fit": _chain()}, dataset=CHAINED)
    run_pipeline(str(pipe), raise_on_error=True)
    out = tmp_path / "out"
    assert _files(out / "sub-01") == [
        "x0.txt@Clean.fif@Fit.done",
        "x0.txt@Clean.fif@Fit.message.txt",
    ]
    manifest = json.loads((out / "sub-01" / "x0.txt@Clean.fif@Fit.done").read_text())
    assert manifest["source"]["rel"] == "sub-01/x0.txt@Clean.fif"

    plan = run_pipeline(str(pipe), dry_run=True)
    final = _final(plan, "Fit", "x0.txt@Clean.fif")
    assert final["cached"] is True
    assert main(["status", str(pipe), "--format", "json"]) == 0
    assert json.loads(capsys.readouterr().out)["derivatives"]["Fit"]["done"] == 2
    df = build_derivative_dataframe(str(pipe))
    assert len(df) == 2


def test_status_ignores_downstream_outputs_in_the_same_folder(tmp_path, capsys):
    """For prefix x@Fit, x@Fit.message.txt@Power.nc matches the glob but is another unit."""
    _inputs(tmp_path / "raw")
    pipe = _project(tmp_path, {"Fit": _chain()})
    run_pipeline(str(pipe), raise_on_error=True)
    sub1 = tmp_path / "out" / "sub-01"
    sub2 = tmp_path / "out" / "sub-02"
    for folder, source in ((sub1, "x0.txt"), (sub2, "x1.txt")):
        for tail in (".nc", ".done", ".error", ".report.html"):
            (folder / f"{source}@Fit.message.txt@Power{tail}").write_text("downstream")
    # The upstream unit of sub-02 is gone: only its downstream outputs remain.
    for p in sub2.glob("x1.txt@Fit.*"):
        if "@Power" not in p.name:
            p.unlink()

    done = unit_state(str(sub1 / "x0.txt@Fit"), "Fit")
    assert done.state == "done"
    assert done.unlisted == ()
    missing = unit_state(str(sub2 / "x1.txt@Fit"), "Fit")
    assert missing.state == "missing"  # never done, never incomplete

    assert main(["status", str(pipe), "--format", "json", "--include-files"]) == 1
    data = json.loads(capsys.readouterr().out)
    assert data["derivatives"]["Fit"]["done"] == 1
    assert data["derivatives"]["Fit"]["missing"] == 1
    assert data["derivatives"]["Fit"]["incomplete"] == 0
    assert data["unlisted_files"] == 0
    # A truly unrecognized leftover of the unit is still reported.
    (sub1 / "x0.txt@Fit.old.txt").write_text("stray")
    assert unit_state(str(sub1 / "x0.txt@Fit"), "Fit").unlisted == (
        str(sub1 / "x0.txt@Fit.old.txt"),
    )


def _legacy_output(path: Path, text: str = "0.3.x output") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def test_unmigrated_ampersand_outputs_stop_with_a_pointer_to_migrate(tmp_path, capsys):
    """0.3.x outputs of a chained source ('&' spelling, no .done): loud, no recompute."""
    _inputs(tmp_path / "raw", ["sub-01/x0.txt@Clean.fif"])
    pipe = _project(tmp_path, {"Fit": _chain()}, dataset=CHAINED)
    legacy = _legacy_output(tmp_path / "out" / "sub-01" / "x0.txt&Clean.fif@Fit.message.txt")

    with pytest.raises(LegacyOutputsError, match="neurodags migrate") as excinfo:
        run_pipeline(str(pipe))
    assert str(legacy) in str(excinfo.value)
    assert isinstance(excinfo.value, DatasetConfigError)
    assert _files(tmp_path / "out") == ["sub-01/x0.txt&Clean.fif@Fit.message.txt"]
    assert main(["dry-run", str(pipe)]) == 2
    assert "neurodags migrate" in capsys.readouterr().err
    assert main(["status", str(pipe)]) == 2
    assert "neurodags migrate" in capsys.readouterr().err
    with pytest.raises(LegacyOutputsError):
        build_derivative_dataframe(str(pipe))

    # The 0.3.x on-disk token of a snake_case name is recognized too.
    pipe = _project(tmp_path, {"fit_x": _chain()}, dataset=CHAINED)
    legacy.rename(legacy.parent / "x0.txt&Clean.fif@FitX.message.txt")
    with pytest.raises(LegacyOutputsError):
        run_pipeline(str(pipe), dry_run=True)


def test_downstream_output_does_not_hide_an_unmigrated_parent(tmp_path):
    """A canonical downstream file (x@Clean.fif@Fit.message.txt@Power.nc) or a folder
    starting with the new prefix is not the parent unit's output: still loud."""
    _inputs(tmp_path / "raw", ["sub-01/x0.txt@Clean.fif"])
    pipe = _project(tmp_path, {"Fit": _chain()}, dataset=CHAINED)
    folder = tmp_path / "out" / "sub-01"
    _legacy_output(folder / "x0.txt&Clean.fif@Fit.message.txt")
    _legacy_output(folder / "x0.txt@Clean.fif@Fit.message.txt@Power.nc", "downstream")
    _legacy_output(folder / "x0.txt@Clean.fif@Fit.message.txt@Power.done", "{}")
    (folder / "x0.txt@Clean.fif@Fit.dir").mkdir()
    assert unit_state(str(folder / "x0.txt@Clean.fif@Fit"), "Fit").state == "missing"
    with pytest.raises(LegacyOutputsError):
        run_pipeline(str(pipe), dry_run=True)


@pytest.mark.parametrize("where", ["pilot-flat", "subdir-added", "upstream-renamed"])
def test_legacy_outputs_at_other_historical_locations_are_refused(tmp_path, where):
    """The guard scans the whole output tree, not only the unit's current folder."""
    dataset = dict(CHAINED)
    source = "sub-01/x0.txt@Clean.fif"
    legacy = tmp_path / "out" / "sub-01" / "x0.txt&Clean.fif@Fit.message.txt"
    if where == "pilot-flat":
        legacy = tmp_path / "out" / "x0.txt&Clean.fif@Fit.message.txt"
    elif where == "subdir-added":
        dataset["derivatives_subdir"] = "stage2"
    else:  # the upstream name changed and the pattern was edited accordingly
        source = "sub-01/x0.txt@clean.fif"
        dataset["file_pattern"] = "raw/**/*@clean.fif"
    _inputs(tmp_path / "raw", [source])
    _legacy_output(legacy)
    pipe = _project(tmp_path, {"Fit": _chain()}, dataset=dataset)
    with pytest.raises(LegacyOutputsError) as excinfo:
        run_pipeline(str(pipe), dry_run=True)
    assert str(legacy) in str(excinfo.value)


def test_real_ampersand_source_is_not_legacy_whatever_is_selected(tmp_path):
    """A 0.4.0 unit of a real 'a&b.txt' source stays valid when discovery is narrowed to
    the 'a@b.txt' source (ownership comes from the .done / the source file, not from the
    current selection)."""
    _inputs(tmp_path / "raw", ["sub-01/a&b.txt", "sub-01/a@b.txt"])
    dataset = {**CHAINED, "file_pattern": "raw/**/*&b.txt"}
    pipe = _project(tmp_path, {"Fit": _chain()}, dataset=dataset)
    run_pipeline(str(pipe), raise_on_error=True)
    out = tmp_path / "out" / "sub-01"
    assert (out / "a&b.txt@Fit.done").exists()
    before = {p.name: p.stat().st_mtime_ns for p in out.iterdir()}

    dataset["file_pattern"] = "raw/**/*@b.txt"
    pipe = _project(tmp_path, {"Fit": _chain()}, dataset=dataset)
    run_pipeline(str(pipe), raise_on_error=True)
    assert unit_state(str(out / "a@b.txt@Fit"), "Fit").state == "done"
    assert {n: (out / n).stat().st_mtime_ns for n in before} == before

    # Even without its .done (an interrupted 0.4.0 write), the real source owns it.
    (out / "a&b.txt@Fit.done").unlink()
    run_pipeline(str(pipe), dry_run=True)


def test_migrated_unit_claims_its_leftover_copies_but_not_other_folders(tmp_path):
    _inputs(tmp_path / "raw", ["sub-01/x0.txt@Clean.fif", "sub-02/x0.txt@Clean.fif"])
    pipe = _project(tmp_path, {"Fit": _chain()}, dataset=CHAINED)
    run_pipeline(str(pipe), raise_on_error=True)  # both units done (0.4.0 names)
    # A pilot-era flat copy left by migrate as a duplicate: claimed by the migrated unit.
    _legacy_output(tmp_path / "out" / "x0.txt&Clean.fif@Fit.message.txt")
    # Unrelated files: another pipeline's name, a non-output name.
    _legacy_output(tmp_path / "out" / "sub-01" / "x0.txt&Clean.fif@Other.nc")
    _legacy_output(tmp_path / "out" / "sub-01" / "notes&todo.txt")
    run_pipeline(str(pipe), dry_run=True)

    # A legacy copy in a folder whose unit has no .done is not claimed by another folder.
    (tmp_path / "out" / "sub-02" / "x0.txt@Clean.fif@Fit.done").unlink()
    stale = _legacy_output(tmp_path / "out" / "sub-02" / "x0.txt&Clean.fif@Fit.message.txt")
    with pytest.raises(LegacyOutputsError) as excinfo:
        run_pipeline(str(pipe), dry_run=True)
    message = str(excinfo.value)
    assert str(stale) in message
    assert "move them out of the output tree" in message


def test_legacy_outputs_next_to_sources_are_refused(tmp_path):
    _inputs(tmp_path / "raw", ["sub-01/x0.txt@Clean.fif"])
    dataset = {k: v for k, v in CHAINED.items() if k != "derivatives_path"}
    pipe = _project(tmp_path, {"Fit": _chain()}, dataset=dataset)
    legacy = _legacy_output(tmp_path / "raw" / "sub-01" / "x0.txt&Clean.fif@Fit.message.txt")
    with pytest.raises(LegacyOutputsError):
        run_pipeline(str(pipe), dry_run=True)
    legacy.unlink()
    run_pipeline(str(pipe), raise_on_error=True)
    assert unit_state(str(tmp_path / "raw/sub-01/x0.txt@Clean.fif@Fit"), "Fit").state == "done"
