"""Tests for exact artifact references (spec section 5).

When a node produces NodeResult with multiple artifacts (e.g. a splitter that
outputs one artifact per condition), downstream derivatives select one with

    derivative: SplitterDerivative.condA.txt

The text after the name must equal an artifact suffix key exactly, on every path:
cached (``.done`` manifest), freshly computed, a recursed sub-derivative that
returned its manifest, and ``save: false`` in memory. A missing suffix, or a bare
name for a derivative with several artifacts, raises ``ArtifactSelectionError``
with the same message on every path; there is no fallback.
"""

from __future__ import annotations

import re
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest

from neurodags import dag
from neurodags.dag import ArtifactSelectionError, run_derivative, select_artifact
from neurodags.definitions import NodeResult
from neurodags.derivatives import unregister_derivative
from neurodags.derivatives.pipeline import register_derivatives_from_dict

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ref_base(tmp_path: Path) -> Path:
    ref = tmp_path / "subject" / "sample"
    ref.parent.mkdir(parents=True, exist_ok=True)
    return ref


def _pipeline_cfg(splitter_keys: list[str], consumer_suffix: str) -> dict:
    """Two-derivative config: MultiSplitter (dummy_multi) + Consumer (reads one artifact)."""
    return {
        "DerivativeDefinitions": {
            "MultiSplitter": {
                "overwrite": False,
                "nodes": [
                    {"id": 0, "node": "dummy_multi", "args": {"keys": splitter_keys}},
                ],
            },
            "Consumer": {
                "overwrite": False,
                "nodes": [
                    {"id": 0, "derivative": f"MultiSplitter.{consumer_suffix}"},
                    {"id": 1, "node": "dummy", "args": {"param1": "id.0"}},
                ],
            },
        }
    }


@pytest.fixture
def registered_pipeline(request):
    """Register a pipeline config into the global derivative registry and clean up after."""
    cfg = request.param
    register_derivatives_from_dict(cfg)
    yield cfg
    for name in cfg["DerivativeDefinitions"]:
        unregister_derivative(name)


def _run(tmp_path: Path, derivative: str, cfg: dict) -> NodeResult:
    ref = _ref_base(tmp_path)
    result = run_derivative(
        cfg["DerivativeDefinitions"][derivative],
        derivative_name=derivative,
        file_path="input.vhdr",
        reference_base=ref,
    )
    assert isinstance(result, NodeResult), f"Expected NodeResult, got {type(result)}"
    return result


# ---------------------------------------------------------------------------
# Core: in-memory artifact selection
# ---------------------------------------------------------------------------


def test_splitter_produces_multiple_artifacts(tmp_path):
    """dummy_multi returns one artifact per key — no registration needed."""
    cfg = _pipeline_cfg(["alpha", "beta"], "alpha.txt")
    result = _run(tmp_path, "MultiSplitter", cfg)
    assert ".alpha.txt" in result.artifacts
    assert ".beta.txt" in result.artifacts
    assert result.artifacts[".alpha.txt"].item == "alpha"
    assert result.artifacts[".beta.txt"].item == "beta"


@pytest.mark.parametrize(
    "registered_pipeline",
    [_pipeline_cfg(["alpha", "beta"], "alpha.txt")],
    indirect=True,
)
def test_consumer_receives_selected_artifact_in_memory(tmp_path, registered_pipeline):
    """Consumer with derivative: MultiSplitter.alpha.txt gets only alpha in-memory."""
    result = _run(tmp_path, "Consumer", registered_pipeline)
    assert "alpha" in result.artifacts[".message.txt"].item
    assert "beta" not in result.artifacts[".message.txt"].item


@pytest.mark.parametrize(
    "registered_pipeline",
    [_pipeline_cfg(["alpha", "beta"], "beta.txt")],
    indirect=True,
)
def test_consumer_selects_second_artifact(tmp_path, registered_pipeline):
    """Selecting the second artifact works regardless of insertion order."""
    result = _run(tmp_path, "Consumer", registered_pipeline)
    assert "beta" in result.artifacts[".message.txt"].item
    assert "alpha" not in result.artifacts[".message.txt"].item


@pytest.mark.parametrize(
    "registered_pipeline",
    [_pipeline_cfg(["cond_a", "cond_b", "cond_c"], "cond_b.txt")],
    indirect=True,
)
def test_consumer_three_artifacts_middle(tmp_path, registered_pipeline):
    """Selection works for middle element in a three-artifact splitter."""
    result = _run(tmp_path, "Consumer", registered_pipeline)
    assert "cond_b" in result.artifacts[".message.txt"].item


# ---------------------------------------------------------------------------
# On-disk path: suffix selection from cached files
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "registered_pipeline",
    [_pipeline_cfg(["alpha", "beta"], "alpha.txt")],
    indirect=True,
)
def test_consumer_selects_artifact_from_disk(tmp_path, registered_pipeline):
    """After splitter writes to disk, suffix selection loads the right cached file."""
    cfg = registered_pipeline
    ref = _ref_base(tmp_path)

    # Run splitter so artifacts land on disk
    run_derivative(
        cfg["DerivativeDefinitions"]["MultiSplitter"],
        derivative_name="MultiSplitter",
        file_path="input.vhdr",
        reference_base=ref,
    )
    alpha_path = Path(f"{ref}@MultiSplitter.alpha.txt")
    beta_path = Path(f"{ref}@MultiSplitter.beta.txt")
    assert alpha_path.exists(), "splitter must write alpha artifact"
    assert beta_path.exists(), "splitter must write beta artifact"
    assert alpha_path.read_text() == "alpha"

    result = run_derivative(
        cfg["DerivativeDefinitions"]["Consumer"],
        derivative_name="Consumer",
        file_path="input.vhdr",
        reference_base=ref,
    )
    assert isinstance(result, NodeResult)
    assert "alpha" in result.artifacts[".message.txt"].item


# ---------------------------------------------------------------------------
# Consistency: in-memory and on-disk selection agree
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "registered_pipeline",
    [_pipeline_cfg(["alpha", "beta"], "beta.txt")],
    indirect=True,
)
def test_in_memory_and_disk_selection_agree(tmp_path, registered_pipeline):
    """In-memory and cached-disk paths produce identical Consumer outputs."""
    cfg = registered_pipeline
    ref = _ref_base(tmp_path)

    # First Consumer run: splitter uncached → in-memory selection
    result_inmem = run_derivative(
        cfg["DerivativeDefinitions"]["Consumer"],
        derivative_name="Consumer",
        file_path="input.vhdr",
        reference_base=ref,
    )

    # Force Consumer re-run; splitter artifacts now on disk
    cfg_overwrite = {
        **cfg,
        "DerivativeDefinitions": {
            **cfg["DerivativeDefinitions"],
            "Consumer": {**cfg["DerivativeDefinitions"]["Consumer"], "overwrite": True},
        },
    }
    register_derivatives_from_dict(cfg_overwrite)
    result_disk = run_derivative(
        cfg_overwrite["DerivativeDefinitions"]["Consumer"],
        derivative_name="Consumer",
        file_path="input.vhdr",
        reference_base=ref,
    )

    inmem_msg = result_inmem.artifacts[".message.txt"].item
    disk_msg = result_disk.artifacts[".message.txt"].item
    assert "beta" in inmem_msg
    assert "beta" in disk_msg


# ---------------------------------------------------------------------------
# Child overwrite flag: parent overwrite=True must not force child to re-run
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "registered_pipeline",
    [_pipeline_cfg(["alpha", "beta"], "alpha.txt")],
    indirect=True,
)
def test_child_overwrite_false_respected_when_parent_overwrite_true(tmp_path, registered_pipeline):
    """Consumer(overwrite=True) must honour MultiSplitter(overwrite=False): cached child not re-run."""
    cfg = registered_pipeline
    ref = _ref_base(tmp_path)

    # Pre-cache MultiSplitter artifacts on disk
    run_derivative(
        cfg["DerivativeDefinitions"]["MultiSplitter"],
        derivative_name="MultiSplitter",
        file_path="input.vhdr",
        reference_base=ref,
    )
    alpha_path = Path(f"{ref}@MultiSplitter.alpha.txt")
    assert alpha_path.exists()
    orig_mtime = alpha_path.stat().st_mtime

    # Consumer with overwrite=True; MultiSplitter stays overwrite=False
    cfg_consumer_overwrite = {
        **cfg,
        "DerivativeDefinitions": {
            **cfg["DerivativeDefinitions"],
            "Consumer": {**cfg["DerivativeDefinitions"]["Consumer"], "overwrite": True},
        },
    }
    register_derivatives_from_dict(cfg_consumer_overwrite)

    result = run_derivative(
        cfg_consumer_overwrite["DerivativeDefinitions"]["Consumer"],
        derivative_name="Consumer",
        file_path="input.vhdr",
        reference_base=ref,
    )

    # Child artifacts must not be re-written
    assert alpha_path.stat().st_mtime == orig_mtime
    # Consumer still produces a valid result using the cached child artifact
    assert isinstance(result, NodeResult)
    assert "alpha" in result.artifacts[".message.txt"].item


# ---------------------------------------------------------------------------
# Exact selection on every path (spec section 5)
# ---------------------------------------------------------------------------

#: How the referenced derivative (MultiSplitter) reaches the consumer:
#: - cached: done unit, read from its ``.done`` manifest;
#: - fresh: computed in the same run and saved (``save: true``);
#: - in_memory: computed in the same run with ``save: false``;
#: - recursed_manifest: the parent's lookup missed, the recursed child early-returned its
#:   manifest (``{"cached": [...], "artifacts": {...}}``).
PATHS = ["cached", "fresh", "in_memory", "recursed_manifest"]


def _ref_cfg(ref: str, keys: list[str], *, splitter_save: bool = True) -> dict:
    return {
        "DerivativeDefinitions": {
            "MultiSplitter": {
                "overwrite": False,
                "save": splitter_save,
                "nodes": [{"id": 0, "node": "dummy_multi", "args": {"keys": keys}}],
            },
            "Consumer": {
                "overwrite": False,
                "nodes": [
                    {"id": 0, "derivative": ref},
                    {"id": 1, "node": "dummy", "args": {"param1": "id.0"}},
                ],
            },
        }
    }


@contextmanager
def _registered(cfg: dict):
    register_derivatives_from_dict(cfg)
    try:
        yield cfg
    finally:
        for name in cfg["DerivativeDefinitions"]:
            unregister_derivative(name)


def _consume(tmp_path: Path, ref: str, keys: list[str], path: str):
    """Run Consumer with ``derivative: <ref>`` with MultiSplitter reached through *path*."""
    cfg = _ref_cfg(ref, keys, splitter_save=path != "in_memory")
    base = _ref_base(tmp_path)
    defs = cfg["DerivativeDefinitions"]
    with _registered(cfg):
        if path in {"cached", "recursed_manifest"}:
            run_derivative(defs["MultiSplitter"], "MultiSplitter", "input.vhdr", base)
            assert Path(f"{base}@MultiSplitter.done").exists()
        if path != "recursed_manifest":
            return run_derivative(defs["Consumer"], "Consumer", "input.vhdr", base)

        real = dag._cached_unit
        seen: list[str] = []

        def miss_first_child_lookup(prefix, name):
            if name == "MultiSplitter" and name not in seen:
                seen.append(name)
                return None
            return real(prefix, name)

        with patch("neurodags.dag._cached_unit", side_effect=miss_first_child_lookup):
            result = run_derivative(defs["Consumer"], "Consumer", "input.vhdr", base)
        assert seen == ["MultiSplitter"], "the recursed child must early-return its manifest"
        return result


@pytest.mark.parametrize("path", PATHS)
def test_exact_suffix_selected_on_every_path(tmp_path, path):
    result = _consume(tmp_path, "MultiSplitter.beta.txt", ["alpha", "beta"], path)
    assert isinstance(result, NodeResult)
    message = result.artifacts[".message.txt"].item
    assert "beta" in message
    assert "alpha" not in message


def _selection_error(tmp_path: Path, ref: str, keys: list[str], path: str) -> str:
    with pytest.raises(ArtifactSelectionError) as excinfo:
        _consume(tmp_path / path, ref, keys, path)
    return str(excinfo.value)


@pytest.mark.parametrize(
    "ref",
    [
        "MultiSplitter.gamma.txt",  # absent suffix
        "MultiSplitter.txt",  # ends both artifacts' file names: no endswith matching
        "MultiSplitter.Beta.txt",  # case differs: string equality
    ],
)
def test_missing_suffix_same_error_on_every_path(tmp_path, ref):
    messages = {path: _selection_error(tmp_path, ref, ["alpha", "beta"], path) for path in PATHS}
    assert len(set(messages.values())) == 1, messages
    message = messages["cached"]
    assert "'Consumer'" in message
    assert repr(ref) in message
    assert "available suffixes: '.alpha.txt', '.beta.txt'" in message


def test_missing_suffix_writes_consumer_error_marker(tmp_path):
    message = _selection_error(tmp_path, "MultiSplitter.gamma.txt", ["alpha", "beta"], "cached")
    base = _ref_base(tmp_path / "cached")
    error_file = Path(f"{base}@Consumer.error")
    assert error_file.exists()
    assert message in error_file.read_text()
    assert not Path(f"{base}@Consumer.done").exists()
    assert not Path(f"{base}@Consumer.message.txt").exists()


def test_missing_suffix_save_false_consumer_writes_no_marker(tmp_path):
    cfg = _ref_cfg("MultiSplitter.gamma.txt", ["alpha", "beta"])
    cfg["DerivativeDefinitions"]["Consumer"]["save"] = False
    base = _ref_base(tmp_path)
    with _registered(cfg), pytest.raises(ArtifactSelectionError):
        run_derivative(cfg["DerivativeDefinitions"]["Consumer"], "Consumer", "input.vhdr", base)
    assert not Path(f"{base}@Consumer.error").exists()


# ---------------------------------------------------------------------------
# Bare references: only for single-artifact derivatives
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", PATHS)
def test_bare_reference_to_multi_artifact_derivative_errors(tmp_path, path):
    message = _selection_error(tmp_path, "MultiSplitter", ["alpha", "beta"], path)
    assert "has no artifact suffix" in message
    assert "2 artifacts" in message
    assert "available suffixes: '.alpha.txt', '.beta.txt'" in message


def test_bare_reference_error_same_on_every_path(tmp_path):
    messages = {
        path: _selection_error(tmp_path, "MultiSplitter", ["a", "b"], path) for path in PATHS
    }
    assert len(set(messages.values())) == 1, messages


@pytest.mark.parametrize("path", PATHS)
def test_bare_reference_to_single_artifact_derivative(tmp_path, path):
    result = _consume(tmp_path, "MultiSplitter", ["alpha"], path)
    assert "alpha" in result.artifacts[".message.txt"].item


@pytest.mark.parametrize("path", PATHS)
def test_bare_and_suffixed_reference_agree_for_single_artifact(tmp_path, path):
    bare = _consume(tmp_path / "bare", "MultiSplitter", ["alpha"], path)
    full = _consume(tmp_path / "full", "MultiSplitter.alpha.txt", ["alpha"], path)
    # Cached paths pass the file path, which differs only by the test's folder.
    bare_message = bare.artifacts[".message.txt"].item.replace(str(tmp_path / "bare"), "<tmp>")
    full_message = full.artifacts[".message.txt"].item.replace(str(tmp_path / "full"), "<tmp>")
    assert bare_message == full_message


def test_bare_reference_cached_passes_main_file(tmp_path):
    """A cached single-artifact derivative is passed as its main file's path."""
    result = _consume(tmp_path, "MultiSplitter", ["alpha"], "cached")
    base = _ref_base(tmp_path)
    assert f"{base}@MultiSplitter.alpha.txt" in result.artifacts[".message.txt"].item


@pytest.mark.parametrize("path", ["fresh", "in_memory"])
def test_bare_reference_to_derivative_without_artifacts_errors(tmp_path, path):
    message = _selection_error(tmp_path, "MultiSplitter", [], path)
    assert "0 artifacts" in message
    assert "available suffixes: none" in message


# ---------------------------------------------------------------------------
# Dry run reports the selection without raising
# ---------------------------------------------------------------------------


def _dry_run_entry(tmp_path: Path, ref: str) -> dict:
    cfg = _ref_cfg(ref, ["alpha", "beta"])
    base = _ref_base(tmp_path)
    defs = cfg["DerivativeDefinitions"]
    with _registered(cfg):
        run_derivative(defs["MultiSplitter"], "MultiSplitter", "input.vhdr", base)
        plan = run_derivative(defs["Consumer"], "Consumer", "input.vhdr", base, dry_run=True)
    assert not Path(f"{base}@Consumer.error").exists()
    return next(e for e in plan["plan"] if e["id"] == 0)


def test_dry_run_reports_selected_artifact(tmp_path):
    entry = _dry_run_entry(tmp_path, "MultiSplitter.beta.txt")
    assert entry["cached"] is True
    assert entry["paths"] == [f"{_ref_base(tmp_path)}@MultiSplitter.beta.txt"]
    assert "selection_problem" not in entry


def test_dry_run_reports_selection_problem(tmp_path):
    entry = _dry_run_entry(tmp_path, "MultiSplitter.txt")
    assert entry["cached"] is False
    assert entry["paths"] == []
    assert "available suffixes: '.alpha.txt', '.beta.txt'" in entry["selection_problem"]


# ---------------------------------------------------------------------------
# select_artifact (pure)
# ---------------------------------------------------------------------------


def test_select_artifact_exact_and_bare():
    arts = {".fif": "clean", ".report.html": "report"}
    kw = {"consumer": "C", "ref": "x"}
    assert select_artifact(arts, "Clean", ".fif", **kw) == (".fif", "clean")
    assert select_artifact(arts, "Clean", ".report.html", **kw) == (".report.html", "report")
    assert select_artifact({".nc": 1}, "Fit", None, **kw) == (".nc", 1)
    with pytest.raises(ArtifactSelectionError, match=re.escape("'.fif', '.report.html'")):
        select_artifact(arts, "Clean", ".html", **kw)
    with pytest.raises(ArtifactSelectionError, match="exactly one"):
        select_artifact(arts, "Clean", None, **kw)


def test_select_artifact_requires_full_suffix():
    """``.EO_baseline.fif`` must be written in full: ``.fif`` does not match it."""
    arts = {".EO_baseline.fif": "eo", ".EC_baseline.fif": "ec"}
    with pytest.raises(ArtifactSelectionError) as excinfo:
        select_artifact(arts, "Split", ".fif", consumer="C", ref="Split.fif")
    assert "'Split.EC_baseline.fif'" in str(excinfo.value)
    assert select_artifact(arts, "Split", ".EO_baseline.fif", consumer="C", ref="r")[1] == "eo"


def test_artifact_selection_error_is_lookup_error():
    assert issubclass(ArtifactSelectionError, LookupError)
