"""Tests for in-memory multi-artifact selection via derivative reference suffix.

When a node produces NodeResult with multiple artifacts (e.g. a splitter that
outputs one artifact per condition), downstream derivatives can select a
specific artifact using the dot-extension syntax:

    derivative: SplitterDerivative.condA.txt

Previously this selection only worked for on-disk (cached) artifacts.
The patch makes it work identically for in-memory (uncached) results.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from neurodags.dag import run_derivative
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


@pytest.mark.parametrize(
    "registered_pipeline",
    [_pipeline_cfg(["alpha", "beta"], "alpha.txt")],
    indirect=True,
)
def test_no_suffix_passes_full_node_result(tmp_path, registered_pipeline):
    """derivative: MultiSplitter (no suffix) passes the full NodeResult unchanged."""
    cfg = registered_pipeline
    cfg["DerivativeDefinitions"]["NoSuffixConsumer"] = {
        "overwrite": False,
        "nodes": [
            {"id": 0, "derivative": "MultiSplitter"},
            {"id": 1, "node": "dummy", "args": {"param1": "id.0"}},
        ],
    }
    register_derivatives_from_dict(cfg)
    try:
        ref = _ref_base(tmp_path)
        result = run_derivative(
            cfg["DerivativeDefinitions"]["NoSuffixConsumer"],
            derivative_name="NoSuffixConsumer",
            file_path="input.vhdr",
            reference_base=ref,
        )
        assert isinstance(result, NodeResult)
        assert ".message.txt" in result.artifacts
    finally:
        unregister_derivative("NoSuffixConsumer")


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
# Missing suffix: warning + fallback
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "registered_pipeline",
    [_pipeline_cfg(["alpha", "beta"], "gamma.txt")],
    indirect=True,
)
def test_missing_suffix_falls_back_to_full_node_result(tmp_path, registered_pipeline):
    """Requesting absent suffix logs a warning and passes full NodeResult."""
    from unittest.mock import patch

    with patch("neurodags.dag.log") as mock_log:
        result = run_derivative(
            registered_pipeline["DerivativeDefinitions"]["Consumer"],
            derivative_name="Consumer",
            file_path="input.vhdr",
            reference_base=_ref_base(tmp_path),
        )

    assert isinstance(result, NodeResult)
    mock_log.warning.assert_called_once()
    call_kwargs = mock_log.warning.call_args
    assert "Artifact suffix not found" in call_kwargs.args[0]


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


@pytest.mark.parametrize(
    "registered_pipeline",
    [_pipeline_cfg(["alpha", "beta"], "alpha.txt")],
    indirect=True,
)
def test_cached_dict_resolved_when_ext_not_in_child_artifacts(tmp_path, registered_pipeline):
    """When Consumer requests a suffix that child never produced, the cached-dict path resolves
    to the first available cached artifact without raising."""
    cfg = registered_pipeline
    ref = _ref_base(tmp_path)

    # Pre-cache MultiSplitter (produces alpha.txt and beta.txt, not gamma.txt)
    run_derivative(
        cfg["DerivativeDefinitions"]["MultiSplitter"],
        derivative_name="MultiSplitter",
        file_path="input.vhdr",
        reference_base=ref,
    )

    # Consumer requests .gamma.txt — an artifact that MultiSplitter never produces.
    # Outer ext-filter → candidates=[] → cached_here=False → recurse into child.
    # Child has overwrite=False and its own artifacts are cached → returns {"cached": [...]}.
    # The new code must resolve that dict without raising.
    gamma_consumer_def = {
        "overwrite": False,
        "nodes": [
            {"id": 0, "derivative": "MultiSplitter.gamma.txt"},
            {"id": 1, "node": "dummy", "args": {"param1": "id.0"}},
        ],
    }
    cfg["DerivativeDefinitions"]["GammaConsumer"] = gamma_consumer_def
    register_derivatives_from_dict(cfg)
    try:
        result = run_derivative(
            gamma_consumer_def,
            derivative_name="GammaConsumer",
            file_path="input.vhdr",
            reference_base=ref,
        )
        assert isinstance(result, NodeResult)
    finally:
        unregister_derivative("GammaConsumer")


def test_cached_dict_no_ext_resolved_to_first_path(tmp_path):
    """No-ext branch of the cached-dict handler: sub-derivative returns {"cached": [path1, ...]}
    and the resolved store entry is path1 (a string), not the raw dict."""
    fake_cached_path = str(tmp_path / "subject@Child.txt")
    Path(fake_cached_path).write_text("content")

    consumer_def = {
        "overwrite": True,  # bypass Consumer's own early-return
        "save": False,
        "nodes": [
            {"id": 0, "derivative": "Child"},  # no ext → else branch in cached-dict handler
            {"id": 1, "node": "dummy", "args": {"param1": "id.0"}},
        ],
    }
    child_def = {"overwrite": False, "nodes": [{"id": 0, "node": "dummy", "args": {}}]}

    from neurodags.derivatives import register_derivative_with_name

    register_derivative_with_name("Child", lambda: None, definition=child_def)
    try:
        ref = tmp_path / "subject" / "sample"
        ref.parent.mkdir(parents=True, exist_ok=True)

        original_run = run_derivative

        def mock_run(definition, derivative_name, *args, **kwargs):
            if derivative_name == "Child":
                # Simulate child early-returning its cached dict
                return {"cached": [fake_cached_path]}
            return original_run(definition, derivative_name, *args, **kwargs)

        # _cached_unit returns None so cached_here=False → we recurse into Child
        with (
            patch("neurodags.dag.run_derivative", side_effect=mock_run),
            patch("neurodags.dag._cached_unit", return_value=None),
        ):
            result = original_run(
                consumer_def,
                "Consumer",
                "input.vhdr",
                reference_base=ref,
            )

        assert isinstance(result, NodeResult)
        # param1 should be the resolved path string, not the raw dict
        assert fake_cached_path in result.artifacts[".message.txt"].item
    finally:
        unregister_derivative("Child")
