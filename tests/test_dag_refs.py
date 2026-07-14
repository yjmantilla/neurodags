"""Tests for id.N / id.N.field reference resolution in dag.py (parser nodes, #8)."""

from __future__ import annotations

import pytest

from neurodags.dag import _collect_id_refs, _is_id_ref, _prep_kwargs, _resolve_refs
from neurodags.definitions import Artifact, NodeResult


def _parser_result(props: dict) -> NodeResult:
    """A parser-style NodeResult: one artifact whose item is a flat dict."""
    return NodeResult(artifacts={".json": Artifact(item=props, writer=lambda p: None)})


# ---------------------------------------------------------------------------
# Whole-result reference (id.N) — unchanged behaviour
# ---------------------------------------------------------------------------


def test_whole_ref_returns_stored_value():
    store = {0: "/data/x.fif", 1: _parser_result({"subject": "001"})}
    assert _resolve_refs("id.0", store) == "/data/x.fif"
    assert _resolve_refs("id.1", store) is store[1]


def test_is_id_ref_matches_both_forms():
    assert _is_id_ref("id.0")
    assert _is_id_ref("id.12.subject")
    assert not _is_id_ref("welch")
    assert not _is_id_ref("id.1.a.b")


def test_missing_node_raises():
    with pytest.raises(KeyError, match="not computed yet"):
        _resolve_refs("id.5", {})
    with pytest.raises(KeyError, match="not computed yet"):
        _resolve_refs("id.5.subject", {})


# ---------------------------------------------------------------------------
# Field access (id.N.field)
# ---------------------------------------------------------------------------


def test_field_access_on_noderesult_dict():
    store = {1: _parser_result({"subject": "001", "task": "rest"})}
    assert _resolve_refs("id.1.subject", store) == "001"
    assert _resolve_refs("id.1.task", store) == "rest"


def test_field_access_on_plain_dict():
    store = {1: {"run": "02"}}
    assert _resolve_refs("id.1.run", store) == "02"


def test_missing_field_raises_with_available():
    store = {1: _parser_result({"subject": "001"})}
    with pytest.raises(KeyError, match=r"not found in id\.1"):
        _resolve_refs("id.1.session", store)


def test_field_on_non_mapping_raises():
    store = {0: "/data/x.fif"}
    with pytest.raises(ValueError, match="requires node 0 to return a mapping"):
        _resolve_refs("id.0.subject", store)


# ---------------------------------------------------------------------------
# The field-key constraint: no nested / dotted / dashed paths
# ---------------------------------------------------------------------------


def test_nested_access_rejected():
    store = {1: _parser_result({"a": {"b": 1}})}
    with pytest.raises(ValueError, match="Unsupported reference"):
        _resolve_refs("id.1.a.b", store)


def test_dashed_field_rejected():
    store = {1: _parser_result({"x": 1})}
    with pytest.raises(ValueError, match="Unsupported reference"):
        _resolve_refs("id.1.some-key", store)


def test_trailing_dot_rejected():
    store = {1: _parser_result({"x": 1})}
    with pytest.raises(ValueError, match="Unsupported reference"):
        _resolve_refs("id.1.", store)


# ---------------------------------------------------------------------------
# Literals that are NOT references pass through untouched
# ---------------------------------------------------------------------------


def test_plain_string_passthrough():
    assert _resolve_refs("welch", {}) == "welch"


def test_path_with_id_substring_passthrough():
    # only a whole value starting with 'id.<n>.' is treated as a reference
    assert _resolve_refs("/data/id_stuff/x.fif", {}) == "/data/id_stuff/x.fif"


# ---------------------------------------------------------------------------
# Dependency collection: id.N.field still registers a dependency on N
# ---------------------------------------------------------------------------


def test_collect_id_refs_includes_field_refs():
    assert _collect_id_refs({"a": "id.3.subject", "b": "id.0"}) == {0, 3}
    assert _collect_id_refs(["id.2.task", "id.2"]) == {2}


# ---------------------------------------------------------------------------
# End-to-end kwarg prep: whole dict + field + path resolved together
# ---------------------------------------------------------------------------


def test_prep_kwargs_resolves_field_whole_and_path():
    store = {
        0: "/data/sub-001_task-rest_eeg.fif",
        1: _parser_result({"subject": "001", "task": "rest"}),
    }
    kwargs = _prep_kwargs({"path": "id.0", "subject": "id.1.subject", "props": "id.1"}, store)
    assert kwargs["path"] == "/data/sub-001_task-rest_eeg.fif"
    assert kwargs["subject"] == "001"
    # id.1 (whole) unwraps the single-artifact NodeResult to the dict
    assert kwargs["props"] == {"subject": "001", "task": "rest"}
