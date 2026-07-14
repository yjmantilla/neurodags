"""Tests for parser nodes (bids_parse) — issue #8."""

from __future__ import annotations

import json
from pathlib import Path

from neurodags.definitions import NodeResult
from neurodags.nodes import get_node


def test_bids_parse_extracts_entities():
    bids_parse = get_node("bids_parse")
    res = bids_parse("sub-001_ses-01_task-rest_run-02_eeg.fif")
    assert isinstance(res, NodeResult)
    props = res.artifacts[".json"].item
    assert props["subject"] == "001"
    assert props["session"] == "01"
    assert props["task"] == "rest"
    assert props["run"] == "02"
    # every key must be a plain identifier so it is referenceable as id.N.<key>
    assert all(k.isidentifier() for k in props)


def test_bids_parse_ignores_directory():
    bids_parse = get_node("bids_parse")
    res = bids_parse("/some/deep/path/sub-007_task-foo_eeg.fif")
    assert res.artifacts[".json"].item["subject"] == "007"


def test_bids_parse_accepts_pathlike():
    bids_parse = get_node("bids_parse")
    res = bids_parse(Path("sub-42_task-rest_eeg.fif"))
    assert res.artifacts[".json"].item["subject"] == "42"


def test_bids_parse_non_bids_is_graceful():
    bids_parse = get_node("bids_parse")
    res = bids_parse("/data/recordings/P001_EO_run2.edf")
    props = res.artifacts[".json"].item
    assert isinstance(props, dict)  # no raise; entities simply absent/partial


def test_bids_parse_omits_absent_entities():
    bids_parse = get_node("bids_parse")
    props = bids_parse("sub-001_task-rest_eeg.fif").artifacts[".json"].item
    assert "session" not in props  # ses- not in the name
    assert "run" not in props


def test_bids_parse_writer_roundtrip(tmp_path):
    bids_parse = get_node("bids_parse")
    res = bids_parse("sub-001_task-rest_eeg.fif")
    out = tmp_path / "entities.json"
    res.artifacts[".json"].writer(str(out))
    loaded = json.loads(out.read_text())
    assert loaded["task"] == "rest"
