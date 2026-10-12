"""Tests for parser nodes (bids_parse) — issue #8."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from neurodags.dag import _resolve_refs, run_derivative
from neurodags.definitions import NodeResult
from neurodags.derivatives import unregister_derivative
from neurodags.derivatives.pipeline import register_derivatives_from_dict
from neurodags.nodes import get_node
from neurodags.orchestrators import run_pipeline


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


# ---------------------------------------------------------------------------
# End-to-end: parser node feeding id.N.field through run_derivative / run_pipeline
# ---------------------------------------------------------------------------

_BIDS_SOURCE = "sub-007_ses-02_task-rest_eeg.vhdr"


@pytest.fixture
def cached_parser_pipeline():
    """BidsInfo (saved parser derivative) + Consumer reading fields from it."""
    cfg = {
        "DerivativeDefinitions": {
            "BidsInfo": {
                "overwrite": False,
                "nodes": [
                    {"id": 0, "derivative": "SourceFile"},
                    {"id": 1, "node": "bids_parse", "args": {"path": "id.0"}},
                ],
            },
            "Consumer": {
                "overwrite": True,
                "nodes": [
                    {"id": 0, "derivative": "BidsInfo.json"},
                    {
                        "id": 1,
                        "node": "dummy",
                        "args": {"param1": "id.0.subject", "param2": "id.0.task"},
                    },
                ],
            },
        }
    }
    register_derivatives_from_dict(cfg)
    yield cfg
    for name in cfg["DerivativeDefinitions"]:
        unregister_derivative(name)


def _run_consumer(cfg: dict, ref: Path) -> str:
    result = run_derivative(
        cfg["DerivativeDefinitions"]["Consumer"],
        derivative_name="Consumer",
        file_path=_BIDS_SOURCE,
        reference_base=ref,
    )
    assert isinstance(result, NodeResult)
    return Path(f"{ref.as_posix()}@Consumer.message.txt").read_text()


def test_field_ref_works_when_parser_derivative_is_cached(tmp_path, cached_parser_pipeline):
    ref = tmp_path / "sub-007" / "sample"
    ref.parent.mkdir(parents=True)
    expected = "param1=007 and param2=rest"

    # First run computes BidsInfo in memory (and saves its .json).
    assert expected in _run_consumer(cached_parser_pipeline, ref)
    assert Path(f"{ref.as_posix()}@BidsInfo.json").exists()

    # Second run hits the BidsInfo cache: the store holds a file path, not a NodeResult.
    assert expected in _run_consumer(cached_parser_pipeline, ref)


def test_field_ref_on_cached_non_json_artifact_raises_clear_error(tmp_path):
    artifact = tmp_path / "x@Prep.fif"
    artifact.write_bytes(b"")
    with pytest.raises(ValueError, match=r"\.json"):
        _resolve_refs("id.0.subject", {0: artifact.as_posix()})


def test_run_pipeline_source_to_bids_parse_to_field_consumer(dummy_pipeline):
    cfg = dummy_pipeline["config"]
    out_dir = dummy_pipeline["out_dir"]
    cfg["DerivativeDefinitions"]["SubjectEcho"] = {
        "overwrite": False,
        "nodes": [
            {"id": 0, "derivative": "SourceFile"},
            {"id": 1, "node": "bids_parse", "args": {"path": "id.0"}},
            {"id": 2, "node": "dummy", "args": {"param1": "id.1.subject"}},
        ],
    }
    cfg["DerivativeList"] = ["SubjectEcho"]

    run_pipeline(cfg, raise_on_error=True)

    sources = sorted(dummy_pipeline["data_dir"].rglob("*.vhdr"))
    outputs = sorted(out_dir.rglob("*@SubjectEcho.message.txt"))
    assert len(outputs) == len(sources) > 0
    bids_parse = get_node("bids_parse")
    subjects = {bids_parse(s).artifacts[".json"].item["subject"] for s in sources}
    assert len(subjects) == len(sources)
    texts = [o.read_text() for o in outputs]
    assert {sub for sub in subjects if any(f"param1={sub} and" in t for t in texts)} == subjects
