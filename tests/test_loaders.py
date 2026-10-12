# tests/test_loaders.py
from __future__ import annotations

import io
import os
from pathlib import Path

import pytest
import yaml

# Import the function under test
from neurodags.loaders import load_configuration


def test_load_from_mapping_returns_deepcopy():
    original = {"a": 1, "nested": {"x": 10}}
    out = load_configuration(original)
    assert out == original
    assert out is not original
    assert out["nested"] is not original["nested"]
    # Mutating the result should not affect the original
    out["nested"]["x"] = 99
    assert original["nested"]["x"] == 10


def test_load_from_path_valid_yaml(tmp_path: Path):
    p = tmp_path / "rules.yml"
    p.write_text("project: cocosprint\nnum: 3\n", encoding="utf-8")
    data = load_configuration(p)
    assert data == {"project": "cocosprint", "num": 3}


def test_load_from_pathlike_str(tmp_path: Path):
    p = tmp_path / "rules.yml"
    p.write_text("a: 1\n", encoding="utf-8")
    data = load_configuration(os.fspath(p))
    assert data == {"a": 1}


def test_empty_file_returns_empty_dict(tmp_path: Path):
    p = tmp_path / "empty.yml"
    p.write_text("", encoding="utf-8")
    data = load_configuration(p)
    assert data == {}


def test_file_like_object():
    buf = io.StringIO("key: value\n")
    data = load_configuration(buf)
    assert data == {"key": "value"}


def test_invalid_yaml_raises_valueerror(tmp_path: Path):
    p = tmp_path / "bad.yml"
    # Missing closing bracket
    p.write_text("arr: [1, 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid YAML") as excinfo:
        load_configuration(p)
    assert "Invalid YAML" in str(excinfo.value)


def test_non_mapping_root_raises_typeerror_list(tmp_path: Path):
    p = tmp_path / "list_root.yml"
    p.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(TypeError) as excinfo:
        load_configuration(p)
    assert "YAML root must be a mapping/dict" in str(excinfo.value)


def test_non_mapping_root_raises_typeerror_scalar(tmp_path: Path):
    p = tmp_path / "scalar_root.yml"
    p.write_text("justastring", encoding="utf-8")
    with pytest.raises(TypeError):
        load_configuration(p)


def test_duplicate_keys_raise_valueerror(tmp_path: Path):
    p = tmp_path / "dup.yml"
    p.write_text("a: 1\na: 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate key") as excinfo:
        load_configuration(p)
    # Error message should mention duplicate key
    assert "Duplicate key 'a'" in str(excinfo.value)


def test_duplicate_keys_nested_raise_valueerror(tmp_path: Path):
    p = tmp_path / "dup_nested.yml"
    p.write_text(
        "outer:\n" "  x: 1\n" "  x: 2\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Duplicate key") as excinfo:
        load_configuration(p)
    assert "Duplicate key 'x'" in str(excinfo.value)


def test_ioerror_when_file_missing(tmp_path: Path):
    p = tmp_path / "does_not_exist.yml"
    with pytest.raises(IOError, match="Couldn't read rules file"):
        load_configuration(p)


# --- YAML merge keys (#44) ---


def _load_text(text: str):
    return load_configuration(io.StringIO(text))


def test_merge_key_with_extra_key():
    doc = "base: &base {x: 1, y: 2}\nm: {<<: *base, z: 3}\n"
    data = _load_text(doc)
    assert data["m"] == {"x": 1, "y": 2, "z": 3}
    assert data == yaml.safe_load(doc)


def test_merge_key_explicit_value_overrides_merged():
    # Overriding a merged key is legal and the explicit value wins, whatever
    # the textual position of the merge key.
    for doc in (
        "base: &base {x: 1, y: 2}\nm:\n  <<: *base\n  x: 9\n",
        "base: &base {x: 1, y: 2}\nm:\n  x: 9\n  <<: *base\n",
    ):
        data = _load_text(doc)
        assert data["m"] == {"x": 9, "y": 2}
        assert data == yaml.safe_load(doc)


def test_merge_key_list_earlier_entries_take_precedence():
    doc = "a: &a {x: 1}\n" "b: &b {x: 2, y: 2}\n" "m:\n" "  <<: [*a, *b]\n" "  z: 3\n"
    data = _load_text(doc)
    assert data["m"] == {"x": 1, "y": 2, "z": 3}
    assert data == yaml.safe_load(doc)


def test_merge_key_chained_anchors():
    doc = "base: &base {p: 1}\nmid: &mid {<<: *base, q: 2}\nm: {<<: *mid, r: 3}\n"
    data = _load_text(doc)
    assert data["m"] == {"p": 1, "q": 2, "r": 3}
    assert data == yaml.safe_load(doc)


def test_merge_key_in_node_args_issue_example():
    doc = (
        "_defaults: &fitargs\n"
        "  n_iter: 100\n"
        "  alphas: [1.0, 1000.0, 20]\n"
        "DerivativeDefinitions:\n"
        "  FitA:\n"
        "    nodes:\n"
        "      - {id: 1, node: my_fit, args: {<<: *fitargs, space: a}}\n"
        "  FitB:\n"
        "    nodes:\n"
        "      - id: 1\n"
        "        node: my_fit\n"
        "        args:\n"
        "          <<: *fitargs\n"
        "          n_iter: 5\n"
        "          space: b\n"
    )
    data = _load_text(doc)
    defs = data["DerivativeDefinitions"]
    assert defs["FitA"]["nodes"][0]["args"] == {
        "n_iter": 100,
        "alphas": [1.0, 1000.0, 20],
        "space": "a",
    }
    assert defs["FitB"]["nodes"][0]["args"] == {
        "n_iter": 5,
        "alphas": [1.0, 1000.0, 20],
        "space": "b",
    }
    assert data == yaml.safe_load(doc)


def test_explicit_duplicate_next_to_merge_key_still_raises():
    doc = "base: &base {x: 1}\nm:\n  <<: *base\n  y: 1\n  y: 2\n"
    with pytest.raises(ValueError, match="Duplicate key 'y' at line 5, column 3"):
        _load_text(doc)


def test_duplicate_inside_anchored_mapping_still_raises():
    doc = "base: &base\n  x: 1\n  x: 2\nm: {<<: *base, y: 3}\n"
    with pytest.raises(ValueError, match="Duplicate key 'x' at line 3, column 3"):
        _load_text(doc)
