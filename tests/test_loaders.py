# tests/test_loaders.py
from __future__ import annotations

import io
import os
from pathlib import Path

import pytest

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
    with pytest.raises(ValueError) as excinfo:
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
    with pytest.raises(ValueError) as excinfo:
        load_configuration(p)
    # Error message should mention duplicate key
    assert "Duplicate key 'a'" in str(excinfo.value)


def test_duplicate_keys_nested_raise_valueerror(tmp_path: Path):
    p = tmp_path / "dup_nested.yml"
    p.write_text(
        "outer:\n" "  x: 1\n" "  x: 2\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError) as excinfo:
        load_configuration(p)
    assert "Duplicate key 'x'" in str(excinfo.value)


def test_ioerror_when_file_missing(tmp_path: Path):
    p = tmp_path / "does_not_exist.yml"
    with pytest.raises(IOError):
        load_configuration(p)
