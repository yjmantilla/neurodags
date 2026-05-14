"""Tests for nodes/loader.py — load_node_definitions and helpers."""
from __future__ import annotations

from pathlib import Path

import pytest

import neurodags.nodes.loader as loader_mod
from neurodags.nodes.loader import _unique_module_name, load_node_definitions


# ---------------------------------------------------------------------------
# _unique_module_name
# ---------------------------------------------------------------------------

def test_unique_module_name_is_deterministic(tmp_path):
    p = tmp_path / "nodes.py"
    assert _unique_module_name(p) == _unique_module_name(p)


def test_unique_module_name_differs_for_different_paths(tmp_path):
    a = tmp_path / "a.py"
    b = tmp_path / "b.py"
    assert _unique_module_name(a) != _unique_module_name(b)


def test_unique_module_name_has_prefix(tmp_path):
    p = tmp_path / "nodes.py"
    assert _unique_module_name(p).startswith("_neurodags_nodes_")


# ---------------------------------------------------------------------------
# load_node_definitions
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clear_loaded_sources():
    """Reset the module-level cache before each test."""
    original = loader_mod._LOADED_SOURCES.copy()
    loader_mod._LOADED_SOURCES.clear()
    yield
    loader_mod._LOADED_SOURCES.clear()
    loader_mod._LOADED_SOURCES.update(original)


def test_load_empty_iterable():
    result = load_node_definitions([])
    assert result == []


def test_load_valid_python_file(tmp_path):
    node_file = tmp_path / "custom_nodes.py"
    node_file.write_text("MY_VAR = 42\n")

    modules = load_node_definitions([node_file])
    assert len(modules) == 1
    assert modules[0].MY_VAR == 42


def test_load_string_path(tmp_path):
    node_file = tmp_path / "custom_nodes.py"
    node_file.write_text("MY_VAR = 99\n")

    modules = load_node_definitions([str(node_file)])
    assert len(modules) == 1
    assert modules[0].MY_VAR == 99


def test_load_relative_path(tmp_path):
    node_file = tmp_path / "custom_nodes.py"
    node_file.write_text("MY_VAR = 7\n")

    modules = load_node_definitions(["custom_nodes.py"], base_dir=tmp_path)
    assert len(modules) == 1
    assert modules[0].MY_VAR == 7


def test_load_skips_already_loaded(tmp_path):
    node_file = tmp_path / "custom_nodes.py"
    node_file.write_text("MY_VAR = 1\n")

    first = load_node_definitions([node_file])
    second = load_node_definitions([node_file])

    assert len(first) == 1
    assert len(second) == 0


def test_load_nonexistent_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="not found"):
        load_node_definitions([tmp_path / "does_not_exist.py"])


def test_load_skips_empty_string():
    result = load_node_definitions(["", "   "])
    assert result == []


def test_load_multiple_files(tmp_path):
    f1 = tmp_path / "n1.py"
    f2 = tmp_path / "n2.py"
    f1.write_text("A = 1\n")
    f2.write_text("B = 2\n")

    modules = load_node_definitions([f1, f2])
    assert len(modules) == 2
    assert modules[0].A == 1
    assert modules[1].B == 2
