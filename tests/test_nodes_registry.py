import pytest

import neurodags.nodes as nodes
from neurodags.nodes.preprocessing import basic_preprocessing
from neurodags.nodes.spectral import mne_spectrum


def test_known_nodes_registered():
    registered = nodes.list_nodes()
    assert "basic_preprocessing" in registered
    assert "mne_spectrum" in registered
    assert nodes.get_node("basic_preprocessing") is basic_preprocessing
    assert nodes.get_node("mne_spectrum") is mne_spectrum


def test_register_node_duplicate_guard():
    @nodes.register_node(name="temporary_node")
    def temporary_node():
        return "ok"

    try:
        with pytest.raises(ValueError):

            @nodes.register_node(name="temporary_node")
            def duplicate_node():
                return "duplicate"

    finally:
        nodes.unregister_node("temporary_node")


def test_get_node_unknown_raises_key_error():
    with pytest.raises(KeyError) as excinfo:
        nodes.get_node("does_not_exist")

    assert "Unknown node" in str(excinfo.value)
    assert "does_not_exist" in str(excinfo.value)
