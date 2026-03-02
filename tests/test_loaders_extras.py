# tests/test_loaders_extra.py
import io
import types
import pytest
import yaml

import neurodags.loaders as loaders


def test_fallback_to_safe_loader(monkeypatch):
    # Force yaml to not have CSafeLoader
    monkeypatch.delattr(yaml, "CSafeLoader", raising=False)
    # Reload the module to trigger the import-time try/except
    import importlib
    import neurodags.loaders as loaders_mod

    importlib.reload(loaders_mod)
    # Should have picked SafeLoader
    assert loaders_mod._BaseSafeLoader is yaml.SafeLoader


def test_construct_mapping_non_mapping_node_raises():
    loader = loaders.UniqueKeySafeLoader(io.StringIO("dummy: 1"))
    # Create a fake scalar node
    scalar_node = yaml.ScalarNode(tag="tag:yaml.org,2002:str", value="hello")
    with pytest.raises(yaml.constructor.ConstructorError) as e:
        loader.construct_mapping(scalar_node)
    assert "Expected a mapping node" in str(e.value)


# --- Tests for load_meeg ---


def test_load_meeg_reads_raw(monkeypatch, tmp_path):
    dummy = object()

    def fake_read_raw(path, **kwargs):
        return dummy

    monkeypatch.setattr(loaders.mne.io, "read_raw", fake_read_raw)
    # Make sure read_epochs would fail if called
    monkeypatch.setattr(
        loaders.mne,
        "read_epochs",
        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("should not call")),
    )

    f = tmp_path / "dummy.fif"
    f.write_text("placeholder")

    out = loaders.load_meeg(f)
    assert out is dummy


def test_load_meeg_fallback_to_epochs(monkeypatch, tmp_path):
    dummy = object()

    def fake_read_raw(path, **kwargs):
        raise RuntimeError("raw failed")

    def fake_read_epochs(path, **kwargs):
        return dummy

    monkeypatch.setattr(loaders.mne.io, "read_raw", fake_read_raw)
    monkeypatch.setattr(loaders.mne, "read_epochs", fake_read_epochs)

    f = tmp_path / "dummy-epo.fif"
    f.write_text("placeholder")

    out = loaders.load_meeg(f)
    assert out is dummy


def test_load_meeg_both_fail(monkeypatch, tmp_path):
    def fake_fail(*a, **kw):
        raise RuntimeError("fail")

    monkeypatch.setattr(loaders.mne.io, "read_raw", fake_fail)
    monkeypatch.setattr(loaders.mne, "read_epochs", fake_fail)

    f = tmp_path / "badfile.fif"
    f.write_text("placeholder")

    with pytest.raises(ValueError) as e:
        loaders.load_meeg(f)
    assert "Could not load MEEG file" in str(e.value)


def test_load_meeg_default_kwargs(monkeypatch, tmp_path):
    captured = {}

    def fake_read_raw(path, **kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(loaders.mne.io, "read_raw", fake_read_raw)

    f = tmp_path / "dummy.fif"
    f.write_text("placeholder")

    loaders.load_meeg(f, kwargs=None)
    assert captured["preload"] is True
    assert captured["verbose"] == "error"
