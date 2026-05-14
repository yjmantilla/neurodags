"""Tests for descriptive nodes using real MNE objects."""
from __future__ import annotations

import datetime
import json

import mne
import numpy as np
import pytest
import xarray as xr

from neurodags.definitions import Artifact, NodeResult
from neurodags.nodes.descriptive import (
    _build_metadata,
    _format_meas_date,
    extract_meeg_metadata,
    meeg_to_xarray,
)


# ---------------------------------------------------------------------------
# _format_meas_date
# ---------------------------------------------------------------------------

def test_format_meas_date_none():
    assert _format_meas_date(None) is None


def test_format_meas_date_datetime():
    dt = datetime.datetime(2024, 1, 15, 10, 30, 0, tzinfo=datetime.timezone.utc)
    result = _format_meas_date(dt)
    assert "2024" in result


def test_format_meas_date_tuple():
    result = _format_meas_date((1700000000, 500000))
    assert result is not None
    assert float(result) > 0


def test_format_meas_date_string_fallback():
    result = _format_meas_date("some-date-string")
    assert result == "some-date-string"


# ---------------------------------------------------------------------------
# _build_metadata
# ---------------------------------------------------------------------------

def test_build_metadata_raw(dummy_raw_obj):
    raw, _ = dummy_raw_obj
    meta = _build_metadata(raw, kind="raw")
    assert meta["kind"] == "raw"
    assert "sfreq" in meta
    assert meta["n_channels"] == raw.info["nchan"]
    assert len(meta["channel_names"]) == raw.info["nchan"]


def test_build_metadata_epochs(dummy_epochs_obj):
    epochs = dummy_epochs_obj
    meta = _build_metadata(epochs, kind="epochs")
    assert meta["kind"] == "epochs"
    assert "sfreq" in meta


def test_build_metadata_extra(dummy_raw_obj):
    raw, _ = dummy_raw_obj
    meta = _build_metadata(raw, kind="raw", extra={"custom_key": "custom_value"})
    assert meta["custom_key"] == "custom_value"


# ---------------------------------------------------------------------------
# extract_meeg_metadata
# ---------------------------------------------------------------------------

def test_extract_meeg_metadata_raw_returns_noderesulet(dummy_raw_obj):
    raw, _ = dummy_raw_obj
    result = extract_meeg_metadata(raw.copy())
    assert isinstance(result, NodeResult)
    assert ".json" in result.artifacts


def test_extract_meeg_metadata_raw_content(dummy_raw_obj):
    raw, _ = dummy_raw_obj
    result = extract_meeg_metadata(raw.copy())
    info_dict = result.artifacts[".json"].item
    assert "sfreq" in info_dict
    assert "ch_names" in info_dict
    assert info_dict["n_channels"] == raw.info["nchan"]


def test_extract_meeg_metadata_epochs(dummy_epochs_obj):
    epochs = dummy_epochs_obj
    result = extract_meeg_metadata(epochs)
    assert isinstance(result, NodeResult)
    info_dict = result.artifacts[".json"].item
    assert "dims" in info_dict
    assert "epochs" in info_dict["dims"]


def test_extract_meeg_metadata_accepts_noderesulet(dummy_raw_obj):
    raw, _ = dummy_raw_obj
    nr = NodeResult(artifacts={".fif": Artifact(item=raw.copy(), writer=lambda p: None)})
    result = extract_meeg_metadata(nr)
    assert isinstance(result, NodeResult)
    assert ".json" in result.artifacts


def test_extract_meeg_metadata_noderesulet_missing_fif_raises():
    with pytest.raises(ValueError, match=".fif"):
        extract_meeg_metadata(NodeResult(artifacts={".nc": Artifact(item=None, writer=lambda p: None)}))


def test_extract_meeg_metadata_from_path(dummy_vhdr_file):
    result = extract_meeg_metadata(dummy_vhdr_file)
    assert isinstance(result, NodeResult)
    assert ".json" in result.artifacts


def test_extract_meeg_metadata_writer_produces_valid_json(dummy_raw_obj, tmp_path):
    raw, _ = dummy_raw_obj
    result = extract_meeg_metadata(raw.copy())
    out_path = tmp_path / "meta.json"
    result.artifacts[".json"].writer(out_path)
    with open(out_path) as f:
        loaded = json.load(f)
    assert "sfreq" in loaded


# ---------------------------------------------------------------------------
# meeg_to_xarray
# ---------------------------------------------------------------------------

def test_meeg_to_xarray_raw_returns_noderesulet(dummy_raw_obj):
    raw, _ = dummy_raw_obj
    result = meeg_to_xarray(raw.copy())
    assert isinstance(result, NodeResult)
    assert ".nc" in result.artifacts


def test_meeg_to_xarray_raw_dims(dummy_raw_obj):
    raw, _ = dummy_raw_obj
    result = meeg_to_xarray(raw.copy())
    da = result.artifacts[".nc"].item
    assert isinstance(da, xr.DataArray)
    assert "spaces" in da.dims
    assert "times" in da.dims
    assert da.sizes["spaces"] == raw.info["nchan"]


def test_meeg_to_xarray_raw_attrs(dummy_raw_obj):
    raw, _ = dummy_raw_obj
    result = meeg_to_xarray(raw.copy())
    da = result.artifacts[".nc"].item
    assert da.attrs.get("kind") == "raw"
    assert "sfreq" in da.attrs


def test_meeg_to_xarray_epochs_dims(dummy_epochs_obj):
    epochs = dummy_epochs_obj
    result = meeg_to_xarray(epochs)
    da = result.artifacts[".nc"].item
    assert isinstance(da, xr.DataArray)
    assert "epochs" in da.dims
    assert "spaces" in da.dims
    assert "times" in da.dims
    assert da.sizes["epochs"] == len(epochs)


def test_meeg_to_xarray_epochs_attrs(dummy_epochs_obj):
    epochs = dummy_epochs_obj
    result = meeg_to_xarray(epochs)
    da = result.artifacts[".nc"].item
    assert da.attrs.get("kind") == "epochs"


def test_meeg_to_xarray_accepts_noderesulet(dummy_raw_obj):
    raw, _ = dummy_raw_obj
    nr = NodeResult(artifacts={".fif": Artifact(item=raw.copy(), writer=lambda p: None)})
    result = meeg_to_xarray(nr)
    assert isinstance(result, NodeResult)
    assert ".nc" in result.artifacts


def test_meeg_to_xarray_noderesulet_missing_fif_raises():
    with pytest.raises(ValueError, match=".fif"):
        meeg_to_xarray(NodeResult(artifacts={".nc": Artifact(item=None, writer=lambda p: None)}))


def test_meeg_to_xarray_from_path(dummy_vhdr_file):
    result = meeg_to_xarray(dummy_vhdr_file)
    assert isinstance(result, NodeResult)
    da = result.artifacts[".nc"].item
    assert isinstance(da, xr.DataArray)


def test_meeg_to_xarray_invalid_type_raises():
    with pytest.raises(TypeError):
        meeg_to_xarray(42)


def test_meeg_to_xarray_writer_produces_netcdf(dummy_raw_obj, tmp_path):
    raw, _ = dummy_raw_obj
    result = meeg_to_xarray(raw.copy())
    out_path = tmp_path / "out.nc"
    result.artifacts[".nc"].writer(out_path)
    loaded = xr.open_dataarray(out_path)
    assert "spaces" in loaded.dims
