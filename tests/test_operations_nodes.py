"""Tests for xarray operation nodes."""
from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from neurodags.definitions import Artifact, NodeResult
from neurodags.nodes.operations import (
    aggregate_across_dimension,
    binarize_with_median,
    extract_data_var,
    mean_across_dimension,
    slice_xarray,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_da():
    return xr.DataArray(
        np.arange(24, dtype=float).reshape(4, 3, 2),
        dims=("times", "channel", "frequency"),
        coords={
            "times": np.arange(4, dtype=float),
            "channel": ["Cz", "Pz", "Fz"],
            "frequency": [10.0, 20.0],
        },
    )


@pytest.fixture
def simple_ds(simple_da):
    return xr.Dataset({"power": simple_da})


# ---------------------------------------------------------------------------
# binarize_with_median
# ---------------------------------------------------------------------------

def test_binarize_returns_noderesulet(simple_da):
    result = binarize_with_median(simple_da, dim="times")
    assert isinstance(result, NodeResult)
    assert ".nc" in result.artifacts


def test_binarize_values_binary(simple_da):
    result = binarize_with_median(simple_da, dim="times")
    arr = result.artifacts[".nc"].item
    unique = set(arr.values.flatten().tolist())
    assert unique <= {0, 1}


def test_binarize_from_path(simple_da, tmp_path):
    nc_path = tmp_path / "data.nc"
    simple_da.to_netcdf(nc_path)
    result = binarize_with_median(nc_path, dim="times")
    assert isinstance(result, NodeResult)


def test_binarize_invalid_type_raises():
    with pytest.raises(ValueError, match="xarray DataArray"):
        binarize_with_median(42, dim="times")


def test_binarize_invalid_path_raises(tmp_path):
    with pytest.raises(ValueError, match="Failed to load"):
        binarize_with_median(tmp_path / "nonexistent.nc", dim="times")


# ---------------------------------------------------------------------------
# mean_across_dimension
# ---------------------------------------------------------------------------

def test_mean_returns_noderesulet(simple_da):
    result = mean_across_dimension(simple_da, dim="times")
    assert isinstance(result, NodeResult)
    assert ".nc" in result.artifacts


def test_mean_reduces_dimension(simple_da):
    result = mean_across_dimension(simple_da, dim="times")
    arr = result.artifacts[".nc"].item
    assert "times" not in arr.dims


def test_mean_from_path(simple_da, tmp_path):
    nc_path = tmp_path / "data.nc"
    simple_da.to_netcdf(nc_path)
    result = mean_across_dimension(nc_path, dim="times")
    assert isinstance(result, NodeResult)


def test_mean_invalid_type_raises():
    with pytest.raises(ValueError, match="xarray DataArray"):
        mean_across_dimension({"not": "xarray"}, dim="times")


# ---------------------------------------------------------------------------
# extract_data_var
# ---------------------------------------------------------------------------

def test_extract_from_dataset(simple_ds):
    result = extract_data_var(simple_ds, data_var="power")
    assert isinstance(result, NodeResult)
    arr = result.artifacts[".nc"].item
    assert isinstance(arr, xr.DataArray)


def test_extract_from_dataarray(simple_da):
    da = simple_da.copy()
    da.name = "power"
    result = extract_data_var(da, data_var="power")
    assert isinstance(result, NodeResult)


def test_extract_from_dataarray_no_name(simple_da):
    da = simple_da.copy()
    da.name = None
    result = extract_data_var(da, data_var="anything")
    arr = result.artifacts[".nc"].item
    assert arr.name == "anything"


def test_extract_from_noderesulet(simple_ds):
    nr = NodeResult(artifacts={".nc": Artifact(item=simple_ds, writer=lambda p: None)})
    result = extract_data_var(nr, data_var="power")
    assert isinstance(result, NodeResult)


def test_extract_noderesulet_missing_nc_raises(simple_da):
    nr = NodeResult(artifacts={".fif": Artifact(item=simple_da, writer=lambda p: None)})
    with pytest.raises(ValueError, match=".nc"):
        extract_data_var(nr, data_var="power")


def test_extract_missing_var_raises(simple_ds):
    with pytest.raises(KeyError, match="nonexistent"):
        extract_data_var(simple_ds, data_var="nonexistent")


def test_extract_dataarray_wrong_name_raises(simple_da):
    da = simple_da.copy()
    da.name = "other"
    with pytest.raises(ValueError):
        extract_data_var(da, data_var="power")


def test_extract_from_path(simple_ds, tmp_path):
    nc_path = tmp_path / "ds.nc"
    simple_ds.to_netcdf(nc_path)
    result = extract_data_var(nc_path, data_var="power")
    assert isinstance(result, NodeResult)


def test_extract_from_path_missing_var_raises(simple_ds, tmp_path):
    nc_path = tmp_path / "ds.nc"
    simple_ds.to_netcdf(nc_path)
    with pytest.raises(KeyError, match="nonexistent"):
        extract_data_var(nc_path, data_var="nonexistent")


def test_extract_invalid_type_raises():
    with pytest.raises(ValueError, match="must be a NodeResult"):
        extract_data_var(42, data_var="power")


# ---------------------------------------------------------------------------
# slice_xarray
# ---------------------------------------------------------------------------

def test_slice_by_index(simple_da):
    result = slice_xarray(simple_da, dim="times", start=1, end=3)
    arr = result.artifacts[".nc"].item
    assert arr.sizes["times"] == 2


def test_slice_by_coord(simple_da):
    result = slice_xarray(simple_da, dim="times", start=1.0, end=2.0)
    arr = result.artifacts[".nc"].item
    assert "times" in arr.dims or arr.ndim < simple_da.ndim


def test_slice_full_range(simple_da):
    result = slice_xarray(simple_da, dim="times")
    arr = result.artifacts[".nc"].item
    assert arr.sizes["times"] == 4


def test_slice_from_path(simple_da, tmp_path):
    nc_path = tmp_path / "data.nc"
    simple_da.to_netcdf(nc_path)
    result = slice_xarray(nc_path, dim="times", start=0, end=2)
    assert isinstance(result, NodeResult)


def test_slice_from_noderesulet(simple_da):
    nr = NodeResult(artifacts={".nc": Artifact(item=simple_da, writer=lambda p: None)})
    result = slice_xarray(nr, dim="times", start=0, end=2)
    assert isinstance(result, NodeResult)


def test_slice_noderesulet_missing_nc_raises(simple_da):
    nr = NodeResult(artifacts={".fif": Artifact(item=simple_da, writer=lambda p: None)})
    with pytest.raises(ValueError, match=".nc"):
        slice_xarray(nr, dim="times", start=0, end=2)


def test_slice_invalid_dim_raises(simple_da):
    with pytest.raises(ValueError, match="Dimension"):
        slice_xarray(simple_da, dim="nonexistent", start=0, end=2)


def test_slice_invalid_type_raises():
    with pytest.raises(ValueError, match="xarray DataArray"):
        slice_xarray(42, dim="times")


def test_slice_single_index_squeezes(simple_da):
    result = slice_xarray(simple_da, dim="times", start=1, end=2)
    arr = result.artifacts[".nc"].item
    assert "times" not in arr.dims


# ---------------------------------------------------------------------------
# aggregate_across_dimension
# ---------------------------------------------------------------------------

def test_aggregate_mean(simple_da):
    result = aggregate_across_dimension(simple_da, dim="times", operation="mean")
    arr = result.artifacts[".nc"].item
    assert "times" not in arr.dims


def test_aggregate_sum(simple_da):
    result = aggregate_across_dimension(simple_da, dim="times", operation="sum")
    arr = result.artifacts[".nc"].item
    assert float(arr.values.sum()) > 0


def test_aggregate_max(simple_da):
    result = aggregate_across_dimension(simple_da, dim="times", operation="max")
    arr = result.artifacts[".nc"].item
    assert "times" not in arr.dims


def test_aggregate_from_path(simple_da, tmp_path):
    nc_path = tmp_path / "data.nc"
    simple_da.to_netcdf(nc_path)
    result = aggregate_across_dimension(nc_path, dim="times", operation="mean")
    assert isinstance(result, NodeResult)


def test_aggregate_from_noderesulet(simple_da):
    nr = NodeResult(artifacts={".nc": Artifact(item=simple_da, writer=lambda p: None)})
    result = aggregate_across_dimension(nr, dim="times", operation="mean")
    assert isinstance(result, NodeResult)


def test_aggregate_noderesulet_missing_nc_raises(simple_da):
    nr = NodeResult(artifacts={".fif": Artifact(item=simple_da, writer=lambda p: None)})
    with pytest.raises(ValueError, match=".nc"):
        aggregate_across_dimension(nr, dim="times", operation="mean")


def test_aggregate_invalid_operation_raises(simple_da):
    with pytest.raises(ValueError, match="not valid"):
        aggregate_across_dimension(simple_da, dim="times", operation="nonexistent_op")


def test_aggregate_invalid_type_raises():
    with pytest.raises(ValueError, match="xarray DataArray"):
        aggregate_across_dimension(42, dim="times", operation="mean")


def test_aggregate_with_args(simple_da):
    result = aggregate_across_dimension(simple_da, dim="times", operation="mean", args={"keepdims": False})
    arr = result.artifacts[".nc"].item
    assert isinstance(arr, xr.DataArray)
