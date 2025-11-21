import json
import os

import mne

from cocofeats.definitions import Artifact, NodeResult, PathLike
from cocofeats.loggers import get_logger
from cocofeats.writers import save_dict_to_json

from cocofeats.nodes import register_node
import xarray as xr
import numpy as np


log = get_logger(__name__)

@register_node(name="binarize_with_median", override=True)
def binarize_with_median(data: xr.DataArray, dim: str) -> xr.DataArray:
    """
    Binarize an xarray DataArray along a specified dimension using the median value.

    Parameters
    ----------
    data : xarray.DataArray
        The input xarray DataArray.
    dim : str
        The dimension name to binarize over.

    Returns
    -------
    xr.DataArray
        A binarized xarray DataArray where values above the median are 1 and others are 0.
    """

    if isinstance(data, (str, os.PathLike)):
        try:
            data = xr.open_dataarray(data)
            log.debug("Loaded xarray DataArray from file", input=data)
        except Exception as e:
            log.error("Failed to load xarray DataArray from file", input=data, error=e)
            raise ValueError("Failed to load xarray DataArray from file")

    if not isinstance(data, xr.DataArray):
        raise ValueError("Input must be an xarray DataArray.")

    median_values = data.median(dim=dim, keep_attrs=True)
    binarized_data = (data > median_values).astype(int)

    return NodeResult(artifacts={".nc": Artifact(item=binarized_data, writer=lambda path: binarized_data.to_netcdf(path))})

@register_node(name="mean_across_dimension", override=True)
def mean_across_dimension(xarray_data, dim):
    """
    Compute the mean across a specified dimension of an xarray DataArray.

    Parameters
    ----------
    xarray_data : xarray.DataArray
        The input xarray DataArray.
    dim : str
        The dimension name to compute the mean over.

    Returns
    -------
    NodeResult
        A feature result containing the mean as a netcdf4 artifact.
    """
    import xarray as xr
    import numpy as np

    if isinstance(xarray_data, (str, os.PathLike)):
        xarray_data = xr.open_dataarray(xarray_data)
        log.debug("Loaded xarray DataArray from file", input=xarray_data)


    if not isinstance(xarray_data, xr.DataArray):
        raise ValueError("Input must be an xarray DataArray.")

    mean_data = xarray_data.mean(dim=dim)

    # return the new xarray in ncdf4 format
    artifacts = {".nc": Artifact(item=mean_data, writer=lambda path: mean_data.to_netcdf(path))}
    return NodeResult(artifacts=artifacts)


@register_node(name="extract_data_var", override=True)
def extract_data_var(dataset_like, data_var: str):
    """Extract a named variable from an ``xarray.Dataset`` artifact.

    Parameters
    ----------
    dataset_like : NodeResult | xarray.Dataset | str | os.PathLike
        Source dataset or reference. Strings are interpreted as paths to
        NetCDF files containing an ``xarray.Dataset``.
    data_var : str
        Name of the dataset variable to extract.

    Returns
    -------
    NodeResult
        A feature result containing the selected variable as a NetCDF artifact.
    """

    if isinstance(dataset_like, NodeResult):
        if ".nc" not in dataset_like.artifacts:
            raise ValueError("NodeResult does not contain a .nc artifact to process.")
        dataset_like = dataset_like.artifacts[".nc"].item

    target_array = None

    if isinstance(dataset_like, xr.Dataset):
        if data_var not in dataset_like.data_vars:
            raise KeyError(f"Variable '{data_var}' not found in dataset.")
        target_array = dataset_like[data_var].copy()
    elif isinstance(dataset_like, xr.DataArray):
        if dataset_like.name not in {data_var, None}:
            raise ValueError(
                "Input DataArray does not match requested variable name and cannot be extracted."
            )
        target_array = dataset_like.copy()
        if target_array.name is None:
            target_array.name = data_var
    elif isinstance(dataset_like, (str, os.PathLike)):
        ds = xr.open_dataset(dataset_like)
        try:
            if data_var not in ds.data_vars:
                raise KeyError(f"Variable '{data_var}' not found in dataset.")
            target_array = ds[data_var].load()
        finally:
            ds.close()
    else:
        raise ValueError("dataset_like must be a NodeResult, Dataset, DataArray, or path.")

    artifacts = {".nc": Artifact(item=target_array, writer=lambda path: target_array.to_netcdf(path))}
    return NodeResult(artifacts=artifacts)


@register_node(name="slice_xarray", override=True)
def slice_xarray(xarray_data, dim, start=None, end=None):
    """
    Slice an xarray DataArray along a specified dimension.

    Parameters
    ----------
    xarray_data : xarray.DataArray
        The input xarray DataArray.
    dim : str
        The dimension name to slice.
    start : int or hashable or None
        The starting index/label for the slice. Indices use zero-based positions (inclusive)
        while labels use coordinate values (inclusive). If None, starts from the beginning.
    end : int or hashable or None
        The ending index/label for the slice. Indices use zero-based positions (exclusive)
        while labels use coordinate values (inclusive). If None, goes to the end.

    Returns
    -------
    NodeResult
        A feature result containing the sliced data as a netcdf4 artifact.
    """

    if isinstance(xarray_data, (str, os.PathLike)):
        xarray_data = xr.open_dataarray(xarray_data)
        log.debug("Loaded xarray DataArray from file", input=xarray_data)

    if isinstance(xarray_data, NodeResult):
        if ".nc" in xarray_data.artifacts:
            xarray_data = xarray_data.artifacts[".nc"].item
        else:
            raise ValueError("NodeResult does not contain a .nc artifact to process.")

    if not isinstance(xarray_data, xr.DataArray):
        raise ValueError("Input must be an xarray DataArray.")

    if dim not in xarray_data.dims:
        raise ValueError(f"Dimension '{dim}' not found in the DataArray.")

    def _is_index_like(value):
        return value is None or isinstance(value, (int, np.integer))

    slicer = {dim: slice(start, end)}

    if _is_index_like(start) and _is_index_like(end):
        sliced_data = xarray_data.isel(**slicer)
    else:
        if dim not in xarray_data.coords:
            raise ValueError(
                f"Dimension '{dim}' cannot be sliced by coordinate labels because it lacks coordinates."
            )
        try:
            sliced_data = xarray_data.sel(**slicer)
        except KeyError as exc:
            raise ValueError(
                f"Coordinate value(s) {start!r}, {end!r} not found along dimension '{dim}'."
            ) from exc

    if start == end or (dim in sliced_data.sizes and sliced_data.sizes[dim] == 1):
        sliced_data = sliced_data.squeeze(dim=dim, drop=True)

    # return the new xarray in ncdf4 format
    artifacts = {".nc": Artifact(item=sliced_data, writer=lambda path: sliced_data.to_netcdf(path))}
    return NodeResult(artifacts=artifacts)

@register_node(name="aggregate_across_dimension", override=True)
def aggregate_across_dimension(xarray_data, dim, operation='mean', args=None):
    """
    Aggregate data across a specified dimension of an xarray DataArray using a given operation.

    Parameters
    ----------
    xarray_data : xarray.DataArray
        The input xarray DataArray.
    dim : str
        The dimension name to aggregate over.
    operation : str
        The aggregation operation to perform ('mean', 'sum', 'max', 'min', etc.).
    args : dict, optional
        Additional arguments to pass to the aggregation function.

    Returns
    -------
    NodeResult
        A feature result containing the aggregated data as a netcdf4 artifact.
    """

    if isinstance(xarray_data, (str, os.PathLike)):
        xarray_data = xr.open_dataarray(xarray_data)
        log.debug("Loaded xarray DataArray from file", input=xarray_data)

    if isinstance(xarray_data, NodeResult):
        if ".nc" in xarray_data.artifacts:
            xarray_data = xarray_data.artifacts[".nc"].item
        else:
            raise ValueError("NodeResult does not contain a .nc artifact to process.")

    if not isinstance(xarray_data, xr.DataArray):
        raise ValueError("Input must be an xarray DataArray.")

    if args is None:
        args = {}

    if not hasattr(xarray_data, operation):
        raise ValueError(f"Operation '{operation}' is not valid for xarray DataArray.")

    agg_func = getattr(xarray_data, operation)
    aggregated_data = agg_func(dim=dim, **args)

    # return the new xarray in ncdf4 format
    artifacts = {".nc": Artifact(item=aggregated_data, writer=lambda path: aggregated_data.to_netcdf(path))}
    return NodeResult(artifacts=artifacts)

if __name__ == "__main__":
    import numpy as np
    # Example usage
    # Example usage
    data = xr.DataArray(np.random.rand(4, 3, 2), dims=("times", "channel", "frequency"), coords={
        "times": np.arange(4),
        "channel": ["Cz", "Pz", "Fz"],
        "frequency": [10, 20]
    })
    result = mean_across_dimension(data, dim="times")
    print(result)

    # test aggregate
    result_agg = aggregate_across_dimension(data, dim="channel", operation='sum')
    print(result_agg)

    # test slice
    result_slice = slice_xarray(data, dim="times", start=1, end=3)
    print(result_slice)
    print(result_slice.artifacts[".nc"].item)  # Access the sliced xarray
