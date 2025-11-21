"""Factory utilities for building xarray-based node operations.

This module provides a flexible backbone for node implementations that apply
one-dimensional numerical routines across higher dimensional ``xarray``
containers.  The core helper accepts a pure function that consumes a 1D
``numpy`` array (optionally with extra ``args``/``kwargs``) and produces either a
scalar or a 1D sequence.  The helper takes care of:

* coercing various inputs (``NodeResult``, ``xarray`` objects, ``mne`` Raw/Epochs)
  into a consistent ``xarray.DataArray`` representation,
* iterating/vectorising the pure function along a chosen dimension,
* rebuilding an output ``DataArray`` with well-defined coordinates, and
* returning a ``NodeResult`` suitable for the node registry.

The exported ``xarray_factory`` node can be used directly from feature pipeline
definitions, and ``apply_1d`` is available for composing bespoke nodes in
Python code.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from importlib import import_module
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence
import time

try:  # pragma: no cover - optional dependency
    import mne
except ImportError:  # pragma: no cover - optional dependency
    mne = None  # type: ignore[assignment]

import numpy as np
import xarray as xr
import time

from cocofeats.definitions import Artifact, NodeResult
from cocofeats.loggers import get_logger
from cocofeats.nodes import register_node
from cocofeats.utils import _resolve_eval_strings
from cocofeats.writers import _json_safe
from cocofeats.loaders import load_meeg

log = get_logger(__name__)

CallableLike = Callable[[np.ndarray], Any]

if TYPE_CHECKING:  # pragma: no cover - typing helpers only
    from mne import BaseEpochs as MNEEpochs  # type: ignore[import-not-found]
    from mne.io import BaseRaw as MNERaw  # type: ignore[import-not-found]
else:  # pragma: no cover - runtime fallback when MNE is absent
    MNEEpochs = object
    MNERaw = object

DataLike = (
    NodeResult
    | xr.DataArray
    | xr.Dataset
    | np.ndarray
    | MNERaw
    | MNEEpochs
    | str
    | os.PathLike[str]
)


class _FactoryError(ValueError):
    """Internal helper error for consistent exception types."""


@dataclass(slots=False) # break parallelization?
class _SliceParameterCache:
    """Cached view of a per-slice argument or keyword value."""

    name: str | None
    dims: tuple[str, ...]
    shape: tuple[int, ...]
    provided_dims: tuple[str, ...]
    flat_values: np.ndarray
    source: str

    def value_at(self, index: int) -> Any:
        value = self.flat_values[index]
        if isinstance(value, np.generic):
            return value.item()
        return value

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dims": list(self.dims),
            "shape": [int(v) for v in self.shape],
            "provided_dims": list(self.provided_dims),
            "source": self.source,
        }


@dataclass(slots=False) # Breaks parallelization?
class _ArgumentSpec:
    """Descriptor for positional/keyword arguments resolved per slice."""

    name: str | None
    provider: _SliceParameterCache | None
    constant_value: Any

    def value_at(self, index: int) -> Any:
        if self.provider is None:
            return self.constant_value
        return self.provider.value_at(index)


def _is_potential_data_like(value: Any) -> bool:
    if xr is not None and isinstance(value, (xr.DataArray, xr.Dataset)):
        return True
    if isinstance(value, (NodeResult, np.ndarray)):
        return True
    if isinstance(value, (str, os.PathLike)):
        text = os.fspath(value).lower()
        return ".nc" in text or ".fif" in text
    if mne is not None:
        base_raw = getattr(mne.io, "BaseRaw", None)
        base_epochs = getattr(mne, "BaseEpochs", None)
        if base_raw is not None and isinstance(value, base_raw):
            return True
        if base_epochs is not None and isinstance(value, base_epochs):
            return True
    return False


def _resolve_callable(candidate: CallableLike | str) -> CallableLike:
    """Return a callable from a candidate value.

    The ``candidate`` can already be callable, an ``eval%`` expression (handled
    via :func:`_resolve_eval_strings`), or a dotted import path such as
    ``"numpy.mean"``.  Any other input raises ``TypeError``.
    """

    if callable(candidate):
        return candidate

    if isinstance(candidate, str):
        resolved = _resolve_eval_strings(candidate)
        if callable(resolved):
            return resolved

        if isinstance(resolved, str):
            module_path, sep, attr = resolved.rpartition(".")
            if sep == "":
                raise TypeError(
                    "String callables must be a dotted import path (e.g. 'numpy.mean')."
                )
            module = import_module(module_path)
            target = getattr(module, attr)
            if callable(target):
                return target
            raise TypeError(f"Attribute '{attr}' in module '{module_path}' is not callable.")

    raise TypeError("pure_function must be callable or a resolvable string reference.")


def _ensure_dataarray(data_like: DataLike, *, context: str) -> xr.DataArray:
    """Normalise supported inputs to an ``xarray.DataArray``.

    Parameters
    ----------
    data_like
        Supported inputs include ``NodeResult`` instances containing ``.nc``
        artifacts, ``xarray`` datasets/arrays, NumPy arrays, ``mne`` Raw/Epochs
        objects, or filesystem paths pointing to NetCDF files or FIF files.
    context
        Short description used in error messages.
    """

    if isinstance(data_like, xr.DataArray):
        return data_like

    if isinstance(data_like, NodeResult):
        if ".nc" not in data_like.artifacts:
            raise _FactoryError(
                f"{context}: NodeResult inputs must contain a '.nc' artifact with an xarray payload."
            )
        candidate = data_like.artifacts[".nc"].item
        return _ensure_dataarray(candidate, context=context)

    if isinstance(data_like, xr.Dataset):
        if len(data_like.data_vars) != 1:
            raise _FactoryError(
                f"{context}: xarray.Dataset inputs must expose exactly one data variable."
            )
        return next(iter(data_like.data_vars.values()))

    if isinstance(data_like, np.ndarray):
        # Expose anonymous dimensions (dim_0, dim_1, ...) for bare numpy arrays.
        dims = tuple(f"dim_{idx}" for idx in range(data_like.ndim))
        return xr.DataArray(data_like, dims=dims)

    if isinstance(data_like, (str, os.PathLike)):
        if '.nc' in str(data_like).lower():
            path = os.fspath(data_like)
            log.debug("Loading DataArray from path", path=path)
            return xr.load_dataarray(path)
        elif '.fif' in str(data_like).lower() and mne is not None:
            log.debug("Loading MNE Raw from path", path=os.fspath(data_like))
            data_like = load_meeg(data_like)
        else:
            raise _FactoryError(
                f"{context}: string/PathLike inputs must point to '.nc' or '.fif' files."
            )

    if mne is not None and isinstance(data_like, mne.io.BaseRaw):
        data = data_like.get_data()
        coords = {
            "spaces": ("spaces", list(data_like.ch_names)),
            "times": ("times", data_like.times.copy()),
        }
        arr = xr.DataArray(data, dims=("spaces", "times"), coords=coords)
        arr.attrs.setdefault(
            "metadata",
            json.dumps(
                {
                    "source": "mne.Raw",  # lightweight provenance
                    "sfreq": float(data_like.info.get("sfreq", 0.0)),
                    "n_times": int(data.shape[1]),
                    "n_channels": int(data.shape[0]),
                }
            ),
        )
        return arr

    if mne is not None and isinstance(data_like, mne.BaseEpochs):
        data = data_like.get_data()
        coords = {
            "epochs": ("epochs", np.arange(data.shape[0])),
            "spaces": ("spaces", list(data_like.ch_names)),
            "times": ("times", data_like.times.copy()),
        }
        arr = xr.DataArray(data, dims=("epochs", "spaces", "times"), coords=coords)
        arr.attrs.setdefault(
            "metadata",
            json.dumps(
                {
                    "source": "mne.Epochs",
                    "sfreq": float(data_like.info.get("sfreq", 0.0)),
                    "n_epochs": int(data.shape[0]),
                    "n_times": int(data.shape[2]),
                    "n_channels": int(data.shape[1]),
                }
            ),
        )
        return arr

    raise _FactoryError(f"{context}: unsupported input type '{type(data_like).__name__}'.")


def _align_candidate_dims(
    candidate: xr.DataArray,
    *,
    other_dims: Sequence[str],
    context: str,
) -> xr.DataArray:
    """Rename anonymous dimensions to match ``other_dims`` when possible."""

    allowed = set(other_dims)
    if set(candidate.dims).issubset(allowed):
        return candidate

    if len(candidate.dims) == len(other_dims):
        rename_map = {old: new for old, new in zip(candidate.dims, other_dims)}
        renamed = candidate.rename(rename_map)
        if set(renamed.dims).issubset(allowed):
            return renamed

    raise _FactoryError(
        f"{context}: argument dimensions {tuple(candidate.dims)} must be a subset of {tuple(other_dims)}."
    )


def _build_slice_parameter(
    value: Any,
    *,
    name: str | None,
    data_xr: xr.DataArray,
    dim: str,
    other_dims: Sequence[str],
    context: str,
) -> _SliceParameterCache | None:
    """Return a cached broadcast of per-slice argument values."""

    if not _is_potential_data_like(value):
        return None

    if not other_dims:
        # Single slice scenario: treat values as constants.
        return None

    candidate = _ensure_dataarray(value, context=context)

    if candidate.ndim == 0:
        return None

    if dim in candidate.dims:
        raise _FactoryError(
            f"{context}: per-slice parameters must not depend on the iteration dimension '{dim}'."
        )

    candidate = _align_candidate_dims(candidate, other_dims=other_dims, context=context)

    template = data_xr.transpose(*other_dims, dim).isel({dim: 0})
    try:
        broadcast = candidate.broadcast_like(template)
    except ValueError as exc:  # pragma: no cover - xarray raises informative errors
        raise _FactoryError(
            f"{context}: unable to broadcast argument to dimensions {tuple(other_dims)} (reason: {exc})."
        ) from exc

    broadcast = broadcast.transpose(*other_dims)
    shape = tuple(int(broadcast.sizes[d]) for d in other_dims)

    if np.prod(shape, dtype=int) == 0:
        raise _FactoryError(f"{context}: broadcast produced empty parameter values.")

    flat_values = np.asarray(broadcast.values).reshape(-1)

    expected = int(np.prod(shape, dtype=int))
    if flat_values.size != expected:
        raise _FactoryError(
            f"{context}: internal error aligning parameter values; expected {expected} elements, got {flat_values.size}."
        )

    provider = _SliceParameterCache(
        name=name,
        dims=tuple(other_dims),
        shape=shape,
        provided_dims=tuple(candidate.dims),
        flat_values=flat_values,
        source=type(value).__name__,
    )
    return provider


def _prepare_argument_specs(
    args: Sequence[Any],
    kwargs: Mapping[str, Any],
    *,
    data_xr: xr.DataArray,
    dim: str,
    other_dims: Sequence[str],
) -> tuple[list[_ArgumentSpec], dict[str, _ArgumentSpec], dict[str, Any], bool]:
    """Classify positional and keyword arguments into per-slice providers."""

    arg_specs: list[_ArgumentSpec] = []
    arg_details: list[dict[str, Any]] = []
    kwarg_specs: dict[str, _ArgumentSpec] = {}
    kwarg_details: dict[str, Any] = {}
    has_per_slice = False

    for idx, value in enumerate(args):
        context = f"apply_1d args[{idx}]"
        provider = _build_slice_parameter(
            value,
            name=f"arg_{idx}",
            data_xr=data_xr,
            dim=dim,
            other_dims=other_dims,
            context=context,
        )
        if provider is None:
            arg_specs.append(_ArgumentSpec(name=None, provider=None, constant_value=value))
        else:
            has_per_slice = True
            arg_specs.append(_ArgumentSpec(name=provider.name, provider=provider, constant_value=None))
            detail = provider.describe()
            detail["position"] = idx
            arg_details.append(detail)

    for key, value in kwargs.items():
        context = f"apply_1d kwargs['{key}']"
        provider = _build_slice_parameter(
            value,
            name=key,
            data_xr=data_xr,
            dim=dim,
            other_dims=other_dims,
            context=context,
        )
        if provider is None:
            kwarg_specs[key] = _ArgumentSpec(name=key, provider=None, constant_value=value)
        else:
            has_per_slice = True
            kwarg_specs[key] = _ArgumentSpec(name=key, provider=provider, constant_value=None)
            kwarg_details[key] = provider.describe()

    metadata: dict[str, Any] = {}
    if arg_details:
        metadata["args"] = arg_details
    if kwarg_details:
        metadata["kwargs"] = kwarg_details

    return arg_specs, kwarg_specs, metadata, has_per_slice


def _normalise_result(value: Any, *, dtype: np.dtype | None) -> tuple[np.ndarray, bool]:
    """Convert function outputs into numpy arrays and flag scalars.

    Returns a tuple ``(array, is_scalar)`` where ``array`` is either a 0-D numpy
    array (scalar case) or a 1-D numpy array.
    """

    if isinstance(value, xr.DataArray):
        value = value.values

    if isinstance(value, np.ndarray):
        arr = value
    else:
        arr = np.array(value)

    if arr.ndim > 1:
        raise _FactoryError("Pure functions must return scalars or 1-D sequences.")

    if dtype is not None and arr.dtype != dtype:
        arr = arr.astype(dtype, copy=False)

    return arr, arr.ndim == 0


def _vectorised_apply(
    data_xr: xr.DataArray,
    *,
    dim: str,
    func: CallableLike,
    args: Sequence[Any],
    kwargs: Mapping[str, Any],
    result_dim: str | None,
    result_coords: Sequence[Any] | None,
    output_dtype: np.dtype | None,
) -> tuple[xr.DataArray, dict[str, Any]]:
    """Apply a 1-D function along ``dim`` and return output + metadata."""

    if dim not in data_xr.dims:
        raise _FactoryError(
            f"Requested dimension '{dim}' not found in input dims {tuple(data_xr.dims)}."
        )

    other_dims = [d for d in data_xr.dims if d != dim]
    if data_xr.sizes[dim] == 0:
        raise _FactoryError(f"Input dimension '{dim}' has size zero; cannot build 1-D slices.")
    for axis in other_dims:
        if data_xr.sizes[axis] == 0:
            raise _FactoryError(
                f"Input dimension '{axis}' has size zero; cannot evaluate pure function across slices."
            )

    selector = {d: 0 for d in other_dims}
    sample_vector = np.asarray(data_xr.isel(selector).data)
    if sample_vector.ndim != 1:
        sample_vector = sample_vector.reshape(-1)

    first_output = func(sample_vector, *args, **kwargs)
    inferred_dtype = np.dtype(output_dtype) if output_dtype is not None else None
    first_array, is_scalar = _normalise_result(first_output, dtype=inferred_dtype)

    if inferred_dtype is None:
        inferred_dtype = first_array.dtype

    if not is_scalar:
        if first_array.ndim != 1:
            raise _FactoryError("Non-scalar outputs must be one-dimensional sequences.")
        result_dim_name = result_dim or "outputs"
        result_length = int(first_array.shape[0])
        if result_length == 0:
            raise _FactoryError("Pure function returned an empty sequence; cannot build outputs.")
        if result_coords is not None and len(tuple(result_coords)) != result_length:
            raise _FactoryError(
                "Length of result_coords does not match the function output length."
            )
        output_core_dims = [[result_dim_name]]
        output_sizes = {result_dim_name: result_length}
        first_shape = first_array.shape
    else:
        result_dim_name = None
        result_length = None
        output_core_dims = [[]]
        output_sizes = None
        first_shape = ()

    def _wrapped(vector: np.ndarray) -> np.ndarray:
        flat = np.asarray(vector)
        if flat.ndim != 1:
            flat = flat.reshape(-1)
        output = func(flat, *args, **kwargs)
        result_array, scalar_flag = _normalise_result(output, dtype=inferred_dtype)
        if scalar_flag != is_scalar:
            raise _FactoryError("Pure function returned inconsistent scalar/non-scalar outputs.")
        if not scalar_flag and result_array.shape != first_shape:
            raise _FactoryError("Pure function returned sequences of varying length.")
        return result_array

    apply_kwargs: dict[str, Any] = {
        "input_core_dims": [[dim]],
        "output_core_dims": output_core_dims,
        "output_dtypes": [inferred_dtype],
        "vectorize": True,
        "keep_attrs": True,
        "dask": "parallelized",
    }
    if output_sizes:
        apply_kwargs["output_sizes"] = output_sizes
    time_start = time.perf_counter()
    result_da = xr.apply_ufunc(_wrapped, data_xr, **apply_kwargs)
    time_end = time.perf_counter()
    log.debug(
        "Vectorized apply_ufunc completed",
        duration_seconds=time_end - time_start,
        input_shape=tuple(data_xr.shape),
        output_shape=tuple(result_da.shape),
    )

    input_dims = list(data_xr.dims)
    if result_dim_name is None:
        target_dims = [d for d in input_dims if d != dim]
    else:
        idx = input_dims.index(dim)
        target_dims = input_dims.copy()
        target_dims[idx : idx + 1] = [result_dim_name]
    result_da = result_da.transpose(*target_dims)

    if result_dim_name is not None:
        if result_coords is not None:
            result_da = result_da.assign_coords({result_dim_name: list(result_coords)})
        elif result_dim_name not in result_da.coords:
            result_da = result_da.assign_coords({result_dim_name: np.arange(first_shape[0])})

    metadata = {
        "mode": "vectorized",
        "factory": "xarray_factory",
        "dimension": dim,
        "is_scalar": is_scalar,
        "result_dimension": result_dim_name,
        "result_length": result_length,
        "input_dims": list(data_xr.dims),
        "input_shape": [int(data_xr.sizes[d]) for d in data_xr.dims],
        "output_dims": list(result_da.dims),
        "output_shape": [int(result_da.sizes[d]) for d in result_da.dims],
        "function": getattr(func, "__name__", repr(func)),
        "function_module": getattr(func, "__module__", None),
        "apply_ufunc_kwargs": {str(k): _json_safe(v) for k, v in apply_kwargs.items()},
        "apply_ufunc_duration_seconds": time_end - time_start,
    }

    return result_da, metadata


def _iterative_apply(
    data_xr: xr.DataArray,
    *,
    dim: str,
    func: CallableLike,
    arg_specs: Sequence[_ArgumentSpec],
    kwarg_specs: Mapping[str, _ArgumentSpec],
    result_dim: str | None,
    result_coords: Sequence[Any] | None,
    output_dtype: np.dtype | None,
) -> tuple[xr.DataArray, dict[str, Any], xr.DataArray]:
    """Iteratively apply a 1-D function along ``dim`` and capture timings."""

    if dim not in data_xr.dims:
        raise _FactoryError(
            f"Requested dimension '{dim}' not found in input dims {tuple(data_xr.dims)}."
        )

    if data_xr.sizes[dim] == 0:
        raise _FactoryError(f"Input dimension '{dim}' has size zero; cannot build 1-D slices.")

    for axis in data_xr.dims:
        if axis != dim and data_xr.sizes[axis] == 0:
            raise _FactoryError(
                f"Input dimension '{axis}' has size zero; cannot evaluate pure function across slices."
            )

    input_dims = list(data_xr.dims)
    other_dims = [d for d in input_dims if d != dim]
    ordered_dims = other_dims + [dim]
    target_length = data_xr.sizes[dim]

    arr = data_xr.transpose(*ordered_dims).values
    flat = arr.reshape((-1, target_length))

    if flat.size == 0:
        raise _FactoryError("Cannot apply factory to empty arrays.")

    first_args = tuple(spec.value_at(0) for spec in arg_specs)
    first_kwargs = {key: spec.value_at(0) for key, spec in kwarg_specs.items()}

    first_output = func(flat[0], *first_args, **first_kwargs)
    inferred_dtype = np.dtype(output_dtype) if output_dtype is not None else None
    first_array, is_scalar = _normalise_result(first_output, dtype=inferred_dtype)

    if inferred_dtype is None:
        inferred_dtype = first_array.dtype

    if not is_scalar:
        if first_array.ndim != 1:
            raise _FactoryError("Non-scalar outputs must be one-dimensional sequences.")
        result_dim_name = result_dim or "outputs"
        result_length = int(first_array.shape[0])
        if result_length == 0:
            raise _FactoryError("Pure function returned an empty sequence; cannot build outputs.")
        if result_coords is not None and len(tuple(result_coords)) != result_length:
            raise _FactoryError(
                "Length of result_coords does not match the function output length."
            )
    else:
        result_dim_name = None
        result_length = None

    if is_scalar:
        target_dims = [d for d in input_dims if d != dim]
    else:
        target_dims = [result_dim_name if d == dim else d for d in input_dims]

    output_shape: list[int] = []
    for dim_name in target_dims:
        if dim_name == result_dim_name:
            output_shape.append(result_length)  # type: ignore[arg-type]
        else:
            output_shape.append(int(data_xr.sizes[dim_name]))

    result_storage = np.empty(tuple(output_shape), dtype=inferred_dtype)
    timing_storage = np.empty(tuple(output_shape), dtype=float)

    if other_dims:
        index_iter = np.ndindex(*(data_xr.sizes[d] for d in other_dims))
    else:
        index_iter = iter([()])

    first_checked = False
    for flat_idx, (base_index, vector) in enumerate(zip(index_iter, flat)):
        call_args = tuple(spec.value_at(flat_idx) for spec in arg_specs)
        call_kwargs = {key: spec.value_at(flat_idx) for key, spec in kwarg_specs.items()}
        start = time.perf_counter()
        output = func(vector, *call_args, **call_kwargs)
        duration = time.perf_counter() - start
        result_array, scalar_flag = _normalise_result(output, dtype=inferred_dtype)

        if not first_checked:
            if scalar_flag != is_scalar:
                raise _FactoryError("Pure function returned inconsistent scalar/non-scalar outputs.")
            if not scalar_flag and result_array.shape != first_array.shape:
                raise _FactoryError("Pure function returned sequences of varying length.")
            first_checked = True

        index_by_dim = {name: idx for name, idx in zip(other_dims, base_index)}
        target_index: list[Any] = []
        for dim_name in target_dims:
            if dim_name == result_dim_name:
                target_index.append(slice(None))
            else:
                target_index.append(index_by_dim.get(dim_name, 0))

        key = tuple(target_index)

        if is_scalar:
            result_storage[key] = result_array.item()
            timing_storage[key] = duration
        else:
            result_storage[key] = result_array
            timing_storage[key] = np.full(result_array.shape, duration, dtype=float)

    coords: dict[str, Any] = {}
    for dim_name in target_dims:
        if dim_name == result_dim_name:
            if result_coords is not None:
                coords[dim_name] = (dim_name, list(result_coords))
            else:
                coords[dim_name] = (dim_name, np.arange(result_length))  # type: ignore[arg-type]
        elif dim_name in data_xr.coords:
            coords[dim_name] = data_xr.coords[dim_name]

    result_da = xr.DataArray(result_storage, dims=target_dims)
    timing_da = xr.DataArray(timing_storage, dims=target_dims)
    if coords:
        result_da = result_da.assign_coords(coords)
        timing_da = timing_da.assign_coords(coords)

    metadata = {
        "mode": "iterative",
        "factory": "xarray_factory",
        "dimension": dim,
        "is_scalar": is_scalar,
        "result_dimension": result_dim_name,
        "result_length": result_length,
        "input_dims": list(data_xr.dims),
        "input_shape": [int(data_xr.sizes[d]) for d in data_xr.dims],
        "output_dims": list(result_da.dims),
        "output_shape": [int(result_da.sizes[d]) for d in result_da.dims],
        "function": getattr(func, "__name__", repr(func)),
        "function_module": getattr(func, "__module__", None),
    }

    return result_da, metadata, timing_da


def apply_1d(
    data_like: DataLike,
    *,
    dim: str,
    pure_function: CallableLike | str,
    args: Sequence[Any] | None = None,
    kwargs: Mapping[str, Any] | None = None,
    result_dim: str | None = None,
    result_coords: Sequence[Any] | None = None,
    output_dtype: Any | None = None,
    metadata: Mapping[str, Any] | None = None,
    keep_input_metadata: bool = True,
    mode: str = "vectorized",
) -> xr.DataArray:
    """Apply a 1-D pure function across a chosen xarray dimension.

    Parameters
    ----------
    data_like
        Input data. See :func:`_ensure_dataarray` for supported types.
    dim
        Dimension name whose slices are fed to the pure function.
    pure_function
        Callable (or resolvable string) that accepts a one-dimensional numpy
        array and returns either a scalar or a one-dimensional sequence.
    args, kwargs
        Optional positional and keyword arguments forwarded to
        ``pure_function`` for each slice.
    result_dim
        Name of the new dimension when the pure function returns a sequence. If
        omitted, ``"outputs"`` is used.
    result_coords
        Coordinate labels for ``result_dim``. Length must match the sequence
        length returned by the pure function.
    output_dtype
        Optional numpy dtype override for the output array.
    metadata
        Extra metadata merged into the automatically generated metadata block.
    keep_input_metadata
        When ``True`` (default) and the input carries a ``metadata`` attribute,
        it is nested under ``source_metadata`` in the output metadata.
    mode
        Execution strategy. ``"vectorized"`` (default) leverages
        :func:`xarray.apply_ufunc`, while ``"iterative"`` performs a Python loop
        and records per-slice timings (stored in metadata).
    """

    func = _resolve_callable(pure_function)
    args = tuple(args or ())
    kwargs = dict(kwargs or {})

    data_xr = _ensure_dataarray(data_like, context="apply_1d")
    other_dims = [d for d in data_xr.dims if d != dim]
    #data_xr_orig_coords = data_xr.coords.copy(deep=True)
    #data_xr_orig_dims = data_xr.dims.copy()

    arg_specs, kwarg_specs, per_slice_details, has_per_slice = _prepare_argument_specs(
        args,
        kwargs,
        data_xr=data_xr,
        dim=dim,
        other_dims=other_dims,
    )

    normalised_mode = mode.lower()
    if normalised_mode not in {"vectorized", "iterative"}:
        raise ValueError("mode must be either 'vectorized' or 'iterative'.")

    if has_per_slice and normalised_mode != "iterative":
        raise ValueError("Per-slice arguments require mode='iterative'.")

    dtype_override = np.dtype(output_dtype) if output_dtype is not None else None

    if normalised_mode == "vectorized":
        result_da, auto_metadata = _vectorised_apply(
            data_xr,
            dim=dim,
            func=func,
            args=args,
            kwargs=kwargs,
            result_dim=result_dim,
            result_coords=result_coords,
            output_dtype=dtype_override,
        )
        timing_da: xr.DataArray | None = None
    else:
        result_da, auto_metadata, timing_da = _iterative_apply(
            data_xr,
            dim=dim,
            func=func,
            arg_specs=arg_specs,
            kwarg_specs=kwarg_specs,
            result_dim=result_dim,
            result_coords=result_coords,
            output_dtype=dtype_override,
        )

    combined_metadata = dict(auto_metadata)
    if arg_specs:
        combined_metadata["function_args"] = [
            _json_safe(spec.constant_value) if spec.provider is None else "<per-slice>"
            for spec in arg_specs
        ]
    if kwarg_specs:
        combined_metadata["function_kwargs"] = {
            str(k): (_json_safe(v.constant_value) if v.provider is None else "<per-slice>")
            for k, v in kwarg_specs.items()
        }
    if metadata:
        combined_metadata.update({str(k): _json_safe(v) for k, v in metadata.items()})
    if keep_input_metadata and "metadata" in data_xr.attrs:
        combined_metadata["source_metadata"] = data_xr.attrs["metadata"]
    if per_slice_details:
        combined_metadata["per_slice_arguments"] = _json_safe(per_slice_details)
    if timing_da is not None:
        combined_metadata["per_slice_duration"] = _json_safe(timing_da)
        combined_metadata["per_slice_duration_unit"] = "seconds"

    result_da.attrs["metadata"] = json.dumps(combined_metadata, indent=2, default=_json_safe)



    return result_da


@register_node(name="xarray_factory", override=True)
def xarray_factory(
    data_like: DataLike,
    *,
    dim: str,
    pure_function: CallableLike | str,
    args: Sequence[Any] | None = None,
    kwargs: Mapping[str, Any] | None = None,
    result_dim: str | None = None,
    result_coords: Sequence[Any] | None = None,
    output_dtype: Any | None = None,
    metadata: Mapping[str, Any] | None = None,
    keep_input_metadata: bool = True,
    mode: str = "vectorized",
) -> NodeResult:
    """Node entry-point delegating to :func:`apply_1d`.

    This thin wrapper enables declarative use from pipeline YAML definitions by
    exposing the factory through the standard node registry.  All parameters are
    forwarded verbatim to :func:`apply_1d`.
    """

    result_da = apply_1d(
        data_like,
        dim=dim,
        pure_function=pure_function,
        args=args,
        kwargs=kwargs,
        result_dim=result_dim,
        result_coords=result_coords,
        output_dtype=output_dtype,
        metadata=metadata,
        keep_input_metadata=keep_input_metadata,
        mode=mode,
    )
    artifacts = {
        ".nc": Artifact(
            item=result_da,
            writer=lambda path: result_da.to_netcdf(path, engine="netcdf4", format="NETCDF4"),
        )
    }
    return NodeResult(artifacts=artifacts)

__all__ = ["apply_1d", "xarray_factory"]
