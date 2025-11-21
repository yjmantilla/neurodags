"""Node wrappers for the ``neurokit2`` feature functions.

The wrappers delegate numerical computation to :func:`cocofeats.nodes.factories.apply_1d`
while collecting per-slice metadata and optional Matplotlib figures returned by
``neurokit2`` routines.  Results are packaged into an ``xarray.Dataset`` that
contains:

* ``value`` – the direct numerical output of the ``neurokit2`` function,
* ``metadata`` – JSON-serialisable metadata associated with each slice, and
* optional ``figure_*`` variables that hold the encoded Matplotlib output.
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import xarray as xr
matplotlib.use("Agg", force=True) # avoid GUI backends for figure generation

try:  # pragma: no cover - optional dependency guidance
    import neurokit2 as nk
except ImportError as exc:  # pragma: no cover - optional dependency guidance
    raise ImportError(
        "The 'neurokit2' extra is required for cocofeats.nodes.neurokit. Install it via 'pip install neurokit2'."
    ) from exc

try:  # pragma: no cover - optional dependency guidance
    import pandas as pd
except ImportError:  # pragma: no cover - optional dependency guidance
    pd = None  # type: ignore[assignment]

from cocofeats.definitions import Artifact, NodeResult
from cocofeats.nodes import register_node
from cocofeats.nodes.factories import apply_1d
from cocofeats.writers import _json_safe


FigureEncoding = str | None
CallableLike = Callable[[np.ndarray], Any]


@dataclass(slots=False) # breaks parallelization?
class _CallRecord:
    metadata: Any | None
    figure: dict[str, np.ndarray] | None


class _ResultCollector:
    """Callable wrapper that records metadata and figure payloads per slice."""

    def __init__(
        self,
        func: CallableLike,
        *,
        figure_encoding: FigureEncoding,
        force_capture: bool,
    ) -> None:
        self._func = func
        self._figure_encoding = (figure_encoding or "none").lower()
        self._force_capture = force_capture and self._figure_encoding != "none"
        self._records: list[_CallRecord] = []

    def __call__(self, vector: np.ndarray, *args: Any, **kwargs: Any) -> Any:
        results = self._func(vector, *args, **kwargs)
        value, metadata_payload = _split_neurokit_results(results)

        figure_payload: dict[str, np.ndarray] | None = None
        try:
            if self._force_capture or bool(kwargs.get("show")):
                figure_payload = _capture_current_figure(self._figure_encoding)
        finally:
            # Always close figures raised by the underlying call to avoid leaks.
            plt.close("all")

        self._records.append(_CallRecord(metadata=metadata_payload, figure=figure_payload))
        return value

    @property
    def records(self) -> list[_CallRecord]:
        """Return per-invocation records excluding the preview lookahead call."""

        if len(self._records) <= 1:
            return []
        return self._records[1:]


def _capture_current_figure(mode: str) -> dict[str, np.ndarray] | None:
    """Return encoded representations for the current Matplotlib figure."""

    if mode in {"none", ""}:
        return None

    fig_numbers = plt.get_fignums()
    if not fig_numbers:
        return None

    fig = plt.gcf()
    payload: dict[str, np.ndarray] = {}

    if mode in {"png", "both"}:  # Encode figure as PNG bytes stored in a uint8 array
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        payload["png_bytes"] = np.frombuffer(buf.getvalue(), dtype=np.uint8)

    if mode in {"rgba", "both"}:  # Encode raw RGBA pixels
        fig.canvas.draw()
        payload["rgba"] = np.array(fig.canvas.buffer_rgba(), dtype=np.uint8)

    return payload or None


def _split_neurokit_results(results: Any) -> tuple[Any, Any | None]:
    """Normalise ``neurokit2`` outputs into ``(value, metadata)`` pairs."""

    if isinstance(results, tuple) or isinstance(results, list):
        if not results:
            return np.nan, None
        value = results[0]
        if len(results) == 1:
            return value, None
        if len(results) == 2:
            return value, results[1]
        return value, {"secondary": results[1], "extra_outputs": list(results[2:])}

    return results, None


def _metadata_to_dict(metadata: Any | None) -> Mapping[str, Any]:
    """Convert metadata payloads to a flat dictionary of JSON-safe values."""

    if metadata is None:
        return {}

    if isinstance(metadata, Mapping):
        return {str(key): _json_safe(value) for key, value in metadata.items()}

    if pd is not None and isinstance(metadata, pd.DataFrame):
        payload: dict[str, Any] = {}
        for column, values in metadata.to_dict(orient="list").items():
            payload[str(column)] = _json_safe(values)
        if metadata.index.name is not None:
            payload[f"index:{metadata.index.name}"] = _json_safe(list(metadata.index))
        return payload

    if pd is not None and isinstance(metadata, pd.Series):
        return {str(idx): _json_safe(val) for idx, val in metadata.items()}

    if isinstance(metadata, np.ndarray):
        return {"array": _json_safe(metadata)}

    if isinstance(metadata, (list, tuple)):
        return {"sequence": _json_safe(metadata)}

    if hasattr(metadata, "to_dict"):
        try:
            serialised = metadata.to_dict()
        except Exception:  # pragma: no cover - best effort fall-back
            pass
        else:
            return {"to_dict": _json_safe(serialised)}

    return {"value": _json_safe(metadata)}


def _build_metadata_array(
    value_da: xr.DataArray,
    records: list[_CallRecord],
) -> xr.DataArray | None:
    """Turn metadata dictionaries into an ``xarray.DataArray``."""

    metadata_dicts = [_metadata_to_dict(record.metadata) for record in records]
    if not metadata_dicts:
        return None

    all_keys: set[str] = set()
    for item in metadata_dicts:
        all_keys.update(item.keys())

    if not all_keys:
        return None

    ordered_keys = tuple(sorted(all_keys))

    flat_rows: list[list[str]] = []
    max_length = 0
    for entry in metadata_dicts:
        row: list[str] = []
        for key in ordered_keys:
            value = entry.get(key, None)
            serialised = json.dumps(value, ensure_ascii=False, default=_json_safe)
            row.append(serialised)
            max_length = max(max_length, len(serialised))
        flat_rows.append(row)

    if max_length == 0:
        max_length = 1

    flat_array = np.array(flat_rows, dtype=f"<U{max_length}")
    reshaped = flat_array.reshape(value_da.shape + (len(ordered_keys),))

    coords = {name: value_da.coords[name] for name in value_da.coords}
    coords["metadata_fields"] = list(ordered_keys)

    metadata_da = xr.DataArray(
        reshaped,
        dims=value_da.dims + ("metadata_fields",),
        coords=coords,
    )
    metadata_da.attrs["metadata_keys"] = json.dumps(ordered_keys)
    return metadata_da


def _build_figure_dataarrays(
    value_da: xr.DataArray,
    records: list[_CallRecord],
) -> dict[str, xr.DataArray]:
    """Create ``xarray`` representations for any captured figures."""

    png_payloads = [rec.figure.get("png_bytes") if rec.figure else None for rec in records]
    rgba_payloads = [rec.figure.get("rgba") if rec.figure else None for rec in records]

    data_vars: dict[str, xr.DataArray] = {}

    if any(payload is not None for payload in png_payloads):
        valid_png = [payload for payload in png_payloads if payload is not None]
        lengths = {payload.size for payload in valid_png}
        if len(lengths) == 1:
            png_len = lengths.pop()
            stacked = np.stack([
                payload if payload is not None else np.zeros(png_len, dtype=np.uint8)
                for payload in png_payloads
            ])
            reshaped = stacked.reshape(value_da.shape + (png_len,))
            coords = dict(value_da.coords)
            coords["png_byte"] = np.arange(png_len)
            data_vars["figure_png"] = xr.DataArray(
                reshaped,
                dims=value_da.dims + ("png_byte",),
                coords=coords,
            )
        else:  # pragma: no cover - differing figure sizes are unexpected but handled
            strings = [
                "" if payload is None else payload.tobytes().hex()
                for payload in png_payloads
            ]
            max_len = max((len(s) for s in strings), default=1)
            array = np.array(strings, dtype=f"<U{max_len}")
            reshaped = array.reshape(value_da.shape)
            data_vars["figure_png_hex"] = xr.DataArray(
                reshaped,
                dims=value_da.dims,
                coords=value_da.coords,
            )

    if any(payload is not None for payload in rgba_payloads):
        valid_rgba = [payload for payload in rgba_payloads if payload is not None]
        shapes = {payload.shape for payload in valid_rgba}
        if len(shapes) == 1:
            height, width, channels = next(iter(shapes))
            stacked = np.stack([
                payload if payload is not None else np.zeros((height, width, channels), dtype=np.uint8)
                for payload in rgba_payloads
            ])
            reshaped = stacked.reshape(value_da.shape + (height, width, channels))
            coords = dict(value_da.coords)
            coords["figure_y"] = np.arange(height)
            coords["figure_x"] = np.arange(width)
            coords["figure_channel"] = np.arange(channels)
            data_vars["figure_rgba"] = xr.DataArray(
                reshaped,
                dims=value_da.dims + ("figure_y", "figure_x", "figure_channel"),
                coords=coords,
            )
        else:  # pragma: no cover - differing figure sizes are unexpected but handled
            strings = [
                "" if payload is None else payload.tobytes().hex()
                for payload in rgba_payloads
            ]
            max_len = max((len(s) for s in strings), default=1)
            array = np.array(strings, dtype=f"<U{max_len}")
            reshaped = array.reshape(value_da.shape)
            data_vars["figure_rgba_hex"] = xr.DataArray(
                reshaped,
                dims=value_da.dims,
                coords=value_da.coords,
            )

    return data_vars


def _to_netcdf_writer(dataset: xr.Dataset) -> Callable[[str], None]:
    return lambda path: dataset.to_netcdf(path, engine="netcdf4", format="NETCDF4")


def _build_node(name: str, func: CallableLike) -> None:
    """Register ``neurokit2`` function ``func`` as a node."""

    @register_node(name=name, override=True)
    def _node(  # type: ignore[override]
        data_like,
        *,
        dim: str,
        mode: str = "iterative",
        keep_input_metadata: bool = True,
        metadata: Mapping[str, Any] | None = None,
        function_args: Sequence[Any] | None = None,
        figure_encoding: FigureEncoding = "png",
        force_show: bool | None = None,
        **function_kwargs: Any,
    ) -> NodeResult:
        if mode.lower() != "iterative":
            raise ValueError("neurokit nodes currently require mode='iterative' to capture metadata and figures.")

        resolved_kwargs = dict(function_kwargs)
        resolved_args = tuple(function_args or ())

        capture_requested = figure_encoding not in {None, "none", ""}
        #if capture_requested and "show" not in resolved_kwargs:
        #    resolved_kwargs["show"] = True
        if force_show is not None:
            resolved_kwargs["show"] = bool(force_show)

        collector = _ResultCollector(
            func,
            figure_encoding=figure_encoding,
            force_capture=capture_requested,
        )

        value_da = apply_1d(
            data_like,
            dim=dim,
            pure_function=collector,
            args=resolved_args,
            kwargs=resolved_kwargs,
            metadata=metadata,
            keep_input_metadata=keep_input_metadata,
            mode="iterative",
        )

        records = collector.records
        metadata_da = _build_metadata_array(value_da, records)
        figure_vars = _build_figure_dataarrays(value_da, records)

        dataset_vars: dict[str, xr.DataArray] = {"value": value_da}
        if metadata_da is not None:
            dataset_vars["metadata"] = metadata_da

        dataset_vars.update(figure_vars)
        dataset = xr.Dataset(dataset_vars)
        if "metadata" in value_da.attrs:
            dataset.attrs["metadata"] = value_da.attrs["metadata"]

        artifact = Artifact(item=dataset, writer=_to_netcdf_writer(dataset))
        return NodeResult(artifacts={".nc": artifact})

    _node.__doc__ = f"Node wrapper for neurokit2.{name}."


_NEUROKIT_FUNCTIONS: Mapping[str, CallableLike] = {
    "neurokit_complexity_delay": nk.complexity_delay,
    "neurokit_entropy_multiscale": nk.entropy_multiscale,
}


for _name, _func in _NEUROKIT_FUNCTIONS.items():  # pragma: no cover - registration side effect
    _build_node(_name, _func)


__all__ = ["neurokit_complexity_delay"]
