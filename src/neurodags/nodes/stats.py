"""Node wrappers for simple statistical descriptors (kurtosis, RMS)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

try:
    from scipy.stats import kurtosis as _scipy_kurtosis
except ImportError as exc:
    raise ImportError(
        "The 'scipy' extra is required for neurodags.nodes.stats. Install via 'pip install scipy'."
    ) from exc

from neurodags.definitions import Artifact, NodeResult
from neurodags.nodes import register_node
from neurodags.nodes.factories import apply_1d


def _to_netcdf_writer(data_array):
    return lambda path, arr=data_array: arr.to_netcdf(path, engine="netcdf4", format="NETCDF4")


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x))))


def _build_node(func, *, name: str, default_mode: str = "iterative") -> None:
    @register_node(name=name, override=True)
    def _node(
        data_like,
        *,
        dim: str,
        mode: str = default_mode,
        keep_input_metadata: bool = True,
        metadata: Mapping[str, Any] | None = None,
        result_dim: str | None = None,
        result_coords: Sequence[str] | None = None,
        function_args: Sequence[Any] | None = None,
        **function_kwargs: Any,
    ) -> NodeResult:
        result_da = apply_1d(
            data_like,
            dim=dim,
            pure_function=func,
            args=tuple(function_args or ()),
            kwargs=function_kwargs,
            result_dim=result_dim,
            result_coords=result_coords,
            metadata=metadata,
            keep_input_metadata=keep_input_metadata,
            mode=mode,
        )
        artifact = Artifact(item=result_da, writer=_to_netcdf_writer(result_da))
        return NodeResult(artifacts={".nc": artifact})

    _node.__doc__ = f"Node wrapper for {name}."


_build_node(_scipy_kurtosis, name="stats_kurtosis")
_build_node(_rms, name="stats_rms")

__all__: list[str] = []
