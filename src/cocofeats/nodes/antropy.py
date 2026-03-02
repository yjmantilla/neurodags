"""Node wrappers for the ``antropy`` derivative functions.

Each wrapper exposes an ``antropy`` computation through the node registry by
delegating to :func:`cocofeats.nodes.factories.apply_1d`.  The nodes expect an
``xarray``-like input and apply the underlying function to one-dimensional
slices along the requested dimension.
"""

from __future__ import annotations

from functools import partial
from typing import Any, Mapping, Sequence

import numpy as np

try:
    from antropy import (
        app_entropy,
        detrended_fluctuation,
        higuchi_fd,
        hjorth_params,
        katz_fd,
        lziv_complexity,
        num_zerocross,
        perm_entropy,
        petrosian_fd,
        sample_entropy,
        spectral_entropy,
        svd_entropy,
    )
except ImportError as exc:  # pragma: no cover - optional dependency guidance
    raise ImportError(
        "The 'antropy' extra is required for cocofeats.nodes.antropy. Install it via 'pip install antropy'."
    ) from exc

from cocofeats.definitions import Artifact, NodeResult
from cocofeats.nodes import register_node
from cocofeats.nodes.factories import apply_1d


_HJORTH_RESULT_DIM = "hjorthComponents"
_HJORTH_RESULT_LABELS = ("mobility", "complexity")


def _to_netcdf_writer(data_array):
    return lambda path, arr=data_array: arr.to_netcdf(path, engine="netcdf4", format="NETCDF4")


def _build_node(
    func,
    *,
    name: str,
    default_result_dim: str | None = None,
    default_result_coords: Sequence[str] | None = None,
    default_mode: str = "iterative",
) -> None:
    """Register a node that applies ``func`` along a chosen dimension."""

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
        """Apply ``antropy.{name}`` along ``dim``."""

        resolved_result_dim = result_dim if result_dim is not None else default_result_dim
        resolved_result_coords: Sequence[str] | None
        if result_coords is not None:
            resolved_result_coords = result_coords
        elif default_result_coords is not None:
            resolved_result_coords = tuple(default_result_coords)
        else:
            resolved_result_coords = None

        result_da = apply_1d(
            data_like,
            dim=dim,
            pure_function=func,
            args=tuple(function_args or ()),
            kwargs=function_kwargs,
            result_dim=resolved_result_dim,
            result_coords=resolved_result_coords,
            metadata=metadata,
            keep_input_metadata=keep_input_metadata,
            mode=mode,
        )

        artifact = Artifact(item=result_da, writer=_to_netcdf_writer(result_da))
        return NodeResult(artifacts={".nc": artifact})

    _node.__doc__ = f"Node wrapper for antropy.{name}."


_build_node(app_entropy, name="antropy_app_entropy")
_build_node(hjorth_params, name="antropy_hjorth_params", default_result_dim=_HJORTH_RESULT_DIM, default_result_coords=_HJORTH_RESULT_LABELS)
_build_node(lziv_complexity, name="antropy_lziv_complexity")
_build_node(num_zerocross, name="antropy_num_zerocross")
_build_node(perm_entropy, name="antropy_perm_entropy")
_build_node(sample_entropy, name="antropy_sample_entropy")
_build_node(spectral_entropy, name="antropy_spectral_entropy")
_build_node(svd_entropy, name="antropy_svd_entropy")
_build_node(detrended_fluctuation, name="antropy_detrended_fluctuation")
_build_node(higuchi_fd, name="antropy_higuchi_fd")
_build_node(katz_fd, name="antropy_katz_fd")
_build_node(petrosian_fd, name="antropy_petrosian_fd")

__all__ = [
    "antropy_app_entropy",
    "antropy_detrended_fluctuation",
    "antropy_higuchi_fd",
    "antropy_hjorth_params",
    "antropy_katz_fd",
    "antropy_lziv_complexity",
    "antropy_num_zerocross",
    "antropy_perm_entropy",
    "antropy_petrosian_fd",
    "antropy_sample_entropy",
    "antropy_spectral_entropy",
    "antropy_svd_entropy",
]

