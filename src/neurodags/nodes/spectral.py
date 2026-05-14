import copy
import io
import json
import math
import os
import time
from collections.abc import Mapping, Sequence
from itertools import permutations
from typing import Any, Literal

import mne
import numpy as np
import xarray as xr
from fooof import FOOOF

from neurodags.definitions import Artifact, NodeResult
from neurodags.loaders import load_meeg
from neurodags.loggers import get_logger
from neurodags.utils import _resolve_eval_strings
from neurodags.writers import _json_safe

from . import register_node

log = get_logger(__name__)


DEFAULT_BANDS: dict[str, tuple[float, float]] = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "preAlpha": (5.5, 8.0),
    "slowTheta": (4.0, 5.5),
}


def _gen_aperiodic_log(
    freqs: np.ndarray,
    params: np.ndarray,
    mode: str = "fixed",
) -> np.ndarray:
    """Generate FOOOF aperiodic component in log10 space from stored parameters."""
    offset = float(params[0])
    safe_freqs = np.maximum(freqs, 1e-10)
    if mode == "knee" and params.size >= 3:
        knee = float(params[1])
        exp = float(params[2])
        return offset - np.log10(np.abs(knee) + safe_freqs**exp)
    exp = float(params[1]) if params.size >= 2 else 1.0
    return offset - np.log10(safe_freqs**exp)


def _resolve_psd_dataarray(
    psd_like: NodeResult | xr.DataArray | str | os.PathLike[str],
) -> xr.DataArray:
    if isinstance(psd_like, xr.DataArray):
        return psd_like

    if isinstance(psd_like, NodeResult):
        if ".nc" not in psd_like.artifacts:
            raise ValueError("NodeResult does not contain a .nc artifact to process.")
        candidate = psd_like.artifacts[".nc"].item
        if isinstance(candidate, xr.DataArray):
            return candidate
        if isinstance(candidate, str | os.PathLike):
            return xr.open_dataarray(candidate)
        raise ValueError("Unsupported artifact payload for .nc in NodeResult.")

    if isinstance(psd_like, str | os.PathLike):
        return xr.open_dataarray(psd_like)

    raise ValueError("Input must be a NodeResult, xarray.DataArray, or path to netCDF artifact.")


def _resolve_fooof_dataarray(
    fooof_like: "NodeResult | xr.DataArray | xr.Dataset | str | os.PathLike[str]",
) -> xr.DataArray:
    """Resolve a FOOOF artifact to the DataArray of JSON strings."""
    if isinstance(fooof_like, xr.DataArray):
        return fooof_like

    if isinstance(fooof_like, xr.Dataset):
        if "fooof" in fooof_like:
            return fooof_like["fooof"]
        if len(fooof_like.data_vars) == 1:
            return next(iter(fooof_like.data_vars.values()))
        raise ValueError(
            "Dataset has multiple variables and no 'fooof' variable; pass the DataArray directly."
        )

    if isinstance(fooof_like, NodeResult):
        if ".nc" not in fooof_like.artifacts:
            raise ValueError("NodeResult does not contain a .nc artifact.")
        candidate = fooof_like.artifacts[".nc"].item
        if isinstance(candidate, xr.DataArray):
            return candidate
        if isinstance(candidate, xr.Dataset):
            return _resolve_fooof_dataarray(candidate)
        if isinstance(candidate, str | os.PathLike):
            try:
                return xr.open_dataarray(candidate)
            except Exception:
                return _resolve_fooof_dataarray(xr.open_dataset(candidate))
        raise ValueError(f"Unsupported .nc artifact payload type: {type(candidate)}")

    if isinstance(fooof_like, str | os.PathLike):
        try:
            return xr.open_dataarray(fooof_like)
        except Exception:
            return _resolve_fooof_dataarray(xr.open_dataset(fooof_like))

    raise ValueError(f"Cannot resolve FOOOF input of type {type(fooof_like)}")


@register_node
def mne_spectrum(
    meeg: mne.io.BaseRaw | mne.BaseEpochs,
    compute_psd_kwargs: dict[str, Any] | None = None,
    extra_artifacts: bool = False,
) -> NodeResult:
    """
    Compute the power spectral density of M/EEG data.

    Parameters
    ----------
    meeg : mne.io.BaseRaw or mne.BaseEpochs
        The M/EEG data to analyze. Can be raw data or epochs.
    compute_psd_kwargs : dict, optional
        Additional keyword arguments to pass to `mne.compute_psd`.
    extra_artifacts : bool, optional
        Whether to generate extra artifacts (MNE Report). Default is True.
    Returns
    -------
    dict
        A dictionary containing the power spectral density results, metadata, and artifacts (MNE Report).
    """

    if isinstance(meeg, NodeResult):
        if ".fif" in meeg.artifacts:
            meeg = meeg.artifacts[".fif"].item
        else:
            raise ValueError("NodeResult does not contain a .fif artifact to process.")

    if isinstance(meeg, str | os.PathLike):
        meeg = load_meeg(meeg)
        log.debug("MNEReport: loaded MNE object from file", input=meeg)

    if compute_psd_kwargs is None:
        compute_psd_kwargs = {}

    if "fmax" in compute_psd_kwargs:
        if (
            compute_psd_kwargs["fmax"] is not None
            and compute_psd_kwargs["fmax"] > meeg.info["sfreq"] / 2
        ):
            log.warning("fmax is greater than Nyquist frequency, adjusting to Nyquist")
            compute_psd_kwargs["fmax"] = meeg.info["sfreq"] / 2

    spectra = meeg.compute_psd(**compute_psd_kwargs)
    log.debug("MNEReport: computed spectra", spectra=spectra)

    if extra_artifacts:
        report = mne.Report(title="Spectrum", verbose="error")
        report.add_figure(spectra.plot(show=False), title="Spectrum")
        log.debug("MNEReport: computed report")

        extra_artifact = Artifact(
            item=report, writer=lambda path: report.save(path, overwrite=True, open_browser=False)
        )
    if isinstance(meeg, mne.io.BaseRaw):
        this_xarray = xr.DataArray(
            data=spectra.get_data(),
            dims=["spaces", "frequencies"],
            coords={"spaces": spectra.ch_names, "frequencies": spectra.freqs},
        )
    elif isinstance(meeg, mne.BaseEpochs):
        this_xarray = xr.DataArray(
            data=spectra.get_data(),
            dims=["epochs", "spaces", "frequencies"],
            coords={
                "epochs": list(range(len(spectra))),
                "spaces": spectra.ch_names,
                "frequencies": spectra.freqs,
            },
        )

    this_metadata = {
        "compute_psd_kwargs": compute_psd_kwargs,
    }

    this_xarray.attrs["metadata"] = json.dumps(this_metadata, indent=2)

    # Also add metadata to the report
    if extra_artifacts:
        extra_artifact.item.add_html(
            f"<pre>{json.dumps(this_metadata, indent=2)}</pre>",
            title="Metadata",
            section="Metadata",
        )

    artifacts = {".nc": Artifact(item=this_xarray, writer=lambda path: this_xarray.to_netcdf(path))}

    if extra_artifacts:
        artifacts[".report.html"] = extra_artifact

    out = NodeResult(artifacts=artifacts)
    return out


@register_node
def mne_spectrum_array(
    meeg: mne.io.BaseRaw | mne.BaseEpochs,
    method: str = "welch",
    method_kwargs: dict[str, Any] | None = None,
) -> NodeResult:
    """Compute PSD from array data using Welch or multitaper algorithms.

    Parameters
    ----------
    meeg : mne.io.BaseRaw or mne.BaseEpochs
        The M/EEG data to analyze. Can be raw data or epochs.
    method : {"welch", "multitaper"}, optional
        PSD estimation routine to call. Defaults to "welch".
    method_kwargs : dict, optional
        Extra keyword arguments forwarded to the selected MNE function.

    Returns
    -------
    NodeResult
        An object containing the PSD as a ``.nc`` artifact (``xarray.Dataset``)
        plus metadata describing the output dimensions. When multitaper is used
        with ``output='complex'``, taper weights are included in the same
        dataset under the ``weights`` variable.
    """

    method = method.lower()
    if method not in {"welch", "multitaper"}:
        raise ValueError("method must be either 'welch' or 'multitaper'")

    method_kwargs = dict(method_kwargs or {})

    if isinstance(meeg, NodeResult):
        if ".fif" in meeg.artifacts:
            meeg = meeg.artifacts[".fif"].item
        else:
            raise ValueError("NodeResult does not contain a .fif artifact to process.")

    if isinstance(meeg, str | os.PathLike):
        meeg = load_meeg(meeg)
        log.debug("MNEReport: loaded MNE object from file", input=meeg)

    if isinstance(meeg, mne.io.BaseRaw):
        data_values = meeg.get_data()
        times = meeg.times
        sfreq = meeg.info["sfreq"]
        time_dim = "times"
        base_dims = ["spaces"]
        base_coords = {"spaces": meeg.ch_names}
    elif isinstance(meeg, mne.BaseEpochs):
        data_values = meeg.get_data()
        times = meeg.times
        sfreq = meeg.info["sfreq"]
        time_dim = "times"
        base_dims = ["epochs", "spaces"]
        base_coords = {
            "epochs": list(range(len(meeg))),
            "spaces": meeg.ch_names,
        }

    # times is the last dimension
    # data_values shape is (..., time)
    if data_values.shape[-1] != len(times):
        raise ValueError("Data last dimension must be time")

    psd_func = {
        "welch": mne.time_frequency.psd_array_welch,
        "multitaper": mne.time_frequency.psd_array_multitaper,
    }[method]

    psd_result = psd_func(data_values, sfreq=sfreq, **method_kwargs)
    weights = None
    if method == "multitaper" and isinstance(psd_result, tuple) and len(psd_result) == 3:
        psds, freqs, weights = psd_result
    else:
        psds, freqs = psd_result  # type: ignore[misc]

    sample_dims = list(base_dims)
    psd_dims = list(sample_dims)
    psd_coords = dict(base_coords)
    dimension_origins = dict.fromkeys(sample_dims, "input")

    dimension_details: list[dict[str, Any]] = []
    for idx, dim in enumerate(sample_dims):
        dimension_details.append(
            {
                "name": dim,
                "origin": "input",
                "size": int(psds.shape[idx]),
            }
        )

    additional_axes: list[str] = []
    average = None
    output_mode = None

    if method == "welch":
        average = method_kwargs.get("average", "mean")
        freq_axis = len(sample_dims)
        psd_dims.append("frequencies")
        psd_coords["frequencies"] = np.asarray(freqs)
        dimension_origins["frequencies"] = "frequency"
        dimension_details.append(
            {
                "name": "frequencies",
                "origin": "welch_frequency",
                "size": int(psds.shape[freq_axis]),
            }
        )
        additional_axes.append("frequencies")
        if average is None:
            seg_axis = freq_axis + 1
            psd_dims.append("segments")
            psd_coords["segments"] = np.arange(psds.shape[seg_axis])
            dimension_origins["segments"] = "welch_segments"
            dimension_details.append(
                {
                    "name": "segments",
                    "origin": "welch_segments",
                    "size": int(psds.shape[seg_axis]),
                }
            )
            additional_axes.append("segments")
    elif method == "multitaper":
        output_mode = method_kwargs.get("output", "power")
        if output_mode == "complex":
            taper_axis = len(sample_dims)
            psd_dims.append("tapers")
            psd_coords["tapers"] = np.arange(psds.shape[taper_axis])
            dimension_origins["tapers"] = "multitaper_tapers"
            dimension_details.append(
                {
                    "name": "tapers",
                    "origin": "multitaper_tapers",
                    "size": int(psds.shape[taper_axis]),
                }
            )
            additional_axes.append("tapers")
            freq_axis = taper_axis + 1
        else:
            freq_axis = len(sample_dims)
        psd_dims.append("frequencies")
        psd_coords["frequencies"] = np.asarray(freqs)
        dimension_origins["frequencies"] = "frequency"
        dimension_details.append(
            {
                "name": "frequencies",
                "origin": "multitaper_frequency",
                "size": int(psds.shape[freq_axis]),
            }
        )
        additional_axes.append("frequencies")

    dimension_notes = {
        "time_dim": time_dim,
        "replaced_with": "frequencies",
        "additional_axes": additional_axes,
    }

    metadata: dict[str, Any] = {
        "method": method,
        "method_kwargs": _json_safe(method_kwargs),
        "sampling_frequency": sfreq,
        "input": {
            "shape": [int(v) for v in data_values.shape],
            "dims": [*sample_dims, time_dim],
        },
        "output": {
            "shape": [int(v) for v in psds.shape],
            "dims": psd_dims,
            "dimension_details": dimension_details,
        },
        "dimension_notes": dimension_notes,
    }

    if method == "welch":
        metadata["output"]["average"] = _json_safe(average)
    if method == "multitaper":
        metadata["output"]["output_parameter"] = output_mode or "power"

    weights_xarray: xr.DataArray | None = None
    if weights is not None:
        weights_array = np.asarray(weights)
        weights_dims = list(sample_dims)
        weights_coords = dict(base_coords)
        weights_dims.append("tapers")
        weights_coords["tapers"] = np.arange(weights_array.shape[-1])
        weights_xarray = xr.DataArray(weights_array, dims=weights_dims, coords=weights_coords)
        weights_metadata = {
            "method": method,
            "description": "DPSS weights returned by psd_array_multitaper",
            "shape": [int(v) for v in weights_array.shape],
            "dims": weights_dims,
        }
        weights_xarray.attrs["metadata"] = json.dumps(
            weights_metadata,
            indent=2,
            default=_json_safe,
        )
        metadata["weights_shape"] = [int(v) for v in weights_array.shape]

    psd_xarray = xr.DataArray(data=psds, dims=psd_dims, coords=psd_coords)
    metadata_json = json.dumps(metadata, indent=2, default=_json_safe)
    psd_xarray.attrs["metadata"] = metadata_json

    dataset_vars: dict[str, xr.DataArray] = {"spectrum": psd_xarray}
    if weights_xarray is not None:
        dataset_vars["weights"] = weights_xarray

    psd_dataset = xr.Dataset(data_vars=dataset_vars)
    psd_dataset.attrs["metadata"] = metadata_json

    artifacts: dict[str, Artifact] = {
        ".nc": Artifact(
            item=psd_dataset,
            writer=lambda path: psd_dataset.to_netcdf(path, engine="netcdf4", format="NETCDF4"),
        ),
    }

    return NodeResult(artifacts=artifacts)


@register_node
def fooof(
    psd_like: NodeResult | xr.DataArray | str | os.PathLike[str],
    *,
    freq_dim: str = "frequencies",
    freqs: Sequence[float] | np.ndarray | None = None,
    fooof_options: Mapping[str, Any] | None = None,
    allow_eval_strings: bool = True,
    failure_value: str | None = "{}",
    include_timings: bool = True,
) -> NodeResult:
    """Fit FOOOF models for every non-frequency slice in a PSD ``xarray`` artifact.

    Parameters
    ----------
    psd_like : NodeResult | xarray.DataArray | path-like
        Output from ``spectrum``/``spectrum_array`` or a compatible ``xarray`` artifact.
    freq_dim : str, optional
        Name of the frequency dimension. Defaults to ``"frequencies"``.
    freqs : sequence of float, optional
        Explicit frequency values. If omitted they are read from the coordinate
        of ``freq_dim``.
    fooof_options : mapping, optional
        Nested configuration dictionary using the legacy layout
        ``{"FOOOF": {...}, "fit": {...}, "save": {...}, "freq_res": ...}``.
    allow_eval_strings : bool, optional
        Interpret values that start with ``"eval%"`` using ``eval`` with
        ``numpy``/``math`` in scope. Mirrors the historic behaviour.
    failure_value : str | None, optional
        Fallback string stored when a FOOOF fit fails. Defaults to ``"{}"``.
    include_timings : bool, optional
        Whether to output a timings artifact (seconds per fit).

    Returns
    -------
    NodeResult
        ``.nc`` artifact containing an ``xarray.Dataset`` with the FOOOF
        payloads stored under the ``fooof`` variable and, when requested,
        timings under ``timings``.
    """

    psd_xr = _resolve_psd_dataarray(psd_like)

    if freq_dim not in psd_xr.dims:
        raise ValueError(
            f"Frequency dimension '{freq_dim}' not present in input dims: {psd_xr.dims}"
        )

    if freqs is None:
        coord = psd_xr.coords.get(freq_dim)
        if coord is None:
            raise ValueError(
                "Frequency dimension must provide coordinates when 'freqs' is not supplied."
            )
        freq_values = np.asarray(coord.values, dtype=float)
    else:
        freq_values = np.asarray(freqs, dtype=float)

    if freq_values.ndim != 1:
        raise ValueError("Frequency information must be one-dimensional.")

    n_freqs = freq_values.size
    if n_freqs == 0:
        raise ValueError("Frequency array must contain at least one value.")

    options = copy.deepcopy(dict(fooof_options or {}))
    if allow_eval_strings:
        options = _resolve_eval_strings(options)

    fooof_init_kwargs = dict(options.pop("FOOOF", {}))
    fit_kwargs = dict(options.pop("fit", {}))
    save_kwargs = dict(options.pop("save", {}))
    freq_res = options.pop("freq_res", None)
    unused_options = options  # Whatever remains is captured for metadata purposes.

    save_defaults = {"save_results": True, "save_settings": True, "save_data": False}
    for key, value in save_defaults.items():
        save_kwargs.setdefault(key, value)

    if freq_res is not None:
        freq_res = float(freq_res)
        if freq_res <= 0:
            raise ValueError("freq_res must be a positive float if provided.")

    if n_freqs > 1:
        diffs = np.diff(freq_values)
        valid_diffs = diffs[np.nonzero(diffs)]
        current_res = float(np.median(np.abs(valid_diffs))) if valid_diffs.size else float("nan")
    else:
        current_res = float("nan")

    downsample_step = 1
    if freq_res is not None and n_freqs > 1 and np.isfinite(current_res) and current_res > 0:
        if freq_res < current_res:
            log.warning(
                "Requested freq_res is finer than available resolution; skipping downsampling",
                requested=freq_res,
                available=current_res,
            )
        else:
            downsample_step = max(1, math.ceil(freq_res / current_res))

    freq_values_downsampled = freq_values[::downsample_step]
    if freq_values_downsampled.size == 0:
        raise ValueError("Downsampling removed all frequency points; check freq_res setting.")

    other_dims = [dim for dim in psd_xr.dims if dim != freq_dim]
    transposed = psd_xr.transpose(*([*other_dims, freq_dim]))
    psd_values = np.asarray(transposed.values)
    if psd_values.ndim == 1:
        psd_values = psd_values[np.newaxis, :]

    other_shape = [int(transposed.sizes[dim]) for dim in other_dims]
    flattened = psd_values.reshape(-1, n_freqs)

    fooof_payloads = np.empty(flattened.shape[0], dtype=object)
    timings = np.full(flattened.shape[0], np.nan, dtype=float)
    failure_records: list[dict[str, Any]] = []

    coords_cache: dict[str, np.ndarray] = {}
    coords_for_output: dict[str, np.ndarray] = {}
    for dim in other_dims:
        if dim in transposed.coords:
            values = np.asarray(transposed.coords[dim].values)
        else:
            values = np.arange(transposed.sizes[dim])
        coords_cache[dim] = values
        coords_for_output[dim] = values

    fallback_value = "" if failure_value is None else str(failure_value)

    for flat_idx in range(flattened.shape[0]):
        if other_dims:
            unravel = np.unravel_index(flat_idx, tuple(other_shape))
            coord_mapping = {
                dim: _json_safe(coords_cache[dim][unravel[idx]])
                for idx, dim in enumerate(other_dims)
            }
        else:
            unravel = ()
            coord_mapping = {}

        start = time.perf_counter()
        try:
            signal = np.asarray(flattened[flat_idx])
            if signal.size != n_freqs:
                raise ValueError(
                    "Each PSD slice must have the same number of frequencies as the coordinate array."
                )

            signal_to_fit = signal[::downsample_step]
            if signal_to_fit.size != freq_values_downsampled.size:
                raise ValueError(
                    "Downsampled signal and frequency vectors must be the same length."
                )

            fm = FOOOF(verbose=False, **fooof_init_kwargs)
            fm.fit(freq_values_downsampled, signal_to_fit, **fit_kwargs)

            buffer = io.StringIO()
            fm.save(
                buffer,
                file_path=None,
                append=False,
                save_results=save_kwargs.get("save_results", True),
                save_settings=save_kwargs.get("save_settings", True),
                save_data=save_kwargs.get("save_data", False),
            )
            payload = buffer.getvalue() or json.dumps({}, indent=2)
            fooof_payloads[flat_idx] = payload
        except Exception as exc:
            duration = time.perf_counter() - start
            timings[flat_idx] = duration
            fooof_payloads[flat_idx] = fallback_value
            failure_records.append(
                {
                    "index": int(flat_idx),
                    "coords": coord_mapping,
                    "error": repr(exc),
                }
            )
            log.warning("FOOOF fit failed", coords=coord_mapping, error=str(exc))
            continue

        timings[flat_idx] = time.perf_counter() - start

    if other_dims:
        result_shape = tuple(other_shape)
        coords = coords_for_output
    else:
        result_shape = ()
        coords = {}

    result_array = fooof_payloads.reshape(result_shape)
    fooof_xr = xr.DataArray(result_array, dims=other_dims, coords=coords, name="fooof")

    metadata: dict[str, Any] = {
        "freq_dim": freq_dim,
        "frequencies": _json_safe(freq_values),
        "frequencies_downsampled": _json_safe(freq_values_downsampled),
        "downsample_step": int(downsample_step),
        "fooof_kwargs": _json_safe(fooof_init_kwargs),
        "fit_kwargs": _json_safe(fit_kwargs),
        "save_kwargs": _json_safe(save_kwargs),
        "unused_options": _json_safe(unused_options) if unused_options else None,
        "allow_eval_strings": allow_eval_strings,
        "failure_value": fallback_value,
        "input_dims": list(psd_xr.dims),
        "input_shape": [int(psd_xr.sizes[dim]) for dim in psd_xr.dims],
        "output_dims": list(fooof_xr.dims),
        "output_shape": [int(fooof_xr.sizes[dim]) for dim in fooof_xr.dims],
        "failures": failure_records,
        "frequency_resolution": {
            "requested": freq_res,
            "available": current_res,
        },
    }

    source_metadata = psd_xr.attrs.get("metadata")
    if source_metadata is not None:
        metadata["source_metadata"] = source_metadata

    metadata_json = json.dumps(metadata, indent=2, default=_json_safe)
    fooof_xr.attrs["metadata"] = metadata_json

    dataset_vars: dict[str, xr.DataArray] = {"fooof": fooof_xr}

    if include_timings:
        timings_array = timings.reshape(result_shape)
        fooof_timings = xr.DataArray(
            timings_array,
            dims=other_dims,
            coords=coords,
            name="timings",
        )
        timing_metadata = {
            "description": "Wall-clock duration per FOOOF fit",
            "unit": "seconds",
        }
        fooof_timings.attrs["metadata"] = json.dumps(timing_metadata, indent=2, default=_json_safe)
        dataset_vars["timings"] = fooof_timings

    fooof_dataset = xr.Dataset(data_vars=dataset_vars)
    fooof_dataset.attrs["metadata"] = metadata_json

    artifacts: dict[str, Artifact] = {
        ".nc": Artifact(
            item=fooof_dataset,
            writer=lambda path: fooof_dataset.to_netcdf(path, engine="netcdf4", format="NETCDF4"),
        )
    }

    return NodeResult(artifacts=artifacts)


@register_node
def fooof_scalars(
    fooof_like: NodeResult | xr.DataArray | str | os.PathLike[str],
    *,
    component: Literal["aperiodic_params", "r_squared", "error", "all"] = "all",
    freq_dim: str = "frequencies",
) -> NodeResult:
    """Extract scalar outputs from serialized FOOOF models.

    Parameters
    ----------
    fooof_like : NodeResult | xarray.DataArray | path-like
        Artifact generated by :func:`fooof`, containing JSON strings per slice.
    component : {"aperiodic_params", "r_squared", "error", "all"}, optional
        Which FOOOF scalar to extract. ``aperiodic_params`` returns offset/(knee)/exponent
        per slice, ``r_squared`` and ``error`` provide the fit metrics, and ``all`` outputs
        the full table as a single artifact.
    freq_dim : str, optional
        Name of the frequency dimension in the original PSD. Used to preserve dimension order
        when aligning with other outputs. Defaults to ``"frequencies"``.

    Returns
    -------
    NodeResult
        ``.nc`` artifact(s) containing the requested scalar(s).
    """

    component = component.lower()
    valid_components = {"aperiodic_params", "r_squared", "error", "all"}
    if component not in valid_components:
        raise ValueError(
            "component must be one of {'aperiodic_params', 'r_squared', 'error', 'all'}"
        )

    fooof_xr = _resolve_fooof_dataarray(fooof_like)
    other_dims = list(fooof_xr.dims)
    other_shape = [int(fooof_xr.sizes[dim]) for dim in other_dims]
    flat_count = int(np.prod(other_shape)) if other_shape else 1

    fooof_flat = np.asarray(fooof_xr.values, dtype=object).reshape(flat_count)
    fooof_meta_raw = fooof_xr.attrs.get("metadata")
    try:
        fooof_meta = json.loads(fooof_meta_raw) if fooof_meta_raw else {}
    except json.JSONDecodeError:
        fooof_meta = {}

    loaded_models: list[FOOOF | None] = [None] * flat_count
    invalid_indices: list[int] = []

    for idx, payload in enumerate(fooof_flat):
        if not isinstance(payload, str) or not payload.strip():
            invalid_indices.append(idx)
            continue

        try:
            fm = FOOOF()
            fm.load(io.StringIO(payload))
            if not fm.has_model:
                raise ValueError("FOOOF object missing model")
        except Exception as exc:
            log.warning("Failed to load FOOOF payload for scalars", index=idx, error=str(exc))
            invalid_indices.append(idx)
            continue

        loaded_models[idx] = fm

    scalar_arrays: dict[str, np.ndarray] = {
        "aperiodic_params": np.empty((flat_count,), dtype=object),
        "r_squared": np.full((flat_count,), np.nan, dtype=float),
        "error": np.full((flat_count,), np.nan, dtype=float),
        "aperiodic_offset": np.empty((flat_count,), dtype=float),
        "aperiodic_knee": np.empty((flat_count,), dtype=float),
        "aperiodic_exponent": np.empty((flat_count,), dtype=float),
    }

    for idx, fm in enumerate(loaded_models):
        if fm is None:
            scalar_arrays["aperiodic_params"][idx] = None
            continue

        try:
            ap_params = getattr(fm, "aperiodic_params_", getattr(fm, "_aperiodic_params", None))
            if ap_params is None:
                raise ValueError("Missing aperiodic parameters")
            ap_params = np.asarray(ap_params, dtype=float)
            ap_mode = getattr(fm, "aperiodic_mode", getattr(fm, "aperiodic_mode_", "fixed"))
            if ap_mode == "knee" and ap_params.size < 3:
                raise ValueError("Knee mode expects three parameters")
            if ap_mode != "knee" and ap_params.size < 2:
                raise ValueError("Fixed mode expects two parameters")
            scalar_arrays["aperiodic_params"][idx] = ap_params.tolist()

            if ap_params.size == 2:
                scalar_arrays["aperiodic_offset"][idx] = float(ap_params[0])
                scalar_arrays["aperiodic_knee"][idx] = np.nan
                scalar_arrays["aperiodic_exponent"][idx] = float(ap_params[1])
            else:
                scalar_arrays["aperiodic_offset"][idx] = float(ap_params[0])
                scalar_arrays["aperiodic_knee"][idx] = float(ap_params[1])
                scalar_arrays["aperiodic_exponent"][idx] = float(ap_params[2])

            scalar_arrays["r_squared"][idx] = float(
                getattr(fm, "r_squared_", getattr(fm, "r_squared", np.nan))
            )
            scalar_arrays["error"][idx] = float(getattr(fm, "error_", getattr(fm, "error", np.nan)))
        except Exception as exc:
            log.warning("Failed to compute FOOOF scalar", index=idx, error=str(exc))
            invalid_indices.append(idx)
            scalar_arrays["aperiodic_params"][idx] = None
            scalar_arrays["r_squared"][idx] = np.nan
            scalar_arrays["error"][idx] = np.nan

    coords = {
        dim: (
            fooof_xr.coords[dim].values
            if dim in fooof_xr.coords
            else np.arange(fooof_xr.sizes[dim])
        )
        for dim in other_dims
    }

    name_map = {
        "aperiodic_offset": "fooof_aperiodic_offset",
        "aperiodic_knee": "fooof_aperiodic_knee",
        "aperiodic_exponent": "fooof_aperiodic_exponent",
        "r_squared": "fooof_r_squared",
        "error": "fooof_error",
    }

    selection_map: dict[str, tuple[str, ...]] = {
        "aperiodic_params": ("aperiodic_offset", "aperiodic_knee", "aperiodic_exponent"),
        "r_squared": ("r_squared",),
        "error": ("error",),
        "all": tuple(name_map.keys()),
    }

    metadata_base: dict[str, Any] = {
        "components": [name_map[key] for key in name_map],
        "invalid_count": len(set(invalid_indices)),
        "total_count": flat_count,
        "invalid_indices": sorted(set(invalid_indices)),
    }
    if fooof_meta:
        metadata_base["fooof_metadata"] = fooof_meta

    def make_array(key: str) -> xr.DataArray:
        data = scalar_arrays[key].reshape(other_shape)
        xarr = xr.DataArray(data, dims=other_dims, coords=coords, name=name_map[key])
        this_meta = dict(metadata_base)
        this_meta["component"] = key
        this_meta["variable"] = name_map[key]
        xarr.attrs["metadata"] = json.dumps(this_meta, indent=2, default=_json_safe)
        return xarr

    selected_keys = selection_map[component]
    dataset_vars = [make_array(key) for key in selected_keys]
    dataset = xr.Dataset({var.name: var for var in dataset_vars})

    dataset_meta = dict(metadata_base)
    dataset_meta["component"] = component
    dataset_meta["variables"] = [var.name for var in dataset_vars]
    dataset.attrs["metadata"] = json.dumps(dataset_meta, indent=2, default=_json_safe)

    artifact = Artifact(
        item=dataset,
        writer=lambda path, ds=dataset: ds.to_netcdf(path, engine="netcdf4", format="NETCDF4"),
    )

    return NodeResult(artifacts={".nc": artifact})


@register_node
def fooof_component(
    fooof_like: NodeResult | xr.DataArray | str | os.PathLike[str],
    *,
    component: Literal["aperiodic", "periodic", "residual", "all"] = "aperiodic",
    freq_dim: str = "frequencies",
    mode: Literal["fooof-api", "manual"] = "fooof-api",
    space: Literal["log", "linear"] = "linear",
) -> NodeResult:
    """Derive spectral components directly from serialized FOOOF models.

    Parameters
    ----------
    fooof_like : NodeResult | xarray.DataArray | path-like
        Artifact generated by :func:`fooof`, containing JSON strings per slice.
    component : {"aperiodic", "periodic", "residual", "all"}, optional
        ``aperiodic`` returns the FOOOF background spectrum, ``periodic`` returns
        the modelled oscillatory power (Gaussians minus background), ``residual``
        subtracts the background from the original power spectrum, and ``all``
        emits all three components as separate artifacts.
    freq_dim : str, optional
        Name of the frequency dimension for the output. Defaults to ``"frequencies"``.
    mode : {"fooof-api", "manual"}, optional
        Strategy used to recover the components. ``"fooof-api"`` leverages
        :meth:`FOOOF.get_data` for the aperiodic and data spectrum while deriving
        the FOOOF model from stored fits (the periodic output matches FOOOF's
        peak component in the selected space). ``"manual"`` reproduces the
        previous logic based on stored log-domain fits.
    space : {"log", "linear"}, optional
        Output space for the recovered components. Defaults to ``"linear"``.

    Returns
    -------
    NodeResult
        ``.nc`` artifact containing an ``xarray.Dataset`` with the requested
        component(s) in the specified space.
    """

    component = component.lower()
    valid_components = {"aperiodic", "periodic", "residual", "all"}
    if component not in valid_components:
        raise ValueError("component must be one of {'aperiodic', 'periodic', 'residual', 'all'}")

    mode = mode.lower()
    if mode not in {"fooof-api", "manual"}:
        raise ValueError("mode must be either 'fooof-api' or 'manual'")

    space = space.lower()
    if space not in {"log", "linear"}:
        raise ValueError("space must be either 'log' or 'linear'")

    fooof_xr = _resolve_fooof_dataarray(fooof_like)
    other_dims = list(fooof_xr.dims)
    other_shape = [int(fooof_xr.sizes[dim]) for dim in other_dims]
    flat_count = int(np.prod(other_shape)) if other_shape else 1

    fooof_flat = np.asarray(fooof_xr.values, dtype=object).reshape(flat_count)

    fooof_meta_raw = fooof_xr.attrs.get("metadata")
    try:
        fooof_meta = json.loads(fooof_meta_raw) if fooof_meta_raw else {}
    except json.JSONDecodeError:
        fooof_meta = {}

    loaded_models: list[FOOOF | None] = [None] * flat_count
    invalid_indices: list[int] = []
    freq_values_model: np.ndarray | None = None

    for idx, payload in enumerate(fooof_flat):
        if not isinstance(payload, str) or not payload.strip():
            invalid_indices.append(idx)
            continue

        try:
            fm = FOOOF()
            fm.load(io.StringIO(payload))
            if not fm.has_model:
                raise ValueError("FOOOF object missing model")
        except Exception as exc:
            log.warning("Failed to load FOOOF payload", index=idx, error=str(exc))
            invalid_indices.append(idx)
            continue

        freqs = np.asarray(getattr(fm, "freqs", None))
        if freqs.size == 0:
            invalid_indices.append(idx)
            continue

        if freq_values_model is None:
            freq_values_model = freqs.astype(float, copy=True)
        elif freqs.shape != freq_values_model.shape or not np.allclose(freqs, freq_values_model):
            log.warning("FOOOF frequencies mismatch; marking slice invalid", index=idx)
            invalid_indices.append(idx)
            continue

        loaded_models[idx] = fm

    if freq_values_model is None or freq_values_model.size == 0:
        raise ValueError("No valid FOOOF models with frequency information were found.")

    freq_len = freq_values_model.size

    def _ensure_shape(values: np.ndarray, label: str) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        if arr.size != freq_len:
            raise ValueError(f"{label} length mismatch")
        return arr

    def _log_to_space(values: np.ndarray, label: str) -> np.ndarray:
        arr = _ensure_shape(values, label)
        return arr if space == "log" else np.power(10.0, arr)

    def _model_to_space(values: np.ndarray, label: str) -> np.ndarray:
        """Convert modeled log spectra to requested space."""
        return _log_to_space(values, label)

    component_arrays: dict[str, np.ndarray] = {
        "aperiodic": np.full((flat_count, freq_len), np.nan, dtype=float),
        "periodic": np.full((flat_count, freq_len), np.nan, dtype=float),
        "residual": np.full((flat_count, freq_len), np.nan, dtype=float),
    }

    for idx, fm in enumerate(loaded_models):
        if fm is None:
            continue

        try:
            need_residual = component in {"residual", "all"}
            need_periodic = component in {"periodic", "all"}
            need_model = need_periodic or need_residual

            if mode == "fooof-api":
                aperiodic = _ensure_shape(
                    fm.get_data("aperiodic", space=space), "aperiodic spectrum"
                )
                if need_periodic:
                    periodic = _ensure_shape(fm.get_data("peak", space=space), "peak spectrum")
                if need_residual:
                    full_in_space = _ensure_shape(fm.get_data("full", space=space), "full spectrum")
                    model_in_space = _ensure_shape(
                        fm.get_data("model", space=space), "model spectrum"
                    )
                    residual = full_in_space - model_in_space
            else:
                ap_fit_log = _ensure_shape(getattr(fm, "_ap_fit", None), "aperiodic fit")
                aperiodic = _log_to_space(ap_fit_log, "aperiodic fit")

                if need_model:
                    model_log = _ensure_shape(
                        getattr(fm, "fooofed_spectrum_", None), "fooofed spectrum"
                    )
                    model_in_space = _model_to_space(model_log, "fooofed spectrum")

                if need_periodic:
                    peak_log = getattr(fm, "_peak_fit", None)
                    if peak_log is not None:
                        peak_log = _ensure_shape(peak_log, "peak fit")
                    else:
                        peak_log = model_log - ap_fit_log
                    if space == "linear":
                        periodic = model_in_space - aperiodic
                    else:
                        periodic = peak_log

                if need_residual:
                    power_log = _ensure_shape(getattr(fm, "power_spectrum", None), "power spectrum")
                    full_in_space = _model_to_space(power_log, "power spectrum")
                    residual = full_in_space - model_in_space

            component_arrays["aperiodic"][idx] = aperiodic
            if need_periodic:
                component_arrays["periodic"][idx] = periodic
            if need_residual:
                component_arrays["residual"][idx] = residual
        except Exception as exc:
            log.warning("Failed to compute FOOOF component", index=idx, error=str(exc))
            invalid_indices.append(idx)
            for key in ("aperiodic", "periodic", "residual"):
                component_arrays[key][idx] = np.nan

    output_shape = (*other_shape, freq_len) if other_dims else (freq_len,)
    output_dims = [*other_dims, freq_dim]

    coords: dict[str, Any] = {}
    for dim in other_dims:
        coord = fooof_xr.coords.get(dim)
        coords[dim] = coord.values if coord is not None else np.arange(fooof_xr.sizes[dim])
    coords[freq_dim] = freq_values_model

    # name_map = {
    #     "aperiodic": f"fooof_aperiodic_{suffix}",
    #     "periodic": f"fooof_periodic_{suffix}",
    #     "residual": f"fooof_residual_{suffix}",
    # }
    name_map = {
        "aperiodic": "aperiodic",
        "periodic": "periodic",
        "residual": "residual",
    }

    selection_map: dict[str, tuple[str, ...]] = {
        "aperiodic": ("aperiodic",),
        "periodic": ("periodic",),
        "residual": ("residual",),
        "all": tuple(name_map.keys()),
    }

    def build_component(key: str) -> xr.DataArray:
        data = component_arrays[key].reshape(output_shape)
        return xr.DataArray(data, dims=output_dims, coords=coords, name=name_map[key])

    metadata_base: dict[str, Any] = {
        "frequency_dimension": freq_dim,
        "frequencies": _json_safe(freq_values_model.tolist()),
        "invalid_count": len(set(invalid_indices)),
        "total_count": flat_count,
        "invalid_indices": sorted(set(invalid_indices)),
        "components": [name_map[k] for k in name_map],
        "mode": mode,
        "space": space,
    }
    if fooof_meta:
        metadata_base["fooof_metadata"] = fooof_meta

    def attach_metadata(xarr: xr.DataArray, key: str) -> xr.DataArray:
        this_meta = dict(metadata_base)
        this_meta["component"] = key
        xarr.attrs["metadata"] = json.dumps(this_meta, indent=2, default=_json_safe)
        return xarr

    selected_keys = selection_map[component]
    dataset_vars = [attach_metadata(build_component(key), key) for key in selected_keys]
    dataset = xr.Dataset({var.name: var for var in dataset_vars})

    dataset_meta = dict(metadata_base)
    dataset_meta["component"] = component
    dataset_meta["variables"] = [var.name for var in dataset_vars]
    dataset.attrs["metadata"] = json.dumps(dataset_meta, indent=2, default=_json_safe)

    artifact = Artifact(
        item=dataset,
        writer=lambda path, ds=dataset: ds.to_netcdf(path, engine="netcdf4", format="NETCDF4"),
    )

    return NodeResult(artifacts={".nc": artifact})


@register_node
def bandpower(
    psd_like: NodeResult | xr.DataArray | str | os.PathLike[str],
    *,
    bands: Mapping[str, tuple[float, float]] | None = None,
    freq_dim: str = "frequencies",
    relative: bool = False,
    log_transform: bool = False,
) -> NodeResult:
    """Compute absolute or relative band power from a PSD ``xarray.DataArray``.

    Parameters
    ----------
    psd_like : NodeResult | xarray.DataArray | path-like
        Output from ``spectrum``/``spectrum_array`` or a compatible ``xarray`` artifact.
    bands : mapping, optional
        Frequency bands as ``{"label": (low, high)}``. If omitted ``DEFAULT_BANDS`` is used.
    freq_dim : str, optional
        Name of the frequency dimension in the PSD. Defaults to ``"frequencies"``.
    relative : bool, optional
        If ``True`` each band power is normalised by the total power across ``freq_dim``.
    log_transform : bool, optional
        If ``True`` apply ``log10`` to the band power values after integration.

    Returns
    -------
    NodeResult
        ``.nc`` artifact whose ``freq_dim`` is replaced by ``freqbands`` containing band powers.
    """

    psd_xr = _resolve_psd_dataarray(psd_like)

    if freq_dim not in psd_xr.dims:
        raise ValueError(
            f"Frequency dimension '{freq_dim}' not present in input dims: {psd_xr.dims}"
        )

    bands_dict = dict(bands or DEFAULT_BANDS)
    if not bands_dict:
        raise ValueError("At least one frequency band must be provided.")

    freqs = psd_xr.coords.get(freq_dim)
    if freqs is None:
        raise ValueError(f"Frequency dimension '{freq_dim}' must have coordinate values.")

    if freqs.ndim != 1:
        raise ValueError("Frequency coordinate must be one-dimensional.")

    freq_axis = psd_xr.get_axis_num(freq_dim)

    total_power = psd_xr.integrate(freq_dim)

    band_arrays: list[xr.DataArray] = []
    band_edges: list[tuple[float, float]] = []

    for label, band_range in bands_dict.items():
        if len(band_range) != 2:
            raise ValueError(f"Band '{label}' must be a (low, high) pair.")

        low, high = map(float, band_range)
        if not np.isfinite(low) or not np.isfinite(
            high
        ):  # TODO: maybe we should allow inf? (as get everything below or above a threshold)
            raise ValueError(f"Band '{label}' has non-finite boundaries: {band_range}.")
        if high <= low:
            raise ValueError(f"Band '{label}' must have high > low (got {band_range}).")

        band_slice = psd_xr.sel({freq_dim: slice(low, high)})
        if band_slice.sizes.get(freq_dim, 0) == 0:
            band_power = xr.full_like(total_power, np.nan)
        else:
            band_power = band_slice.integrate(freq_dim)

        # Insert the new freqbands dimension where the original frequencies lived
        band_power = band_power.expand_dims({"freqbands": [label]}, axis=freq_axis)
        band_arrays.append(band_power)
        band_edges.append((low, high))

    band_power_xr = xr.concat(band_arrays, dim="freqbands")

    # Restore dimension order so freqbands replaces the frequency dimension position
    original_dims = list(psd_xr.dims)
    target_dims = ["freqbands" if dim == freq_dim else dim for dim in original_dims]
    band_power_xr = band_power_xr.transpose(*target_dims)

    band_power_xr = band_power_xr.assign_coords(
        freqbands=list(bands_dict),
    )
    band_power_xr.coords["freqband_low"] = ("freqbands", [edge[0] for edge in band_edges])
    band_power_xr.coords["freqband_high"] = ("freqbands", [edge[1] for edge in band_edges])

    if relative:
        denom = total_power
        with np.errstate(divide="ignore", invalid="ignore"):
            normalised = band_power_xr / denom
        band_power_xr = xr.where(denom == 0, np.nan, normalised)

    if log_transform:
        with np.errstate(divide="ignore", invalid="ignore"):
            band_power_xr = np.log10(band_power_xr)

    metadata: dict[str, Any] = {
        "bands": {
            label: {"low": float(low), "high": float(high)}
            for label, (low, high) in bands_dict.items()
        },
        "relative": relative,
        "log_transform": log_transform,
        "freq_dim": freq_dim,
        "input_dims": list(psd_xr.dims),
        "input_shape": [int(psd_xr.sizes[dim]) for dim in psd_xr.dims],
        "output_dims": list(band_power_xr.dims),
        "output_shape": [int(band_power_xr.sizes[dim]) for dim in band_power_xr.dims],
        "integration": "xarray.DataArray.integrate (trapezoidal)",
    }

    source_metadata = psd_xr.attrs.get("metadata")
    if source_metadata is not None:
        metadata["source_metadata"] = source_metadata

    band_power_xr.attrs["metadata"] = json.dumps(metadata, indent=2, default=_json_safe)

    artifacts = {
        ".nc": Artifact(item=band_power_xr, writer=lambda path: band_power_xr.to_netcdf(path)),
    }

    return NodeResult(artifacts=artifacts)


@register_node
def band_ratios(
    bandpower_like: NodeResult | xr.DataArray | str | os.PathLike[str],
    *,
    freqband_dim: str = "freqbands",
    combinations: Sequence[tuple[str, str]] | None = None,
    eps: float | None = None,
) -> NodeResult:
    """Compute ordered band power ratios from an ``xarray`` bandpower artifact.

    Parameters
    ----------
    bandpower_like : NodeResult | xarray.DataArray | path-like
        Output from :func:`bandpower` or a compatible ``xarray`` artifact that
        exposes a ``freqbands`` dimension.
    freqband_dim : str, optional
        Name of the dimension containing band labels. Defaults to ``"freqbands"``.
    combinations : sequence of tuple[str, str], optional
        Explicit ordered band pairs ``(numerator, denominator)``. If omitted,
        all permutations of length 2 across the available band labels are used.
    eps : float, optional
        Minimum absolute denominator value. Values with ``|denominator| <= eps``
        yield ``NaN`` to avoid unstable ratios. Defaults to machine epsilon for
        the bandpower dtype.

    Returns
    -------
    NodeResult
        ``.nc`` artifact with ``freqband_dim`` replaced by ``freqbandPairs``.
    """

    band_da = _resolve_psd_dataarray(bandpower_like)

    if freqband_dim not in band_da.dims:
        raise ValueError(
            f"Frequency band dimension '{freqband_dim}' not present in input dims: {band_da.dims}"
        )

    labels = band_da.coords.get(freqband_dim)
    if labels is None:
        raise ValueError(f"Band dimension '{freqband_dim}' must have coordinate labels.")

    band_labels = [str(label) for label in labels.values.tolist()]
    if combinations is None:
        band_pairs = list(permutations(band_labels, 2))
    else:
        band_pairs = [(str(top), str(bottom)) for top, bottom in combinations]

    if not band_pairs:
        raise ValueError("At least one band pair must be provided.")

    freqband_axis = band_da.get_axis_num(freqband_dim)

    if eps is None:
        try:
            eps = float(np.finfo(np.asarray(0.0, dtype=band_da.dtype).dtype).eps)
        except (TypeError, ValueError):
            eps = float(np.finfo(np.float64).eps)
    else:
        eps = float(eps)

    ratio_arrays: list[xr.DataArray] = []
    tops: list[str] = []
    bottoms: list[str] = []
    pair_labels: list[str] = []

    for top, bottom in band_pairs:
        numerator = band_da.sel({freqband_dim: top})
        denominator = band_da.sel({freqband_dim: bottom})

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = numerator / denominator

        small_denom = np.abs(denominator) <= eps
        ratio = xr.where(small_denom, np.nan, ratio)

        ratio = ratio.expand_dims({"freqbandPairs": [f"{top}/{bottom}"]}, axis=freqband_axis)
        ratio_arrays.append(ratio)
        tops.append(top)
        bottoms.append(bottom)
        pair_labels.append(f"{top}/{bottom}")

    ratio_xr = xr.concat(ratio_arrays, dim="freqbandPairs")

    original_dims = list(band_da.dims)
    target_dims = ["freqbandPairs" if dim == freqband_dim else dim for dim in original_dims]
    ratio_xr = ratio_xr.transpose(*target_dims)

    ratio_xr = ratio_xr.assign_coords(freqbandPairs=pair_labels)
    ratio_xr.coords["freqband_top"] = ("freqbandPairs", tops)
    ratio_xr.coords["freqband_bottom"] = ("freqbandPairs", bottoms)

    metadata: dict[str, Any] = {
        "pairs": [
            {"label": pair_label, "top": top, "bottom": bottom}
            for pair_label, top, bottom in zip(pair_labels, tops, bottoms, strict=True)
        ],
        "freqband_dim": freqband_dim,
        "eps": eps,
        "input_dims": list(band_da.dims),
        "input_shape": [int(band_da.sizes[dim]) for dim in band_da.dims],
        "output_dims": list(ratio_xr.dims),
        "output_shape": [int(ratio_xr.sizes[dim]) for dim in ratio_xr.dims],
    }

    source_metadata = band_da.attrs.get("metadata")
    if source_metadata is not None:
        metadata["source_metadata"] = source_metadata

    ratio_xr.attrs["metadata"] = json.dumps(metadata, indent=2, default=_json_safe)

    artifacts = {
        ".nc": Artifact(item=ratio_xr, writer=lambda path: ratio_xr.to_netcdf(path)),
    }

    return NodeResult(artifacts=artifacts)


@register_node
def fooof_peaks(
    fooof_like: "NodeResult | xr.DataArray | xr.Dataset | str | os.PathLike[str]",
    *,
    alpha_band: tuple[float, float] = (8.0, 13.0),
) -> NodeResult:
    """Extract peak summary scalars from serialized FOOOF models.

    Parameters
    ----------
    fooof_like : NodeResult | xarray.DataArray | Dataset | path-like
        Artifact generated by :func:`fooof`, containing JSON strings per slice.
    alpha_band : tuple[float, float], optional
        Frequency range defining the alpha band for alpha-peak extraction.
        Defaults to ``(8.0, 13.0)`` Hz.

    Returns
    -------
    NodeResult
        ``.nc`` artifact containing an ``xarray.Dataset`` with variables:
        ``n_peaks``, ``dominant_peak_cf``, ``dominant_peak_pw``,
        ``dominant_peak_bw``, ``alpha_peak_cf``, ``alpha_peak_pw``,
        ``alpha_peak_bw``.
    """

    fooof_xr = _resolve_fooof_dataarray(fooof_like)
    dims = list(fooof_xr.dims)
    shape = [int(fooof_xr.sizes[d]) for d in dims]
    flat_count = int(np.prod(shape)) if shape else 1
    fooof_flat = np.asarray(fooof_xr.values, dtype=object).reshape(flat_count)

    n_peaks = np.zeros(flat_count, dtype=int)
    dominant_cf = np.full(flat_count, np.nan)
    dominant_pw = np.full(flat_count, np.nan)
    dominant_bw = np.full(flat_count, np.nan)
    alpha_cf = np.full(flat_count, np.nan)
    alpha_pw = np.full(flat_count, np.nan)
    alpha_bw = np.full(flat_count, np.nan)

    alpha_lo, alpha_hi = float(alpha_band[0]), float(alpha_band[1])

    for idx, payload in enumerate(fooof_flat):
        if not isinstance(payload, str) or not payload.strip():
            continue
        try:
            fm = FOOOF()
            fm.load(io.StringIO(payload))
            if not fm.has_model:
                continue
            peaks = np.asarray(getattr(fm, "peak_params_", []))
            if peaks.ndim != 2 or peaks.shape[1] != 3 or len(peaks) == 0:
                continue
            n = len(peaks)
            n_peaks[idx] = n
            dom_idx = int(np.argmax(peaks[:, 1]))
            dominant_cf[idx] = float(peaks[dom_idx, 0])
            dominant_pw[idx] = float(peaks[dom_idx, 1])
            dominant_bw[idx] = float(peaks[dom_idx, 2])
            in_alpha = (peaks[:, 0] >= alpha_lo) & (peaks[:, 0] <= alpha_hi)
            alpha_peaks = peaks[in_alpha]
            if len(alpha_peaks) > 0:
                best = int(np.argmax(alpha_peaks[:, 1]))
                alpha_cf[idx] = float(alpha_peaks[best, 0])
                alpha_pw[idx] = float(alpha_peaks[best, 1])
                alpha_bw[idx] = float(alpha_peaks[best, 2])
        except Exception as exc:
            log.warning("fooof_peaks: failed to process slice", index=idx, error=str(exc))

    coords = {
        d: fooof_xr.coords[d].values if d in fooof_xr.coords else np.arange(fooof_xr.sizes[d])
        for d in dims
    }

    def _make(arr: np.ndarray, vname: str, dtype) -> xr.DataArray:
        return xr.DataArray(arr.reshape(shape), dims=dims, coords=coords, name=vname).astype(dtype)

    dataset = xr.Dataset(
        {
            "n_peaks": _make(n_peaks, "n_peaks", int),
            "dominant_peak_cf": _make(dominant_cf, "dominant_peak_cf", float),
            "dominant_peak_pw": _make(dominant_pw, "dominant_peak_pw", float),
            "dominant_peak_bw": _make(dominant_bw, "dominant_peak_bw", float),
            "alpha_peak_cf": _make(alpha_cf, "alpha_peak_cf", float),
            "alpha_peak_pw": _make(alpha_pw, "alpha_peak_pw", float),
            "alpha_peak_bw": _make(alpha_bw, "alpha_peak_bw", float),
        }
    )

    metadata: dict[str, Any] = {
        "alpha_band": list(alpha_band),
        "dims": dims,
        "shape": shape,
        "variables": list(dataset.data_vars),
    }
    dataset.attrs["metadata"] = json.dumps(metadata, indent=2, default=_json_safe)

    artifact = Artifact(
        item=dataset,
        writer=lambda path, ds=dataset: ds.to_netcdf(path, engine="netcdf4", format="NETCDF4"),
    )
    return NodeResult(artifacts={".nc": artifact})


@register_node
def bandpower_corrected(
    psd_like: "NodeResult | xr.DataArray | str | os.PathLike[str]",
    fooof_like: "NodeResult | xr.DataArray | xr.Dataset | str | os.PathLike[str]",
    *,
    bands: "Mapping[str, tuple[float, float]] | None" = None,
    freq_dim: str = "frequencies",
    relative: bool = False,
    log_transform: bool = False,
    aperiodic_floor: float = 1e-20,
) -> NodeResult:
    """Compute aperiodic-corrected band power using fitted FOOOF models.

    Subtracts the FOOOF aperiodic component (in log10 space) from the PSD before
    integrating, yielding band power with the 1/f background removed.

    Parameters
    ----------
    psd_like : NodeResult | xarray.DataArray | path-like
        PSD artifact from ``mne_spectrum`` / ``mne_spectrum_array``.
    fooof_like : NodeResult | xarray.DataArray | Dataset | path-like
        FOOOF artifact from :func:`fooof`.
    bands : mapping, optional
        Frequency bands as ``{"label": (low, high)}``. Defaults to ``DEFAULT_BANDS``.
    freq_dim : str, optional
        Name of the frequency dimension. Defaults to ``"frequencies"``.
    relative : bool, optional
        Normalise each corrected band power by the total corrected power.
    log_transform : bool, optional
        Apply ``log10`` to corrected band power values.
    aperiodic_floor : float, optional
        Minimum PSD value before taking log (avoids log(0)).

    Returns
    -------
    NodeResult
        ``.nc`` artifact with ``freqbands`` dimension (aperiodic-corrected).
    """

    psd_xr = _resolve_psd_dataarray(psd_like)
    fooof_xr = _resolve_fooof_dataarray(fooof_like)

    if freq_dim not in psd_xr.dims:
        raise ValueError(f"Frequency dimension '{freq_dim}' not in PSD dims: {psd_xr.dims}")

    freqs = np.asarray(psd_xr.coords[freq_dim].values, dtype=float)
    bands_dict = dict(bands or DEFAULT_BANDS)
    if not bands_dict:
        raise ValueError("At least one frequency band must be provided.")

    fooof_meta_raw = fooof_xr.attrs.get("metadata", "{}")
    try:
        fooof_meta = json.loads(fooof_meta_raw)
    except json.JSONDecodeError:
        fooof_meta = {}
    aperiodic_mode = fooof_meta.get("fooof_kwargs", {}).get("aperiodic_mode", "fixed")

    fooof_dims = list(fooof_xr.dims)

    psd_transposed = psd_xr.transpose(*[*fooof_dims, freq_dim])
    psd_values = np.asarray(psd_transposed.values, dtype=float)
    other_shape = [int(psd_transposed.sizes[d]) for d in fooof_dims]
    flat_count = int(np.prod(other_shape)) if other_shape else 1

    psd_flat = psd_values.reshape(-1, len(freqs))
    fooof_flat = np.asarray(fooof_xr.values, dtype=object).reshape(flat_count)

    corrected_flat = np.full_like(psd_flat, np.nan)

    for idx in range(flat_count):
        payload = fooof_flat[idx]
        if not isinstance(payload, str) or not payload.strip():
            continue
        try:
            fm = FOOOF()
            fm.load(io.StringIO(payload))
            if not fm.has_model:
                continue
            ap_params = np.asarray(
                getattr(fm, "aperiodic_params_", getattr(fm, "_aperiodic_params", None)),
                dtype=float,
            )
            if ap_params is None or ap_params.size < 2:
                continue
            ap_log = _gen_aperiodic_log(freqs, ap_params, aperiodic_mode)
            psd_log = np.log10(np.maximum(psd_flat[idx], aperiodic_floor))
            corrected_flat[idx] = np.power(10.0, psd_log - ap_log)
        except Exception as exc:
            log.warning("bandpower_corrected: failed slice", index=idx, error=str(exc))

    corrected_shape = (*other_shape, len(freqs)) if other_shape else (len(freqs),)
    corrected_psd = corrected_flat.reshape(corrected_shape)

    coords_base = {
        d: (
            psd_transposed.coords[d].values
            if d in psd_transposed.coords
            else np.arange(psd_transposed.sizes[d])
        )
        for d in fooof_dims
    }
    coords_base[freq_dim] = freqs

    corrected_xr = xr.DataArray(
        corrected_psd,
        dims=[*fooof_dims, freq_dim],
        coords=coords_base,
    )

    freq_axis = corrected_xr.get_axis_num(freq_dim)
    total_power = corrected_xr.integrate(freq_dim)
    band_arrays: list[xr.DataArray] = []
    band_edges: list[tuple[float, float]] = []

    for label, band_range in bands_dict.items():
        low, high = float(band_range[0]), float(band_range[1])
        band_slice = corrected_xr.sel({freq_dim: slice(low, high)})
        if band_slice.sizes.get(freq_dim, 0) == 0:
            band_power = xr.full_like(total_power, np.nan)
        else:
            band_power = band_slice.integrate(freq_dim)
        band_power = band_power.expand_dims({"freqbands": [label]}, axis=freq_axis)
        band_arrays.append(band_power)
        band_edges.append((low, high))

    band_power_xr = xr.concat(band_arrays, dim="freqbands")
    original_dims = [*fooof_dims, freq_dim]
    target_dims = ["freqbands" if dim == freq_dim else dim for dim in original_dims]
    band_power_xr = band_power_xr.transpose(*target_dims)
    band_power_xr = band_power_xr.assign_coords(freqbands=list(bands_dict))
    band_power_xr.coords["freqband_low"] = ("freqbands", [e[0] for e in band_edges])
    band_power_xr.coords["freqband_high"] = ("freqbands", [e[1] for e in band_edges])

    if relative:
        denom = total_power
        with np.errstate(divide="ignore", invalid="ignore"):
            normalised = band_power_xr / denom
        band_power_xr = xr.where(denom == 0, np.nan, normalised)

    if log_transform:
        with np.errstate(divide="ignore", invalid="ignore"):
            band_power_xr = np.log10(band_power_xr)

    metadata: dict[str, Any] = {
        "bands": {
            label: {"low": float(low), "high": float(high)}
            for label, (low, high) in bands_dict.items()
        },
        "relative": relative,
        "log_transform": log_transform,
        "aperiodic_mode": aperiodic_mode,
        "aperiodic_floor": aperiodic_floor,
        "freq_dim": freq_dim,
        "output_dims": list(band_power_xr.dims),
        "output_shape": [int(band_power_xr.sizes[dim]) for dim in band_power_xr.dims],
        "integration": "xarray.DataArray.integrate (trapezoidal)",
    }
    band_power_xr.attrs["metadata"] = json.dumps(metadata, indent=2, default=_json_safe)

    artifacts = {
        ".nc": Artifact(item=band_power_xr, writer=lambda path: band_power_xr.to_netcdf(path)),
    }
    return NodeResult(artifacts=artifacts)
