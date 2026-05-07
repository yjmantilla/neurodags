"""Tests for spectral nodes using real MNE Raw and Epochs objects."""
from __future__ import annotations

import numpy as np
import xarray as xr

from neurodags.definitions import Artifact, NodeResult
from neurodags.nodes.spectral import mne_spectrum, mne_spectrum_array


# ---------------------------------------------------------------------------
# mne_spectrum (compute_psd wrapper)
# ---------------------------------------------------------------------------

def test_mne_spectrum_raw_returns_2d_xarray(dummy_raw_obj):
    raw, _ = dummy_raw_obj
    result = mne_spectrum(raw.copy())
    da = result.artifacts[".nc"].item
    assert isinstance(da, xr.DataArray)
    assert da.dims == ("spaces", "frequencies")
    assert da.sizes["spaces"] == raw.info["nchan"]
    assert da.sizes["frequencies"] > 0


def test_mne_spectrum_epochs_returns_3d_xarray(dummy_epochs_obj):
    epochs = dummy_epochs_obj
    result = mne_spectrum(epochs)
    da = result.artifacts[".nc"].item
    assert isinstance(da, xr.DataArray)
    assert da.dims == ("epochs", "spaces", "frequencies")
    assert da.sizes["spaces"] == epochs.info["nchan"]
    assert da.sizes["epochs"] == len(epochs)


def test_mne_spectrum_accepts_noderesulet(dummy_raw_obj):
    raw, _ = dummy_raw_obj
    nr = NodeResult(artifacts={".fif": Artifact(item=raw.copy(), writer=lambda p: None)})
    result = mne_spectrum(nr)
    da = result.artifacts[".nc"].item
    assert "spaces" in da.dims


def test_mne_spectrum_fmax_clipped_to_nyquist(dummy_raw_obj):
    raw, _ = dummy_raw_obj
    nyquist = raw.info["sfreq"] / 2
    result = mne_spectrum(raw.copy(), compute_psd_kwargs={"fmax": nyquist * 10})
    da = result.artifacts[".nc"].item
    assert da.coords["frequencies"].values.max() <= nyquist + 1e-6


# ---------------------------------------------------------------------------
# mne_spectrum_array (welch / multitaper)
# ---------------------------------------------------------------------------

def test_mne_spectrum_array_welch_raw_output_shape(dummy_raw_obj):
    raw, _ = dummy_raw_obj
    result = mne_spectrum_array(raw.copy(), method="welch")
    artifact = result.artifacts[".nc"].item
    # raw → dataset with 'spectrum' variable shaped (spaces, frequencies)
    assert isinstance(artifact, xr.Dataset)
    spectrum = artifact["spectrum"]
    assert "spaces" in spectrum.dims
    assert "frequencies" in spectrum.dims


def test_mne_spectrum_array_welch_epochs_output_shape(dummy_epochs_obj):
    epochs = dummy_epochs_obj
    # n_per_seg must be <= epoch length (91 samples at 100 Hz / 0.9 s epochs)
    result = mne_spectrum_array(epochs, method="welch", method_kwargs={"n_per_seg": 64})
    artifact = result.artifacts[".nc"].item
    assert isinstance(artifact, xr.Dataset)
    spectrum = artifact["spectrum"]
    assert "epochs" in spectrum.dims
    assert "spaces" in spectrum.dims
    assert "frequencies" in spectrum.dims
    assert spectrum.sizes["epochs"] == len(epochs)
    assert spectrum.sizes["spaces"] == epochs.info["nchan"]


def test_mne_spectrum_array_multitaper_epochs(dummy_epochs_obj):
    epochs = dummy_epochs_obj
    result = mne_spectrum_array(
        epochs,
        method="multitaper",
        method_kwargs={"low_bias": True, "normalization": "length"},
    )
    artifact = result.artifacts[".nc"].item
    assert isinstance(artifact, xr.Dataset)
    assert "spectrum" in artifact.data_vars


def test_mne_spectrum_array_values_non_negative(dummy_epochs_obj):
    epochs = dummy_epochs_obj
    result = mne_spectrum_array(epochs, method="welch", method_kwargs={"n_per_seg": 64})
    spectrum = result.artifacts[".nc"].item["spectrum"]
    assert float(spectrum.values.min()) >= 0.0


def test_mne_spectrum_array_frequencies_monotonic(dummy_raw_obj):
    raw, _ = dummy_raw_obj
    result = mne_spectrum_array(raw.copy(), method="welch")
    freqs = result.artifacts[".nc"].item["spectrum"].coords["frequencies"].values
    assert np.all(np.diff(freqs) > 0)


def test_mne_spectrum_array_accepts_noderesulet(dummy_epochs_obj):
    epochs = dummy_epochs_obj
    nr = NodeResult(artifacts={".fif": Artifact(item=epochs, writer=lambda p: None)})
    result = mne_spectrum_array(nr, method="welch", method_kwargs={"n_per_seg": 64})
    artifact = result.artifacts[".nc"].item
    assert "spectrum" in artifact.data_vars
