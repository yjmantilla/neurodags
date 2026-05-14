"""Extra coverage tests for spectral nodes."""
from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from neurodags.definitions import Artifact, NodeResult
from neurodags.nodes.spectral import (
    _resolve_psd_dataarray,
    band_ratios,
    bandpower,
    fooof,
    fooof_component,
    fooof_scalars,
    mne_spectrum,
    mne_spectrum_array,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def psd_da():
    freqs = np.linspace(1.0, 40.0, 100)
    data = np.ones((3, len(freqs)))
    return xr.DataArray(
        data,
        dims=("spaces", "frequencies"),
        coords={"spaces": ["ch1", "ch2", "ch3"], "frequencies": freqs},
    )


@pytest.fixture
def bandpower_da():
    data = np.array([[1.0, 2.0, 0.5]])
    return xr.DataArray(
        data,
        dims=("spaces", "freqbands"),
        coords={"spaces": ["ch1"], "freqbands": ["delta", "alpha", "beta"]},
    )


@pytest.fixture
def fooof_psd_da():
    freqs = np.linspace(1.0, 40.0, 100)
    psd = 1.0 / (freqs**2) + 0.05 * np.exp(-((freqs - 10.0) ** 2) / (2 * 1.5**2))
    return xr.DataArray(
        psd[np.newaxis, :],
        dims=("spaces", "frequencies"),
        coords={"spaces": ["ch1"], "frequencies": freqs},
    )


@pytest.fixture
def fooof_result(fooof_psd_da):
    return fooof(
        fooof_psd_da,
        fooof_options={
            "FOOOF": {"max_n_peaks": 3, "peak_threshold": 2.0},
            "fit": {"freq_range": [2.0, 40.0]},
            "save": {"save_results": True, "save_settings": True, "save_data": False},
        },
    )


# ---------------------------------------------------------------------------
# _resolve_psd_dataarray — error/edge branches
# ---------------------------------------------------------------------------

def test_resolve_psd_xarray_returns_itself(psd_da):
    assert _resolve_psd_dataarray(psd_da) is psd_da


def test_resolve_psd_noderesulet_xarray(psd_da):
    nr = NodeResult(artifacts={".nc": Artifact(item=psd_da, writer=lambda p: None)})
    result = _resolve_psd_dataarray(nr)
    assert result is psd_da


def test_resolve_psd_noderesulet_missing_nc_raises():
    nr = NodeResult(artifacts={".fif": Artifact(item=None, writer=lambda p: None)})
    with pytest.raises(ValueError, match=".nc"):
        _resolve_psd_dataarray(nr)


def test_resolve_psd_noderesulet_bad_item_raises():
    nr = NodeResult(artifacts={".nc": Artifact(item=42, writer=lambda p: None)})
    with pytest.raises(ValueError, match="Unsupported artifact"):
        _resolve_psd_dataarray(nr)


def test_resolve_psd_from_path(psd_da, tmp_path):
    nc_path = tmp_path / "psd.nc"
    psd_da.to_netcdf(nc_path)
    result = _resolve_psd_dataarray(nc_path)
    assert "frequencies" in result.dims


def test_resolve_psd_invalid_type_raises():
    with pytest.raises(ValueError, match="must be a NodeResult"):
        _resolve_psd_dataarray(42)


# ---------------------------------------------------------------------------
# mne_spectrum — missing branches
# ---------------------------------------------------------------------------

def test_mne_spectrum_noderesulet_missing_fif_raises():
    nr = NodeResult(artifacts={".nc": Artifact(item=None, writer=lambda p: None)})
    with pytest.raises(ValueError, match=".fif"):
        mne_spectrum(nr)


def test_mne_spectrum_from_path(dummy_vhdr_file):
    result = mne_spectrum(dummy_vhdr_file)
    assert isinstance(result, NodeResult)
    assert ".nc" in result.artifacts


def test_mne_spectrum_extra_artifacts_raw(dummy_raw_obj):
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    raw, _ = dummy_raw_obj
    result = mne_spectrum(raw.copy(), extra_artifacts=True)
    assert ".report.html" in result.artifacts


def test_mne_spectrum_extra_artifacts_epochs(dummy_epochs_obj):
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    epochs = dummy_epochs_obj
    result = mne_spectrum(epochs, extra_artifacts=True)
    assert ".report.html" in result.artifacts


# ---------------------------------------------------------------------------
# mne_spectrum_array — missing branches
# ---------------------------------------------------------------------------

def test_mne_spectrum_array_noderesulet_missing_fif_raises():
    nr = NodeResult(artifacts={".nc": Artifact(item=None, writer=lambda p: None)})
    with pytest.raises(ValueError, match=".fif"):
        mne_spectrum_array(nr, method="welch")


def test_mne_spectrum_array_invalid_method_raises(dummy_raw_obj):
    raw, _ = dummy_raw_obj
    with pytest.raises(ValueError, match="method must be either"):
        mne_spectrum_array(raw.copy(), method="invalid_method")


def test_mne_spectrum_array_from_path(dummy_vhdr_file):
    result = mne_spectrum_array(dummy_vhdr_file, method="welch")
    assert isinstance(result, NodeResult)
    assert ".nc" in result.artifacts


def test_mne_spectrum_array_welch_no_average_has_segments(dummy_epochs_obj):
    epochs = dummy_epochs_obj
    result = mne_spectrum_array(
        epochs,
        method="welch",
        method_kwargs={"n_per_seg": 32, "average": None},
    )
    artifact = result.artifacts[".nc"].item
    assert isinstance(artifact, xr.Dataset)


def test_mne_spectrum_array_multitaper_with_weights(dummy_epochs_obj):
    epochs = dummy_epochs_obj
    result = mne_spectrum_array(
        epochs,
        method="multitaper",
        method_kwargs={"low_bias": True, "normalization": "length"},
    )
    artifact = result.artifacts[".nc"].item
    assert "spectrum" in artifact.data_vars


# ---------------------------------------------------------------------------
# fooof — basic run + error branches
# ---------------------------------------------------------------------------

def test_fooof_returns_noderesulet(fooof_psd_da):
    result = fooof(fooof_psd_da)
    assert isinstance(result, NodeResult)
    assert ".nc" in result.artifacts


def test_fooof_output_contains_strings(fooof_psd_da):
    result = fooof(
        fooof_psd_da,
        fooof_options={
            "FOOOF": {"max_n_peaks": 3},
            "fit": {"freq_range": [2.0, 40.0]},
            "save": {"save_results": True, "save_settings": True, "save_data": False},
        },
    )
    ds = result.artifacts[".nc"].item
    fooof_xr = ds["fooof"]
    assert isinstance(fooof_xr.values.flat[0], str)


def test_fooof_missing_freq_dim_raises(psd_da):
    with pytest.raises(ValueError, match="not present in input dims"):
        fooof(psd_da, freq_dim="nonexistent_dim")


def test_fooof_explicit_freqs(fooof_psd_da):
    freqs = fooof_psd_da.coords["frequencies"].values
    da_no_coord = fooof_psd_da.drop_vars("frequencies")
    result = fooof(da_no_coord, freqs=freqs)
    assert isinstance(result, NodeResult)


def test_fooof_freq_res(fooof_psd_da):
    result = fooof(fooof_psd_da, fooof_options={"FOOOF": {}, "save": {"save_results": True, "save_settings": True, "save_data": False}})
    assert isinstance(result, NodeResult)


def test_fooof_no_timings(fooof_psd_da):
    result = fooof(fooof_psd_da, include_timings=False)
    ds = result.artifacts[".nc"].item
    assert "timings" not in ds.data_vars


def test_fooof_with_timings(fooof_psd_da):
    result = fooof(fooof_psd_da, include_timings=True)
    ds = result.artifacts[".nc"].item
    assert "timings" in ds.data_vars


def test_fooof_from_path(fooof_psd_da, tmp_path):
    nc_path = tmp_path / "psd.nc"
    fooof_psd_da.to_netcdf(nc_path)
    result = fooof(nc_path)
    assert isinstance(result, NodeResult)


def test_fooof_from_noderesulet(fooof_psd_da):
    nr = NodeResult(artifacts={".nc": Artifact(item=fooof_psd_da, writer=lambda p: None)})
    result = fooof(nr)
    assert isinstance(result, NodeResult)


# ---------------------------------------------------------------------------
# fooof_scalars — basic run + error branches
# ---------------------------------------------------------------------------

def test_fooof_scalars_all(fooof_result):
    ds = fooof_result.artifacts[".nc"].item
    fooof_xr = ds["fooof"]
    result = fooof_scalars(fooof_xr, component="all")
    assert isinstance(result, NodeResult)
    out_ds = result.artifacts[".nc"].item
    assert isinstance(out_ds, xr.Dataset)


def test_fooof_scalars_r_squared(fooof_result):
    ds = fooof_result.artifacts[".nc"].item
    fooof_xr = ds["fooof"]
    result = fooof_scalars(fooof_xr, component="r_squared")
    out_ds = result.artifacts[".nc"].item
    assert "fooof_r_squared" in out_ds.data_vars


def test_fooof_scalars_error(fooof_result):
    ds = fooof_result.artifacts[".nc"].item
    fooof_xr = ds["fooof"]
    result = fooof_scalars(fooof_xr, component="error")
    out_ds = result.artifacts[".nc"].item
    assert "fooof_error" in out_ds.data_vars


def test_fooof_scalars_aperiodic_params(fooof_result):
    ds = fooof_result.artifacts[".nc"].item
    fooof_xr = ds["fooof"]
    result = fooof_scalars(fooof_xr, component="aperiodic_params")
    out_ds = result.artifacts[".nc"].item
    assert "fooof_aperiodic_offset" in out_ds.data_vars


def test_fooof_scalars_invalid_component_raises(fooof_result):
    ds = fooof_result.artifacts[".nc"].item
    fooof_xr = ds["fooof"]
    with pytest.raises(ValueError, match="component must be one of"):
        fooof_scalars(fooof_xr, component="invalid")


def test_fooof_scalars_from_noderesulet(fooof_result):
    ds = fooof_result.artifacts[".nc"].item
    fooof_xr = ds["fooof"]
    nr = NodeResult(artifacts={".nc": Artifact(item=fooof_xr, writer=lambda p: None)})
    result = fooof_scalars(nr, component="all")
    assert isinstance(result, NodeResult)


# ---------------------------------------------------------------------------
# fooof_component — basic run + error branches
# ---------------------------------------------------------------------------

def test_fooof_component_aperiodic(fooof_result):
    ds = fooof_result.artifacts[".nc"].item
    fooof_xr = ds["fooof"]
    result = fooof_component(fooof_xr, component="aperiodic")
    assert isinstance(result, NodeResult)
    out_ds = result.artifacts[".nc"].item
    assert "aperiodic" in out_ds.data_vars


def test_fooof_component_periodic(fooof_result):
    ds = fooof_result.artifacts[".nc"].item
    fooof_xr = ds["fooof"]
    result = fooof_component(fooof_xr, component="periodic")
    out_ds = result.artifacts[".nc"].item
    assert "periodic" in out_ds.data_vars


def test_fooof_component_all(fooof_result):
    ds = fooof_result.artifacts[".nc"].item
    fooof_xr = ds["fooof"]
    result = fooof_component(fooof_xr, component="all")
    out_ds = result.artifacts[".nc"].item
    assert "aperiodic" in out_ds.data_vars
    assert "periodic" in out_ds.data_vars


def test_fooof_component_invalid_component_raises(fooof_result):
    ds = fooof_result.artifacts[".nc"].item
    fooof_xr = ds["fooof"]
    with pytest.raises(ValueError, match="component must be one of"):
        fooof_component(fooof_xr, component="bad")


def test_fooof_component_invalid_mode_raises(fooof_result):
    ds = fooof_result.artifacts[".nc"].item
    fooof_xr = ds["fooof"]
    with pytest.raises(ValueError, match="mode must be either"):
        fooof_component(fooof_xr, component="aperiodic", mode="bad_mode")


def test_fooof_component_invalid_space_raises(fooof_result):
    ds = fooof_result.artifacts[".nc"].item
    fooof_xr = ds["fooof"]
    with pytest.raises(ValueError, match="space must be either"):
        fooof_component(fooof_xr, component="aperiodic", space="bad_space")


def test_fooof_component_from_noderesulet(fooof_result):
    ds = fooof_result.artifacts[".nc"].item
    fooof_xr = ds["fooof"]
    nr = NodeResult(artifacts={".nc": Artifact(item=fooof_xr, writer=lambda p: None)})
    result = fooof_component(nr, component="aperiodic")
    assert isinstance(result, NodeResult)


# ---------------------------------------------------------------------------
# bandpower — error branches + input variants
# ---------------------------------------------------------------------------

def test_bandpower_missing_freq_dim_raises(psd_da):
    with pytest.raises(ValueError, match="not present in input dims"):
        bandpower(psd_da, freq_dim="nonexistent")




def test_bandpower_nonfinite_band_raises(psd_da):
    with pytest.raises(ValueError, match="non-finite"):
        bandpower(psd_da, bands={"bad": (float("nan"), 10.0)})


def test_bandpower_high_leq_low_raises(psd_da):
    with pytest.raises(ValueError, match="must have high > low"):
        bandpower(psd_da, bands={"bad": (10.0, 5.0)})


def test_bandpower_empty_band_slice_returns_nan(psd_da):
    result = bandpower(psd_da, bands={"out_of_range": (100.0, 200.0)})
    arr = result.artifacts[".nc"].item
    assert np.isnan(arr.sel(freqbands="out_of_range").values).all()


def test_bandpower_from_path(psd_da, tmp_path):
    nc_path = tmp_path / "psd.nc"
    psd_da.to_netcdf(nc_path)
    result = bandpower(nc_path, bands={"alpha": (8.0, 13.0)})
    assert isinstance(result, NodeResult)


def test_bandpower_source_metadata_propagated(psd_da):
    da = psd_da.copy()
    da.attrs["metadata"] = '{"test": "value"}'
    result = bandpower(da, bands={"alpha": (8.0, 13.0)})
    arr = result.artifacts[".nc"].item
    assert "source_metadata" in arr.attrs.get("metadata", "")


# ---------------------------------------------------------------------------
# band_ratios — error branches + input variants
# ---------------------------------------------------------------------------

def test_band_ratios_missing_freqband_dim_raises(psd_da):
    with pytest.raises(ValueError, match="not present in input dims"):
        band_ratios(psd_da, freqband_dim="nonexistent")



def test_band_ratios_explicit_combinations(bandpower_da):
    result = band_ratios(bandpower_da, combinations=[("alpha", "delta"), ("beta", "alpha")])
    arr = result.artifacts[".nc"].item
    assert "alpha/delta" in arr.coords["freqbandPairs"].values
    assert "beta/alpha" in arr.coords["freqbandPairs"].values


def test_band_ratios_empty_combinations_raises(bandpower_da):
    with pytest.raises(ValueError, match="At least one band pair"):
        band_ratios(bandpower_da, combinations=[])


def test_band_ratios_from_path(bandpower_da, tmp_path):
    nc_path = tmp_path / "bp.nc"
    bandpower_da.to_netcdf(nc_path)
    result = band_ratios(nc_path)
    assert isinstance(result, NodeResult)
