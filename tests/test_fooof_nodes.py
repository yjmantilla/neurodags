"""Tests for FOOOF-based spectral nodes.

Covers fooof, fooof_scalars, fooof_component, fooof_peaks, and
bandpower_corrected.  Written before the fooof→specparam migration so they
serve as a regression harness throughout that change.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_allclose

from neurodags.definitions import Artifact, NodeResult
from neurodags.nodes.spectral import (
    bandpower,
    bandpower_corrected,
    fooof as fooof_node,
    fooof_component,
    fooof_peaks,
    fooof_scalars,
)

fooof_pkg = pytest.importorskip("fooof")
from fooof.sim import gen_power_spectrum  # noqa: E402

# ── synthetic signal helpers ──────────────────────────────────────────────────

FREQ_RANGE = [1.0, 40.0]
OFFSET = 1.0
EXPONENT = 1.5
PEAK_CF = 10.0
PEAK_PW = 0.5
PEAK_BW = 1.0


def _make_psd_da(n_channels=2, peaks=None, offset=OFFSET, exponent=EXPONENT):
    """Build a PSD DataArray using FOOOF's own simulator for exact parameter recovery."""
    ap_params = [offset, exponent]
    peak_params = list(peaks) if peaks else []
    freqs, psd0 = gen_power_spectrum(FREQ_RANGE, ap_params, peak_params, freq_res=0.2)
    data = np.stack([psd0 for _ in range(n_channels)])
    return xr.DataArray(
        data,
        dims=("spaces", "frequencies"),
        coords={"spaces": [f"ch{i}" for i in range(n_channels)], "frequencies": freqs},
    )


def _run_fooof(psd_da, **kwargs):
    kwargs.setdefault("fooof_options", {"fit": {"freq_range": FREQ_RANGE}})
    return fooof_node(psd_da, **kwargs)


def _fooof_da(result: NodeResult) -> xr.DataArray:
    item = result.artifacts[".nc"].item
    if isinstance(item, xr.Dataset):
        return item["fooof"]
    return item


def _nr(da: xr.DataArray) -> NodeResult:
    return NodeResult(artifacts={".nc": Artifact(item=da, writer=lambda p: None)})


# ── fooof ────────────────────────────────────────────────────────────────────


class TestFooof:
    def test_returns_node_result_with_nc(self):
        result = _run_fooof(_make_psd_da())
        assert ".nc" in result.artifacts

    def test_output_is_dataset(self):
        result = _run_fooof(_make_psd_da())
        assert isinstance(result.artifacts[".nc"].item, xr.Dataset)

    def test_dataset_contains_fooof_variable(self):
        result = _run_fooof(_make_psd_da())
        ds = result.artifacts[".nc"].item
        assert "fooof" in ds

    def test_timings_present_by_default(self):
        result = _run_fooof(_make_psd_da())
        ds = result.artifacts[".nc"].item
        assert "timings" in ds

    def test_timings_absent_when_disabled(self):
        result = _run_fooof(_make_psd_da(), include_timings=False)
        ds = result.artifacts[".nc"].item
        assert "timings" not in ds

    def test_output_shape_removes_freq_dim(self):
        result = _run_fooof(_make_psd_da(n_channels=3))
        fooof_da = _fooof_da(result)
        assert fooof_da.dims == ("spaces",)
        assert fooof_da.sizes["spaces"] == 3

    def test_3d_input_epochs_and_spaces(self):
        psd_2ch = _make_psd_da(n_channels=2)
        psd_3d = xr.DataArray(
            np.stack([psd_2ch.values, psd_2ch.values]),
            dims=("epochs", "spaces", "frequencies"),
            coords={"epochs": [0, 1], "spaces": ["ch0", "ch1"], "frequencies": psd_2ch.coords["frequencies"].values},
        )
        result = _run_fooof(psd_3d)
        fooof_da = _fooof_da(result)
        assert set(fooof_da.dims) == {"epochs", "spaces"}
        assert fooof_da.sizes["epochs"] == 2
        assert fooof_da.sizes["spaces"] == 2

    def test_output_payloads_are_parseable_json(self):
        result = _run_fooof(_make_psd_da())
        fooof_da = _fooof_da(result)
        for payload in fooof_da.values.flat:
            parsed = json.loads(str(payload))
            assert isinstance(parsed, dict)

    def test_accepts_noderesult_input(self):
        psd = _make_psd_da()
        nr = _nr(psd)
        result = _run_fooof(nr)
        assert ".nc" in result.artifacts

    def test_metadata_on_dataset(self):
        result = _run_fooof(_make_psd_da())
        ds = result.artifacts[".nc"].item
        assert "metadata" in ds.attrs
        meta = json.loads(ds.attrs["metadata"])
        assert "frequencies" in meta
        assert "fooof_kwargs" in meta
        assert "failures" in meta

    def test_invalid_psd_stores_fallback_value(self):
        psd = _make_psd_da()
        psd_nan = psd.copy(data=np.full_like(psd.values, np.nan))
        result = _run_fooof(psd_nan, failure_value="{}")
        fooof_da = _fooof_da(result)
        for payload in fooof_da.values.flat:
            assert str(payload) == "{}"

    def test_freq_range_stored_in_fit_kwargs_metadata(self):
        psd = _make_psd_da()
        result = fooof_node(psd, fooof_options={"fit": {"freq_range": [2.0, 30.0]}})
        ds = result.artifacts[".nc"].item
        meta = json.loads(ds.attrs["metadata"])
        assert meta["fit_kwargs"]["freq_range"] == [2.0, 30.0]


# ── fooof_scalars ────────────────────────────────────────────────────────────


class TestFooofScalars:
    @pytest.fixture
    def fooof_result(self):
        psd = _make_psd_da(peaks=[[PEAK_CF, PEAK_PW, PEAK_BW]])
        return _run_fooof(psd)

    def test_aperiodic_params_variables_present(self, fooof_result):
        result = fooof_scalars(fooof_result, component="aperiodic_params")
        ds = result.artifacts[".nc"].item
        assert "fooof_aperiodic_offset" in ds
        assert "fooof_aperiodic_exponent" in ds
        assert "fooof_aperiodic_knee" in ds

    def test_aperiodic_offset_close_to_ground_truth(self, fooof_result):
        result = fooof_scalars(fooof_result, component="aperiodic_params")
        ds = result.artifacts[".nc"].item
        assert_allclose(ds["fooof_aperiodic_offset"].values, OFFSET, atol=0.1)

    def test_aperiodic_exponent_close_to_ground_truth(self, fooof_result):
        result = fooof_scalars(fooof_result, component="aperiodic_params")
        ds = result.artifacts[".nc"].item
        assert_allclose(ds["fooof_aperiodic_exponent"].values, EXPONENT, atol=0.1)

    def test_r_squared_above_threshold_on_clean_signal(self, fooof_result):
        result = fooof_scalars(fooof_result, component="r_squared")
        ds = result.artifacts[".nc"].item
        assert "fooof_r_squared" in ds
        assert np.all(ds["fooof_r_squared"].values > 0.95)

    def test_error_is_finite(self, fooof_result):
        result = fooof_scalars(fooof_result, component="error")
        ds = result.artifacts[".nc"].item
        assert "fooof_error" in ds
        assert np.all(np.isfinite(ds["fooof_error"].values))

    def test_all_component_contains_every_variable(self, fooof_result):
        result = fooof_scalars(fooof_result, component="all")
        ds = result.artifacts[".nc"].item
        expected = {
            "fooof_aperiodic_offset",
            "fooof_aperiodic_knee",
            "fooof_aperiodic_exponent",
            "fooof_r_squared",
            "fooof_error",
        }
        assert expected.issubset(set(ds.data_vars))

    def test_fixed_mode_knee_is_nan(self, fooof_result):
        result = fooof_scalars(fooof_result, component="aperiodic_params")
        ds = result.artifacts[".nc"].item
        assert np.all(np.isnan(ds["fooof_aperiodic_knee"].values))

    def test_output_dims_match_input(self, fooof_result):
        result = fooof_scalars(fooof_result, component="all")
        ds = result.artifacts[".nc"].item
        for var in ds.data_vars:
            assert ds[var].dims == ("spaces",)

    def test_accepts_fooof_noderesult_directly(self, fooof_result):
        result = fooof_scalars(fooof_result, component="r_squared")
        assert ".nc" in result.artifacts

    def test_invalid_payload_yields_nan(self):
        fooof_da = xr.DataArray(
            np.array(["", "{}", "not-json"], dtype=object),
            dims=("spaces",),
            coords={"spaces": ["ch0", "ch1", "ch2"]},
        )
        fooof_da.attrs["metadata"] = json.dumps({})
        result = fooof_scalars(_nr(fooof_da), component="r_squared")
        ds = result.artifacts[".nc"].item
        assert np.all(np.isnan(ds["fooof_r_squared"].values))

    def test_invalid_component_raises(self, fooof_result):
        with pytest.raises(ValueError, match="component must be one of"):
            fooof_scalars(fooof_result, component="bogus")


# ── fooof_component ──────────────────────────────────────────────────────────


class TestFooofComponent:
    @pytest.fixture
    def fooof_result(self):
        psd = _make_psd_da(n_channels=2, peaks=[[PEAK_CF, PEAK_PW, PEAK_BW]])
        return _run_fooof(psd)

    def test_aperiodic_component_has_freq_dim(self, fooof_result):
        result = fooof_component(fooof_result, component="aperiodic", mode="manual")
        ds = result.artifacts[".nc"].item
        assert "aperiodic" in ds
        assert "frequencies" in ds["aperiodic"].dims

    def test_all_returns_three_variables(self, fooof_result):
        result = fooof_component(fooof_result, component="all", mode="manual")
        ds = result.artifacts[".nc"].item
        assert set(ds.data_vars) == {"aperiodic", "periodic", "residual"}

    def test_space_log_vs_linear_differ(self, fooof_result):
        log_r = fooof_component(fooof_result, component="aperiodic", space="log", mode="manual")
        lin_r = fooof_component(fooof_result, component="aperiodic", space="linear", mode="manual")
        log_vals = log_r.artifacts[".nc"].item["aperiodic"].values
        lin_vals = lin_r.artifacts[".nc"].item["aperiodic"].values
        assert not np.allclose(log_vals, lin_vals)

    def test_linear_equals_pow10_of_log(self, fooof_result):
        log_r = fooof_component(fooof_result, component="aperiodic", space="log", mode="manual")
        lin_r = fooof_component(fooof_result, component="aperiodic", space="linear", mode="manual")
        log_vals = log_r.artifacts[".nc"].item["aperiodic"].values
        lin_vals = lin_r.artifacts[".nc"].item["aperiodic"].values
        assert_allclose(np.power(10.0, log_vals), lin_vals, rtol=1e-5)

    def test_aperiodic_is_monotone_decreasing_in_log(self, fooof_result):
        result = fooof_component(fooof_result, component="aperiodic", space="log", mode="manual")
        ap = result.artifacts[".nc"].item["aperiodic"].values
        for row in ap:
            diffs = np.diff(row)
            assert np.all(diffs <= 1e-6), "Aperiodic (log) must be monotone decreasing"

    def test_output_shape_matches_input_plus_freq(self, fooof_result):
        result = fooof_component(fooof_result, component="aperiodic", mode="manual")
        ds = result.artifacts[".nc"].item
        ap = ds["aperiodic"]
        assert ap.sizes["spaces"] == 2
        assert ap.sizes["frequencies"] > 0

    def test_invalid_component_raises(self, fooof_result):
        with pytest.raises(ValueError, match="component must be one of"):
            fooof_component(fooof_result, component="bogus")

    def test_invalid_space_raises(self, fooof_result):
        with pytest.raises(ValueError, match="space must be"):
            fooof_component(fooof_result, component="aperiodic", space="db")

    def test_invalid_mode_raises(self, fooof_result):
        with pytest.raises(ValueError, match="mode must be"):
            fooof_component(fooof_result, component="aperiodic", mode="unknown")


# ── fooof_peaks ──────────────────────────────────────────────────────────────


class TestFooofPeaks:
    def _run_with_peak(self, cf, pw, bw, n_channels=2):
        psd = _make_psd_da(n_channels=n_channels, peaks=[[cf, pw, bw]])
        return _run_fooof(
            psd,
            fooof_options={"FOOOF": {"max_n_peaks": 5}, "fit": {"freq_range": FREQ_RANGE}},
        )

    def test_output_has_all_variables(self):
        result = _run_fooof(_make_psd_da())
        peaks_r = fooof_peaks(result)
        ds = peaks_r.artifacts[".nc"].item
        expected = {
            "n_peaks",
            "dominant_peak_cf",
            "dominant_peak_pw",
            "dominant_peak_bw",
            "alpha_peak_cf",
            "alpha_peak_pw",
            "alpha_peak_bw",
        }
        assert expected.issubset(set(ds.data_vars))

    def test_no_peaks_gives_zero_count_and_nan(self):
        psd = _make_psd_da()
        result = _run_fooof(
            psd,
            fooof_options={"FOOOF": {"max_n_peaks": 0}, "fit": {"freq_range": FREQ_RANGE}},
        )
        peaks_r = fooof_peaks(result)
        ds = peaks_r.artifacts[".nc"].item
        assert np.all(ds["n_peaks"].values == 0)
        assert np.all(np.isnan(ds["dominant_peak_cf"].values))
        assert np.all(np.isnan(ds["alpha_peak_cf"].values))

    def test_alpha_peak_detected_and_close_to_cf(self):
        result = self._run_with_peak(PEAK_CF, PEAK_PW, PEAK_BW)
        peaks_r = fooof_peaks(result, alpha_band=(8.0, 13.0))
        ds = peaks_r.artifacts[".nc"].item
        alpha_cf = ds["alpha_peak_cf"].values
        assert np.all(np.isfinite(alpha_cf))
        assert_allclose(alpha_cf, PEAK_CF, atol=1.0)

    def test_out_of_alpha_band_peak_gives_nan_alpha(self):
        psd = _make_psd_da(n_channels=2, peaks=[[25.0, PEAK_PW, PEAK_BW]])
        result = _run_fooof(
            psd,
            fooof_options={"FOOOF": {"max_n_peaks": 1}, "fit": {"freq_range": FREQ_RANGE}},
        )
        peaks_r = fooof_peaks(result, alpha_band=(8.0, 13.0))
        ds = peaks_r.artifacts[".nc"].item
        assert np.all(np.isnan(ds["alpha_peak_cf"].values))

    def test_dominant_peak_is_highest_power_peak(self):
        psd = _make_psd_da(n_channels=1, peaks=[[PEAK_CF, 0.2, 1.0], [25.0, 1.0, 1.0]])
        result = _run_fooof(
            psd,
            fooof_options={"FOOOF": {"max_n_peaks": 5}, "fit": {"freq_range": FREQ_RANGE}},
        )
        peaks_r = fooof_peaks(result)
        ds = peaks_r.artifacts[".nc"].item
        dom_cf = float(ds["dominant_peak_cf"].values.flat[0])
        assert_allclose(dom_cf, 25.0, atol=3.0)

    def test_output_dims_match_input(self):
        psd = _make_psd_da(n_channels=3)
        result = _run_fooof(psd)
        peaks_r = fooof_peaks(result)
        ds = peaks_r.artifacts[".nc"].item
        for var in ds.data_vars:
            assert ds[var].dims == ("spaces",)
            assert ds[var].sizes["spaces"] == 3

    def test_custom_alpha_band_theta_peak(self):
        result = self._run_with_peak(6.0, PEAK_PW, PEAK_BW)
        peaks_r = fooof_peaks(result, alpha_band=(4.0, 8.0))
        ds = peaks_r.artifacts[".nc"].item
        assert np.all(np.isfinite(ds["alpha_peak_cf"].values))

    def test_accepts_fooof_noderesult_directly(self):
        result = _run_fooof(_make_psd_da())
        peaks_r = fooof_peaks(result)
        assert ".nc" in peaks_r.artifacts

    def test_n_peaks_is_integer_dtype(self):
        result = _run_fooof(_make_psd_da())
        peaks_r = fooof_peaks(result)
        ds = peaks_r.artifacts[".nc"].item
        assert np.issubdtype(ds["n_peaks"].dtype, np.integer)


# ── bandpower_corrected ──────────────────────────────────────────────────────


class TestBandpowerCorrected:
    @pytest.fixture
    def psd_and_fooof(self):
        psd = _make_psd_da(n_channels=2, peaks=[[PEAK_CF, PEAK_PW, PEAK_BW]])
        fooof_r = _run_fooof(psd)
        return psd, fooof_r

    def test_returns_dataarray_with_freqbands(self, psd_and_fooof):
        psd, fooof_r = psd_and_fooof
        result = bandpower_corrected(psd, fooof_r, bands={"alpha": (8.0, 13.0)})
        da = result.artifacts[".nc"].item
        assert isinstance(da, xr.DataArray)
        assert "freqbands" in da.dims
        assert "alpha" in da.coords["freqbands"].values

    def test_output_shape(self, psd_and_fooof):
        psd, fooof_r = psd_and_fooof
        result = bandpower_corrected(psd, fooof_r, bands={"alpha": (8.0, 13.0), "beta": (13.0, 30.0)})
        da = result.artifacts[".nc"].item
        assert da.sizes["spaces"] == 2
        assert da.sizes["freqbands"] == 2

    def test_corrected_differs_from_uncorrected(self, psd_and_fooof):
        psd, fooof_r = psd_and_fooof
        bands = {"alpha": (8.0, 13.0)}
        corrected = bandpower_corrected(psd, fooof_r, bands=bands)
        uncorrected = bandpower(psd, bands=bands)
        corr_vals = corrected.artifacts[".nc"].item.values
        uncorr_vals = uncorrected.artifacts[".nc"].item.values
        assert not np.allclose(corr_vals, uncorr_vals)

    def test_corrected_values_are_positive(self, psd_and_fooof):
        psd, fooof_r = psd_and_fooof
        result = bandpower_corrected(psd, fooof_r, bands={"alpha": (8.0, 13.0)})
        da = result.artifacts[".nc"].item
        assert np.all(da.values > 0)

    def test_relative_values_between_zero_and_one(self, psd_and_fooof):
        psd, fooof_r = psd_and_fooof
        result = bandpower_corrected(
            psd, fooof_r,
            bands={"delta": (1.0, 4.0), "alpha": (8.0, 13.0)},
            relative=True,
        )
        da = result.artifacts[".nc"].item
        assert np.all(da.values >= 0)
        assert np.all(da.values <= 1)

    def test_log_transform_produces_finite_values(self, psd_and_fooof):
        psd, fooof_r = psd_and_fooof
        result = bandpower_corrected(psd, fooof_r, bands={"alpha": (8.0, 13.0)}, log_transform=True)
        da = result.artifacts[".nc"].item
        assert np.all(np.isfinite(da.values))

    def test_invalid_fooof_payload_yields_nan(self):
        psd = _make_psd_da(n_channels=2)
        bad_da = xr.DataArray(
            np.array(["", "{}"], dtype=object),
            dims=("spaces",),
            coords={"spaces": ["ch0", "ch1"]},
        )
        bad_da.attrs["metadata"] = json.dumps({})
        result = bandpower_corrected(psd, _nr(bad_da), bands={"alpha": (8.0, 13.0)})
        da = result.artifacts[".nc"].item
        assert np.all(np.isnan(da.values))

    def test_metadata_recorded(self, psd_and_fooof):
        psd, fooof_r = psd_and_fooof
        result = bandpower_corrected(psd, fooof_r, bands={"alpha": (8.0, 13.0)})
        da = result.artifacts[".nc"].item
        assert "metadata" in da.attrs
        meta = json.loads(da.attrs["metadata"])
        assert "bands" in meta
        assert "aperiodic_mode" in meta
        assert "aperiodic_floor" in meta

    def test_freqband_coords_present(self, psd_and_fooof):
        psd, fooof_r = psd_and_fooof
        result = bandpower_corrected(psd, fooof_r, bands={"alpha": (8.0, 13.0)})
        da = result.artifacts[".nc"].item
        assert "freqband_low" in da.coords
        assert "freqband_high" in da.coords
