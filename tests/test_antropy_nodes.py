from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from neurodags.nodes import get_node

antropy = pytest.importorskip("antropy")


def _make_dataarray() -> xr.DataArray:
    data = np.vstack([
        np.sin(np.linspace(0.0, np.pi, 128)),
        np.cos(np.linspace(0.0, np.pi, 128)),
        np.linspace(0.0, 1.0, 128),
    ])
    return xr.DataArray(data, dims=("spaces", "times"))


def test_app_entropy_node_matches_antropy() -> None:
    da = _make_dataarray()
    node = get_node("antropy_app_entropy")

    result = node(da, dim="times", order=2, tolerance=0.2)
    out = result.artifacts[".nc"].item

    expected = np.array([
        antropy.app_entropy(signal, order=2, tolerance=0.2)
        for signal in da.values
    ])

    assert out.dims == ("spaces",)
    np.testing.assert_allclose(out.values, expected, rtol=1e-6, atol=1e-6)


def test_hjorth_params_node_sets_component_labels() -> None:
    da = _make_dataarray()
    node = get_node("antropy_hjorth_params")

    result = node(da, dim="times")
    out = result.artifacts[".nc"].item

    assert out.dims == ("spaces", "hjorthComponents")
    assert list(out.coords["hjorthComponents"].values) == ["mobility", "complexity"]

    expected = np.array([antropy.hjorth_params(signal) for signal in da.values])
    np.testing.assert_allclose(out.sel(hjorthComponents="mobility").values, expected[:, 0])
    np.testing.assert_allclose(out.sel(hjorthComponents="complexity").values, expected[:, 1])


def test_spectral_entropy_node_requires_sampling_frequency() -> None:
    da = _make_dataarray()
    node = get_node("antropy_spectral_entropy")

    sf = 128.0
    result = node(da, dim="times", sf=sf)
    out = result.artifacts[".nc"].item

    expected = np.array([antropy.spectral_entropy(signal, sf=sf) for signal in da.values])

    assert out.dims == ("spaces",)
    np.testing.assert_allclose(out.values, expected, rtol=1e-6, atol=1e-6)

