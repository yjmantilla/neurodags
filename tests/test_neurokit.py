from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

import matplotlib

from neurodags.nodes import get_node

matplotlib.use("Agg", force=True)

nk = pytest.importorskip("neurokit2")


def _make_dataarray() -> xr.DataArray:
    data = np.vstack([
        nk.signal_simulate(duration=2.0, sampling_rate=100, frequency=5.0, noise=0.01, random_state=0),
        nk.signal_simulate(duration=2.0, sampling_rate=100, frequency=8.0, noise=0.01, random_state=1),
    ])
    return xr.DataArray(data, dims=("spaces", "times"), coords={"spaces": ["A", "B"], "times": np.arange(data.shape[1])})


def test_complexity_delay_node_produces_dataset_with_png_figures() -> None:
    da = _make_dataarray()
    node = get_node("neurokit_complexity_delay")

    result = node(
        da,
        dim="times",
        delay_max=40,
        method="fraser1986",
        figure_encoding="png",
    )

    dataset = result.artifacts[".nc"].item
    assert isinstance(dataset, xr.Dataset)

    value_da = dataset["value"]
    expected = np.array([
        nk.complexity_delay(signal, delay_max=40, method="fraser1986", show=False)[0]
        for signal in da.values
    ])
    np.testing.assert_allclose(value_da.values, expected, rtol=1e-6, atol=1e-6)

    metadata_da = dataset["metadata"]
    assert metadata_da.dims == value_da.dims + ("metadata_fields",)
    assert metadata_da.dtype.kind in {"U", "S"}

    assert "figure_png_hex" in dataset.data_vars
    figure_da = dataset["figure_png_hex"]
    #assert figure_da.dims == value_da.dims #+ ("png_byte",)
    #assert figure_da.dtype == np.uint8
    assert figure_da.values.shape[-1] > 0


def test_complexity_delay_node_supports_rgba_encoding() -> None:
    da = _make_dataarray()
    node = get_node("neurokit_complexity_delay")

    result = node(
        da,
        dim="times",
        delay_max=30,
        method="fraser1986",
        figure_encoding="rgba",
    )

    dataset = result.artifacts[".nc"].item
    value_da = dataset["value"]

    assert "figure_rgba" in dataset.data_vars
    rgba_da = dataset["figure_rgba"]

    assert rgba_da.dims == value_da.dims + ("figure_y", "figure_x", "figure_channel")
    assert rgba_da.dtype == np.uint8
    assert rgba_da.sizes["figure_channel"] == 4
    assert rgba_da.values.shape[-3] > 0
    assert rgba_da.values.shape[-2] > 0

if __name__ == "__main__":
#    pytest.main([__file__])
    pytest.main(["-v", "-s", "-q", "--no-cov", "--pdb", __file__])
