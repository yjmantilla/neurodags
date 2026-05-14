"""Tests for the visualization entry point."""

from __future__ import annotations

import json
from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr
from dash import dcc, html

from neurodags.visualization import (
    _compute_figure,
    _controls_for_variable,
    _make_dim_controls,
    apply_transform,
    build_visualization_app,
    load_visualization_dataset,
    main,
    safe_sel,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def da_1d() -> xr.DataArray:
    return xr.DataArray(
        [1.0, 4.0, 9.0],
        dims=("freq",),
        coords={"freq": [1.0, 2.0, 3.0]},
    )


@pytest.fixture()
def da_2d() -> xr.DataArray:
    return xr.DataArray(
        np.arange(6.0).reshape(2, 3),
        dims=("channel", "freq"),
        coords={"channel": ["ch1", "ch2"], "freq": [1.0, 2.0, 3.0]},
    )


@pytest.fixture()
def da_3d() -> xr.DataArray:
    return xr.DataArray(
        np.ones((2, 3, 4)),
        dims=("epoch", "channel", "freq"),
        coords={
            "epoch": [0, 1],
            "channel": ["ch1", "ch2", "ch3"],
            "freq": [1.0, 2.0, 3.0, 4.0],
        },
    )


@pytest.fixture()
def dataset_multi(da_1d: xr.DataArray, da_2d: xr.DataArray) -> xr.Dataset:
    return xr.Dataset({"spectrum": da_1d, "bandpower": da_2d})


# ---------------------------------------------------------------------------
# load_visualization_dataset
# ---------------------------------------------------------------------------


def test_load_rejects_unknown_suffix(tmp_path: pytest.TempPathFactory) -> None:
    bad = tmp_path / "bad.txt"
    bad.write_text("x")
    with pytest.raises(ValueError, match="Expected \\.fif or \\.nc"):
        load_visualization_dataset(bad)


def test_load_nc_dataarray(tmp_path: pytest.TempPathFactory, da_1d: xr.DataArray) -> None:
    nc = tmp_path / "arr.nc"
    da_1d.to_netcdf(nc)
    result = load_visualization_dataset(nc)
    assert isinstance(result, xr.DataArray)
    assert list(result.dims) == ["freq"]


def test_load_nc_single_var_dataset_as_dataarray(
    tmp_path: pytest.TempPathFactory, da_2d: xr.DataArray
) -> None:
    # Single-variable Dataset saved to .nc is openable as DataArray
    nc = tmp_path / "ds.nc"
    da_2d.to_dataset(name="spectrum").to_netcdf(nc)
    result = load_visualization_dataset(nc)
    assert isinstance(result, xr.DataArray)


def test_load_nc_multivar_dataset(
    tmp_path: pytest.TempPathFactory, dataset_multi: xr.Dataset
) -> None:
    # Multi-variable Dataset cannot be opened as DataArray — falls back to Dataset
    nc = tmp_path / "multi.nc"
    dataset_multi.to_netcdf(nc)
    result = load_visualization_dataset(nc)
    assert isinstance(result, xr.Dataset)
    assert set(result.data_vars) == {"spectrum", "bandpower"}


# ---------------------------------------------------------------------------
# safe_sel
# ---------------------------------------------------------------------------


def test_safe_sel_numeric_nearest(da_1d: xr.DataArray) -> None:
    result = safe_sel(da_1d, {"freq": "1.4"})
    assert float(result.coords["freq"]) == pytest.approx(1.0)


def test_safe_sel_string_coord(da_2d: xr.DataArray) -> None:
    result = safe_sel(da_2d, {"channel": "ch2"})
    assert result.dims == ("freq",)
    assert result.shape == (3,)


def test_safe_sel_multiple_dims(da_2d: xr.DataArray) -> None:
    result = safe_sel(da_2d, {"channel": "ch1", "freq": "2.0"})
    assert result.ndim == 0


def test_safe_sel_empty_dict_is_identity(da_2d: xr.DataArray) -> None:
    result = safe_sel(da_2d, {})
    assert result.shape == da_2d.shape


# ---------------------------------------------------------------------------
# apply_transform
# ---------------------------------------------------------------------------


def test_apply_transform_none() -> None:
    arr = np.array([1.0, 2.0, 3.0])
    np.testing.assert_array_equal(apply_transform(arr, "none"), arr)


def test_apply_transform_log10() -> None:
    arr = np.array([1.0, 10.0, 100.0])
    result = apply_transform(arr, "log10")
    np.testing.assert_allclose(result, [0.0, 1.0, 2.0])


def test_apply_transform_log10_zeros_become_nan() -> None:
    arr = np.array([0.0, 1.0])
    result = apply_transform(arr, "log10")
    assert np.isnan(result[0])
    assert not np.isnan(result[1])


def test_apply_transform_square() -> None:
    arr = np.array([2.0, 3.0])
    result = apply_transform(arr, "square")
    np.testing.assert_allclose(result, [4.0, 9.0])


def test_apply_transform_log20() -> None:
    arr = np.array([1.0, 10.0])
    result = apply_transform(arr, "log20")
    np.testing.assert_allclose(result, [0.0, 2.0])


def test_apply_transform_non_numeric_passthrough() -> None:
    arr = np.array(["a", "b"])
    result = apply_transform(arr, "log10")
    np.testing.assert_array_equal(result, arr)


# ---------------------------------------------------------------------------
# _make_dim_controls
# ---------------------------------------------------------------------------


def test_make_dim_controls_structure() -> None:
    dims = ["channel", "freq"]
    coords = {"channel": ["ch1", "ch2"], "freq": [1.0, 2.0, 3.0]}
    children = _make_dim_controls(dims, coords)
    # 2 children per dim: Label + Dropdown
    assert len(children) == 4
    assert isinstance(children[0], html.Label)
    assert isinstance(children[1], dcc.Dropdown)
    assert isinstance(children[2], html.Label)
    assert isinstance(children[3], dcc.Dropdown)


def test_make_dim_controls_pattern_ids() -> None:
    dims = ["epoch", "channel"]
    coords = {"epoch": [0, 1], "channel": ["ch1"]}
    children = _make_dim_controls(dims, coords)
    dropdowns = [c for c in children if isinstance(c, dcc.Dropdown)]
    ids = [d.id for d in dropdowns]
    assert ids[0] == {"type": "dim-dropdown", "index": "epoch"}
    assert ids[1] == {"type": "dim-dropdown", "index": "channel"}


def test_make_dim_controls_initial_value_is_first() -> None:
    dims = ["freq"]
    coords = {"freq": [5.0, 10.0, 20.0]}
    children = _make_dim_controls(dims, coords)
    dropdown = children[1]
    assert dropdown.value == "5.0"


def test_make_dim_controls_empty_dims() -> None:
    assert _make_dim_controls([], {}) == []


# ---------------------------------------------------------------------------
# _controls_for_variable
# ---------------------------------------------------------------------------


def test_controls_for_variable_xval_is_last_dim(da_2d: xr.DataArray) -> None:
    _, x_opts, x_val, y_opts, y_val = _controls_for_variable(da_2d)
    assert x_val == "freq"


def test_controls_for_variable_y_includes_none(da_2d: xr.DataArray) -> None:
    _, _, _, y_opts, y_val = _controls_for_variable(da_2d)
    assert y_opts[0] == {"label": "None", "value": "none"}
    assert y_val == "none"


def test_controls_for_variable_x_opts_match_dims(da_2d: xr.DataArray) -> None:
    _, x_opts, _, _, _ = _controls_for_variable(da_2d)
    labels = [o["label"] for o in x_opts]
    assert labels == ["channel", "freq"]


def test_controls_for_variable_1d(da_1d: xr.DataArray) -> None:
    children, x_opts, x_val, y_opts, _ = _controls_for_variable(da_1d)
    assert x_val == "freq"
    assert len(x_opts) == 1
    # 2 children (label + dropdown) for the single dim
    assert len(children) == 2


# ---------------------------------------------------------------------------
# _compute_figure
# ---------------------------------------------------------------------------


def test_compute_figure_1d_line(da_1d: xr.DataArray) -> None:
    fig, debug = _compute_figure(da_1d, {}, "freq", None, "line", "none", "none")
    assert len(fig.data) == 1
    assert fig.data[0].type == "scatter"
    assert fig.data[0].mode == "lines"


def test_compute_figure_1d_scatter(da_1d: xr.DataArray) -> None:
    fig, _ = _compute_figure(da_1d, {}, "freq", None, "scatter", "none", "none")
    assert fig.data[0].mode == "markers"


def test_compute_figure_1d_bar(da_1d: xr.DataArray) -> None:
    fig, _ = _compute_figure(da_1d, {}, "freq", None, "bar", "none", "none")
    assert fig.data[0].type == "bar"


def test_compute_figure_2d_heatmap(da_2d: xr.DataArray) -> None:
    fig, _ = _compute_figure(da_2d, {}, "freq", "channel", "heatmap", "none", "none")
    assert fig.data[0].type == "heatmap"


def test_compute_figure_2d_line_per_series(da_2d: xr.DataArray) -> None:
    fig, _ = _compute_figure(da_2d, {}, "freq", "channel", "line", "none", "none")
    assert len(fig.data) == 2  # one trace per channel value
    assert all(t.type == "scatter" for t in fig.data)
    names = [t.name for t in fig.data]
    assert "channel=ch1" in names
    assert "channel=ch2" in names


def test_compute_figure_unsupported_shape(da_3d: xr.DataArray) -> None:
    # 3D after no slicing → annotation
    fig, _ = _compute_figure(da_3d, {}, "freq", None, "line", "none", "none")
    assert len(fig.data) == 0
    assert len(fig.layout.annotations) == 1
    assert "Unsupported shape" in fig.layout.annotations[0].text


def test_compute_figure_slice_reduces_dim(da_2d: xr.DataArray) -> None:
    # slice channel → 1D, then plot
    fig, _ = _compute_figure(da_2d, {"channel": "ch1"}, "freq", None, "line", "none", "none")
    assert len(fig.data) == 1


def test_compute_figure_x_transform_applied(da_1d: xr.DataArray) -> None:
    fig, _ = _compute_figure(da_1d, {}, "freq", None, "line", "log10", "none")
    x_vals = np.array(fig.data[0].x)
    expected = np.log10([1.0, 2.0, 3.0])
    np.testing.assert_allclose(x_vals, expected)


def test_compute_figure_y_transform_applied(da_1d: xr.DataArray) -> None:
    fig, _ = _compute_figure(da_1d, {}, "freq", None, "line", "none", "square")
    y_vals = np.array(fig.data[0].y)
    expected = np.square([1.0, 4.0, 9.0])
    np.testing.assert_allclose(y_vals, expected)


def test_compute_figure_debug_json_keys(da_1d: xr.DataArray) -> None:
    _, debug = _compute_figure(da_1d, {}, "freq", None, "line", "none", "none")
    info = json.loads(debug)
    for key in ("slice_dict", "xdim", "ydim", "plot_type", "arr_shape", "arr_dims"):
        assert key in info


def test_compute_figure_debug_reflects_selections(da_2d: xr.DataArray) -> None:
    _, debug = _compute_figure(
        da_2d, {"channel": "ch1"}, "freq", None, "bar", "log10", "square"
    )
    info = json.loads(debug)
    assert info["xdim"] == "freq"
    assert info["plot_type"] == "bar"
    assert info["x_transform"] == "log10"
    assert info["slice_dict"] == {"channel": "ch1"}


# ---------------------------------------------------------------------------
# build_visualization_app
# ---------------------------------------------------------------------------


def _find_component(layout: Any, component_id: str | dict) -> Any:
    """DFS search for a component by id in a Dash layout tree."""
    if hasattr(layout, "id") and layout.id == component_id:
        return layout
    children = getattr(layout, "children", None)
    if children is None:
        return None
    if not isinstance(children, list):
        children = [children]
    for child in children:
        found = _find_component(child, component_id)
        if found is not None:
            return found
    return None


def test_app_returns_dash_instance(da_1d: xr.DataArray) -> None:
    from dash import Dash

    app = build_visualization_app(da_1d, "demo.nc")
    assert isinstance(app, Dash)


def test_app_dataarray_normalised_to_single_var(da_1d: xr.DataArray) -> None:
    app = build_visualization_app(da_1d, "demo.nc")
    selector = _find_component(app.layout, "variable-selector")
    assert selector is not None
    assert len(selector.options) == 1


def test_app_dataarray_unnamed_defaults_to_data(da_1d: xr.DataArray) -> None:
    app = build_visualization_app(da_1d, "demo.nc")
    selector = _find_component(app.layout, "variable-selector")
    assert selector.options[0]["value"] == "data"


def test_app_dataarray_named_uses_name() -> None:
    da = xr.DataArray([1.0, 2.0], dims=("x",), name="myvar")
    app = build_visualization_app(da, "demo.nc")
    selector = _find_component(app.layout, "variable-selector")
    assert selector.options[0]["value"] == "myvar"


def test_app_dataset_multivar_selector(dataset_multi: xr.Dataset) -> None:
    app = build_visualization_app(dataset_multi, "demo.nc")
    selector = _find_component(app.layout, "variable-selector")
    option_values = {o["value"] for o in selector.options}
    assert option_values == {"spectrum", "bandpower"}


def test_app_has_dim_controls_container(da_2d: xr.DataArray) -> None:
    app = build_visualization_app(da_2d, "demo.nc")
    container = _find_component(app.layout, "dim-controls")
    assert container is not None


def test_app_initial_xdim_is_last_dim(da_2d: xr.DataArray) -> None:
    app = build_visualization_app(da_2d, "demo.nc")
    x_dim = _find_component(app.layout, "x-dim")
    assert x_dim.value == "freq"


def test_app_has_plot_and_debug_components(da_1d: xr.DataArray) -> None:
    app = build_visualization_app(da_1d, "demo.nc")
    assert _find_component(app.layout, "plot") is not None
    assert _find_component(app.layout, "debug-output") is not None


def test_app_dataset_single_var(da_1d: xr.DataArray) -> None:
    ds = da_1d.to_dataset(name="power")
    app = build_visualization_app(ds, "demo.nc")
    selector = _find_component(app.layout, "variable-selector")
    assert len(selector.options) == 1
    assert selector.options[0]["value"] == "power"


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def test_main_runs_with_dataarray_nc(tmp_path: pytest.TempPathFactory, da_1d: xr.DataArray) -> None:
    nc = tmp_path / "demo.nc"
    da_1d.to_netcdf(nc)
    with patch("dash.Dash.run") as run:
        assert main([str(nc), "--port", "9999"]) == 0
        run.assert_called_once()


def test_main_runs_with_dataset_nc(
    tmp_path: pytest.TempPathFactory, dataset_multi: xr.Dataset
) -> None:
    nc = tmp_path / "multi.nc"
    dataset_multi.to_netcdf(nc)
    with patch("dash.Dash.run") as run:
        assert main([str(nc)]) == 0
        run.assert_called_once()


def test_main_rejects_bad_suffix(tmp_path: pytest.TempPathFactory) -> None:
    bad = tmp_path / "bad.txt"
    bad.write_text("x")
    with pytest.raises(ValueError, match="Expected \\.fif or \\.nc"):
        main([str(bad)])
