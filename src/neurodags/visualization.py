"""Dash-based explorer for NeuroDAGs .nc and .fif outputs."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
import xarray as xr
from dash import ALL, Dash, Input, Output, callback_context, dcc, html

from neurodags.loaders import load_meeg
from neurodags.nodes.descriptive import meeg_to_xarray


def load_visualization_dataset(filename: str | Path) -> xr.DataArray | xr.Dataset:
    """Load a supported visualization input into an xarray DataArray or Dataset."""
    path = Path(filename)
    suffix = path.suffix.lower()

    if suffix == ".fif":
        meeg = load_meeg(path)
        return meeg_to_xarray(meeg).artifacts[".nc"].item

    if suffix == ".nc":
        try:
            return xr.open_dataarray(path)
        except ValueError:
            pass
        try:
            return xr.open_dataset(path)
        except Exception as exc:
            raise ValueError(
                f"Failed to open {path} as xarray DataArray or Dataset: {exc}"
            ) from exc

    raise ValueError(f"Unsupported file type '{path.suffix}'. Expected .fif or .nc")


def make_dropdown(name: str, options: Any) -> dcc.Dropdown:
    """Build a dimension-value dropdown."""
    return dcc.Dropdown(
        id=f"dropdown-{name}",
        options=[{"label": str(option), "value": str(option)} for option in options],
        value=str(options[0]),
        clearable=False,
    )


def safe_sel(arr: xr.DataArray, slice_dict: dict[str, str]) -> xr.DataArray:
    """Select values from a DataArray with numeric nearest-neighbor fallback."""
    for dim, val in slice_dict.items():
        coord = arr.coords[dim].values
        if np.issubdtype(coord.dtype, np.number) and np.all(np.diff(coord) >= 0):
            arr = arr.sel({dim: float(val)}, method="nearest")
        else:
            arr = arr.sel({dim: val})
    return arr


def apply_transform(data: Any, transform: str) -> np.ndarray:
    """Apply a display transform to plotted data."""
    arr = np.array(data)

    if not np.issubdtype(arr.dtype, np.number):
        return arr

    if transform == "log10":
        return np.log10(np.where(arr > 0, arr, np.nan))
    if transform == "square":
        return np.square(arr)
    if transform == "log20":
        return np.log10(np.where(arr**2 > 0, arr**2, np.nan))
    return arr


def _make_dim_controls(dims: list[str], coords: dict[str, Any]) -> list:
    """Build per-dimension value selector dropdowns using pattern-match IDs."""
    children: list = []
    for dim in dims:
        children.append(html.Label(dim))
        children.append(
            dcc.Dropdown(
                id={"type": "dim-dropdown", "index": dim},
                options=[{"label": str(v), "value": str(v)} for v in coords[dim]],
                value=str(coords[dim][0]),
                clearable=False,
            )
        )
    return children


def _controls_for_variable(
    da: xr.DataArray,
) -> tuple[list, list[dict], str | None, list[dict], str]:
    """Return (dim_children, x_options, x_value, y_options, y_value) for a DataArray."""
    dims = list(da.dims)
    coords = {dim: da.coords[dim].values for dim in dims}
    children = _make_dim_controls(dims, coords)
    dim_opts = [{"label": d, "value": d} for d in dims]
    none_opt = [{"label": "None", "value": "none"}]
    x_val = dims[-1] if dims else None
    return children, dim_opts, x_val, none_opt + dim_opts, "none"


def _compute_figure(
    da: xr.DataArray,
    slice_dict: dict[str, str],
    xdim: str,
    ydim: str | None,
    plot_type: str,
    x_transform: str,
    y_transform: str,
) -> tuple[go.Figure, str]:
    """Return (figure, debug_json) for the given selections."""
    arr = safe_sel(da, slice_dict)
    fig = go.Figure()

    if ydim is None and arr.ndim == 1:
        x_t = apply_transform(arr.coords[xdim].values, x_transform)
        y_t = apply_transform(arr.values, y_transform)
        if plot_type == "line":
            fig.add_trace(go.Scatter(x=x_t, y=y_t, mode="lines"))
        elif plot_type == "scatter":
            fig.add_trace(go.Scatter(x=x_t, y=y_t, mode="markers"))
        elif plot_type == "bar":
            fig.add_trace(go.Bar(x=x_t, y=y_t))
    elif ydim is not None and arr.ndim == 2:
        xvals = apply_transform(arr.coords[xdim].values, x_transform)
        yvals = apply_transform(arr.coords[ydim].values, y_transform)
        if plot_type == "heatmap":
            fig.add_trace(go.Heatmap(x=xvals, y=yvals, z=arr.values))
        else:
            for val in arr.coords[ydim].values:
                sl = arr.sel({ydim: val})
                x_t = apply_transform(sl.coords[xdim].values, x_transform)
                y_t = apply_transform(sl.values, y_transform)
                fig.add_trace(go.Scatter(x=x_t, y=y_t, mode="lines", name=f"{ydim}={val}"))
    else:
        fig.add_annotation(
            text=f"Unsupported shape after slicing: {arr.shape}, ndim={arr.ndim}",
            x=0.5,
            y=0.5,
            showarrow=False,
        )

    debug_info = {
        "slice_dict": {key: str(value) for key, value in slice_dict.items()},
        "xdim": str(xdim),
        "ydim": str(ydim),
        "plot_type": plot_type,
        "x_transform": x_transform,
        "y_transform": y_transform,
        "arr_shape": tuple(int(x) for x in arr.shape),
        "arr_dims": [str(dim) for dim in arr.dims],
        "metadata": {key: str(value) for key, value in arr.attrs.items()},
        "coords": {key: str(value.values) for key, value in arr.coords.items()},
        "dimensions": {key: str(value) for key, value in arr.sizes.items()},
    }
    return fig, json.dumps(debug_info, indent=2)


def build_visualization_app(ds: xr.DataArray | xr.Dataset, filename: str | Path) -> Dash:
    """Create the Dash app for a loaded DataArray or Dataset."""
    if isinstance(ds, xr.DataArray):
        name = ds.name or "data"
        dataset = ds.to_dataset(name=name)
    else:
        dataset = ds

    var_names = list(dataset.data_vars)
    initial_var = var_names[0]
    initial_da = dataset[initial_var]

    dim_children, x_opts, x_val, y_opts, _ = _controls_for_variable(initial_da)

    app = Dash(__name__)
    app.layout = html.Div(
        [
            html.H2(f"Explorer for {filename}"),
            html.Label("Variable"),
            dcc.Dropdown(
                id="variable-selector",
                options=[{"label": v, "value": v} for v in var_names],
                value=initial_var,
                clearable=False,
            ),
            html.Div(dim_children, id="dim-controls"),
            html.Label("X-axis dimension"),
            dcc.Dropdown(id="x-dim", options=x_opts, value=x_val, clearable=False),
            html.Label("Y-axis dimension (optional)"),
            dcc.Dropdown(id="y-dim", options=y_opts, value="none", clearable=False),
            html.Label("Plot type"),
            dcc.Dropdown(
                id="plot-type",
                options=[
                    {"label": "Line", "value": "line"},
                    {"label": "Scatter (points)", "value": "scatter"},
                    {"label": "Bar", "value": "bar"},
                    {"label": "Heatmap (2D only)", "value": "heatmap"},
                ],
                value="line",
                clearable=False,
            ),
            html.Label("X-axis transform"),
            dcc.Dropdown(
                id="x-transform",
                options=[
                    {"label": "None", "value": "none"},
                    {"label": "Log10", "value": "log10"},
                    {"label": "Square", "value": "square"},
                    {"label": "Log20", "value": "log20"},
                ],
                value="none",
                clearable=False,
            ),
            html.Label("Y-axis transform"),
            dcc.Dropdown(
                id="y-transform",
                options=[
                    {"label": "None", "value": "none"},
                    {"label": "Log10", "value": "log10"},
                    {"label": "Square", "value": "square"},
                    {"label": "Log20", "value": "log20"},
                ],
                value="none",
                clearable=False,
            ),
            dcc.Graph(id="plot"),
            html.H3("Debug info"),
            html.Pre(
                id="debug-output",
                style={
                    "whiteSpace": "pre-wrap",
                    "border": "1px solid #ccc",
                    "padding": "10px",
                },
            ),
        ]
    )

    @app.callback(
        Output("dim-controls", "children"),
        Output("x-dim", "options"),
        Output("x-dim", "value"),
        Output("y-dim", "options"),
        Output("y-dim", "value"),
        Input("variable-selector", "value"),
    )
    def update_variable_controls(var_name: str) -> tuple:
        da = dataset[var_name]
        return _controls_for_variable(da)

    @app.callback(
        [Output("plot", "figure"), Output("debug-output", "children")],
        Input({"type": "dim-dropdown", "index": ALL}, "value"),
        Input("variable-selector", "value"),
        Input("x-dim", "value"),
        Input("y-dim", "value"),
        Input("plot-type", "value"),
        Input("x-transform", "value"),
        Input("y-transform", "value"),
    )
    def update_plot(
        dim_values: list[str],
        var_name: str,
        xdim: str | None,
        ydim: str,
        plot_type: str,
        x_transform: str,
        y_transform: str,
    ) -> tuple[go.Figure, str]:
        if not xdim or var_name not in dataset:
            return go.Figure(), "{}"

        dim_inputs = callback_context.inputs_list[0]
        dim_names = [item["id"]["index"] for item in dim_inputs]

        if not dim_names:
            return go.Figure(), "{}"

        resolved_ydim = None if ydim == "none" else ydim
        da = dataset[var_name]
        slice_dict = {
            dim: val
            for dim, val in zip(dim_names, dim_values, strict=False)
            if dim not in (xdim, resolved_ydim)
        }
        return _compute_figure(
            da, slice_dict, xdim, resolved_ydim, plot_type, x_transform, y_transform
        )

    return app


def build_parser() -> argparse.ArgumentParser:
    """Build the visualization CLI parser."""
    parser = argparse.ArgumentParser(description="Launch the NeuroDAGs Dash explorer.")
    parser.add_argument("path", help="Path to a .nc or .fif file.")
    parser.add_argument("--host", default="127.0.0.1", help="Dash host to bind to.")
    parser.add_argument("--port", type=int, default=8050, help="Dash port to bind to.")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run the Dash server with debug mode enabled.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for the Dash explorer."""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    ds = load_visualization_dataset(args.path)
    app = build_visualization_app(ds, args.path)
    app.run(host=args.host, port=args.port, debug=args.debug)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
