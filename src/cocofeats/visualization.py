import sys
import xarray as xr
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output
import webbrowser
import json
import numpy as np
from cocofeats.loaders import load_meeg
from cocofeats.nodes.descriptive import meeg_to_xarray
# ---------- Load Data ----------
filename = sys.argv[1]   # <--- uncomment to accept from command line


if filename.endswith('.fif'):
    meeg = load_meeg(filename)
    ds = meeg_to_xarray(meeg).artifacts['.nc'].item

if filename.endswith('.nc'):
    try:
        ds = xr.open_dataarray(filename)
    except Exception as e:
        raise ValueError(f"Failed to open {filename} as xarray DataArray: {e}")

dims = list(ds.dims)
coords = {dim: ds.coords[dim].values for dim in dims}

# ---------- Build App ----------
app = Dash(__name__)

def make_dropdown(name, options):
    return dcc.Dropdown(
        id=f"dropdown-{name}",
        options=[{"label": str(o), "value": str(o)} for o in options],
        value=str(options[0]),
        clearable=False,
    )

# Dropdowns for dimension values
value_dropdowns = []
for dim in dims:
    value_dropdowns.append(html.Label(dim))
    value_dropdowns.append(make_dropdown(dim, coords[dim]))

app.layout = html.Div([
    html.H2(f"Explorer for {filename}"),
    html.Div(value_dropdowns, style={"marginBottom": "20px"}),

    html.Label("X-axis dimension"),
    dcc.Dropdown(
        id="x-dim",
        options=[{"label": d, "value": d} for d in dims],
        value=dims[-1],
        clearable=False
    ),

    html.Label("Y-axis dimension (optional)"),
    dcc.Dropdown(
        id="y-dim",
        options=[{"label": "None", "value": "none"}] + [{"label": d, "value": d} for d in dims],
        value="none",
        clearable=False
    ),

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
        clearable=False
    ),

    html.Label("X-axis transform"),
    dcc.Dropdown(
        id="x-transform",
        options=[
            {"label": "None", "value": "none"},
            {"label": "Log10", "value": "log10"},
            {"label": "Square", "value": "square"},
            {"label": "Log20", "value": "log20"}
        ],
        value="none",
        clearable=False
    ),

    html.Label("Y-axis transform"),
    dcc.Dropdown(
        id="y-transform",
        options=[
            {"label": "None", "value": "none"},
            {"label": "Log10", "value": "log10"},
            {"label": "Square", "value": "square"},
            {"label": "Log20", "value": "log20"}
        ],
        value="none",
        clearable=False
    ),

    dcc.Graph(id="plot"),

    html.H3("Debug info"),
    html.Pre(id="debug-output", style={"whiteSpace": "pre-wrap", "border": "1px solid #ccc", "padding": "10px"})
])

# ---------- Utils ----------

def safe_sel(arr, slice_dict):
    for dim, val in slice_dict.items():
        coord = arr.coords[dim].values
        if np.issubdtype(coord.dtype, np.number) and np.all(np.diff(coord) >= 0):
            arr = arr.sel({dim: float(val)}, method="nearest")
        else:
            arr = arr.sel({dim: val})
    return arr

def apply_transform(data, transform):
    # Convert to numpy array
    arr = np.array(data)

    # If not numeric, skip transforms
    if not np.issubdtype(arr.dtype, np.number):
        return arr

    if transform == "log10":
        return np.log10(np.where(arr > 0, arr, np.nan))
    elif transform == "square":
        return np.square(arr)
    elif transform == "log20":
        return np.log10(np.where(arr**2 > 0, arr**2, np.nan))  # log10(x^2)
    return arr

# ---------- Callbacks ----------

@app.callback(
    [Output("plot", "figure"),
     Output("debug-output", "children")],
    [Input(f"dropdown-{dim}", "value") for dim in dims] +
    [Input("x-dim", "value"),
     Input("y-dim", "value"),
     Input("plot-type", "value"),
     Input("x-transform", "value"),
     Input("y-transform", "value")]
)
def update_plot(*vals):
    values = vals[:len(dims)]
    xdim, ydim, plot_type, x_transform, y_transform = vals[len(dims):]
    if ydim == "none":
        ydim = None
    # freeze all dims except xdim, ydim
    slice_dict = {dim: val for dim, val in zip(dims, values) if dim not in (xdim, ydim)}
    arr = safe_sel(ds, slice_dict)

    fig = go.Figure()

    # 1D: just one axis
    if ydim is None and arr.ndim == 1:
        x_t = apply_transform(arr.coords[xdim].values, x_transform)
        y_t = apply_transform(arr.values, y_transform)
        if plot_type == "line":
            fig.add_trace(go.Scatter(x=x_t, y=y_t, mode="lines"))
        elif plot_type == "scatter":
            fig.add_trace(go.Scatter(x=x_t, y=y_t, mode="markers"))
        elif plot_type == "bar":
            fig.add_trace(go.Bar(x=x_t, y=y_t))

    # 2D: two free dims
    elif ydim is not None and arr.ndim == 2:
        xvals = apply_transform(arr.coords[xdim].values, x_transform)
        yvals = apply_transform(arr.coords[ydim].values, y_transform)
        if plot_type == "heatmap":
            fig.add_trace(go.Heatmap(x=xvals, y=yvals, z=arr.values))
        else:
            # plot each slice across ydim
            for val in arr.coords[ydim].values:
                sl = arr.sel({ydim: val})
                x_t = apply_transform(sl.coords[xdim].values, x_transform)
                y_t = apply_transform(sl.values, y_transform)
                fig.add_trace(go.Scatter(x=x_t, y=y_t, mode="lines", name=f"{ydim}={val}"))

    else:
        fig.add_annotation(
            text=f"Unsupported shape after slicing: {arr.shape}, ndim={arr.ndim}",
            x=0.5, y=0.5, showarrow=False
        )

    debug_info = {
        "slice_dict": {k: str(v) for k, v in slice_dict.items()},
        "xdim": str(xdim),
        "ydim": str(ydim),
        "plot_type": plot_type,
        "x_transform": x_transform,
        "y_transform": y_transform,
        "arr_shape": tuple(int(x) for x in arr.shape),
        "arr_dims": [str(d) for d in arr.dims],
        "metadata": {k: str(v) for k, v in arr.attrs.items()},
        "coords": {k: str(v.values) for k, v in arr.coords.items()},
        "dimensions": {k: str(v) for k, v in arr.sizes.items()},
    }

    return fig, json.dumps(debug_info, indent=2)

# ---------- Launch ----------
if __name__ == "__main__":
    # webbrowser.open("http://127.0.0.1:8050")   # <--- uncomment to auto-open browser
    app.run(debug=True)
