# Custom Nodes

NeuroDAGs lets you define new computation nodes without modifying or forking the package. Custom nodes are registered at runtime via a Python module pointed to by `new_definitions` in `pipeline.yml`.

The built-in nodes target EEG/MEG/ECG via MNE-Python and xarray, but the node system is domain-agnostic. If your data comes in files — audio, fMRI volumes, tabular CSVs, images, genomics — you can use NeuroDAGs orchestration with entirely custom nodes and loaders. The caching, dependency ordering, HPC templates, and dataframe assembly work identically regardless of what the nodes do.

## Key Ideas

1. A node is a Python function.
2. It returns a `NodeResult`.
3. A `NodeResult` contains `artifacts` — a dict mapping file extension to `Artifact(item, writer)`.

## Minimal Example

```python
# custom_nodes.py
from neurodags.nodes import register_node
from neurodags.definitions import Artifact, NodeResult

@register_node
def my_node(data) -> NodeResult:
    result = compute(data)
    return NodeResult(
        artifacts={
            ".nc": Artifact(
                item=result,
                writer=lambda path: result.to_netcdf(path),
            ),
        },
    )
```

## Multiple Artifacts

A node can produce more than one artifact:

```python
@register_node
def my_node_with_report(data) -> NodeResult:
    result = compute(data)
    report = make_html_report(result)

    return NodeResult(
        artifacts={
            ".nc": Artifact(
                item=result,
                writer=lambda path: result.to_netcdf(path),
            ),
            ".report.html": Artifact(
                item=report,
                writer=lambda path: report.save(path),
            ),
        },
    )
```

## Custom Node Name

By default the node is registered under the function name. Override it:

```python
@register_node(name="my_custom_name", override=True)
def _internal_function_name(data) -> NodeResult:
    ...
```

`override=True` allows re-registering a name (e.g. in tests or hot-reload scenarios).

## Artifact Types

The `item` field holds the in-memory object; `writer` is a callable that receives a full file path and saves it. Common patterns:

```python
# xarray DataArray → NetCDF (recommended for numerical results)
".nc": Artifact(item=da, writer=lambda path: da.to_netcdf(path))

# MNE object → FIF
".fif": Artifact(item=raw, writer=lambda path: raw.save(path, overwrite=True))

# Pandas DataFrame → CSV
".csv": Artifact(item=df, writer=lambda path: df.to_csv(path))

# Any object → pickle
".pkl": Artifact(item=obj, writer=lambda path: pickle.dump(obj, open(path, "wb")))
```

## xarray Convention

For numerical derivatives, prefer returning an `xr.DataArray` or `xr.Dataset` saved as `.nc`. This enables:

- Dimension-aware caching
- Automatic dataframe assembly via `build_derivative_dataframe`
- Visualization in the built-in Dash-Plotly explorer
- Language-agnostic downstream analysis

## Registering in pipeline.yml

```yaml
new_definitions: custom_nodes.py

# or multiple files:
new_definitions:
  - custom_nodes/nodes_a.py
  - /abs/path/to/nodes_b.py
```

Paths relative to the pipeline YAML. Each module is imported once before any derivative executes.

### Keep definition files cheap to import

Definition files are imported in every process that runs derivatives, including each parallel
worker (`--n-jobs`) and each SLURM array task. Import heavy third-party packages inside the node
function rather than at module level, so processes that never call the node do not pay for the
import:

```python
@register_node
def my_entropy(data) -> NodeResult:
    import antropy  # heavy (numba JIT at import); only paid when this node runs

    ...
```

The built-in nodes follow the same convention: `antropy`, `neurokit2`, `fooof` and
`matplotlib` are imported the first time a node that needs them executes.

## Splitting Definitions Across Files

For large projects, split nodes by domain (e.g. `nodes_preprocessing.py`, `nodes_features.py`, `nodes_cleaning.py`) and list all files under `new_definitions`.

**Cross-file imports require one extra step.** Definition files are loaded via `importlib.util.spec_from_file_location` with a unique synthetic module name, so they are not automatically findable by a plain `import`. If one definition file needs to import helpers from another, the shared file must add its own directory to `sys.path`:

```python
# nodes_utils.py  — shared helpers
import os
import sys

# Make this directory importable by other definition files
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def _to_nc_writer(da):
    return lambda path, obj=da: obj.to_netcdf(path)
```

Then any other definition file loaded after it can import normally:

```python
# nodes_features.py
import nodes_utils

_to_nc_writer = nodes_utils._to_nc_writer

@register_node
def my_feature(data) -> NodeResult:
    result = compute(data)
    return NodeResult(artifacts={".nc": Artifact(item=result, writer=_to_nc_writer(result))})
```

```yaml
new_definitions:
  - nodes_utils.py      # must come first — adds directory to sys.path
  - nodes_features.py
  - nodes_cleaning.py
```

The shared file must appear first in `new_definitions` so `sys.path` is patched before the files that import from it are loaded.

## Using Custom Nodes in Derivatives

```yaml
DerivativeDefinitions:
  MyResult:
    nodes:
      - id: 0
        derivative: SourceFile
      - id: 1
        node: my_node         # matches the registered name
        args:
          data: id.0
          threshold: 0.5
```

## Signaling That a Derivative Does Not Apply: `SkipDerivative`

Sometimes a source file is legitimately not processable by a given derivative — not because
of a bug, but because the input simply does not contain the required data. The canonical
example is a multi-condition EEG study where not every subject underwent every condition:
a node that extracts epochs for `HV_EO` (hyperventilation eyes-open) should silently skip
subjects who never performed that task, rather than raising an unhandled error.

Raise `SkipDerivative` from inside a node to signal this intent:

```python
from neurodags.definitions import SkipDerivative, NodeResult, Artifact

@register_node
def extract_condition_features(raw_path: str, condition_name: str) -> NodeResult:
    epochs = load_and_epoch(raw_path, condition_name)
    if epochs is None or len(epochs) == 0:
        raise SkipDerivative(
            f"Condition '{condition_name}' not present in '{raw_path}'."
        )
    result = compute_features(epochs)
    return NodeResult(artifacts={".nc": Artifact(item=result, writer=result.to_netcdf)})
```

**What neurodags does when `SkipDerivative` is raised:**

1. Writes a `.skip` marker file alongside where the artifact would have been saved
   (e.g. `sub-0002@SpectrumWelch.skip`). The exception message is written into the file.
2. Propagates the skip to all parent derivatives that depend on this one — they each write
   their own `.skip` marker without attempting to compute.
3. Reports the derivative as **skipped** in `neurodags status` output (a distinct column,
   separate from *missing* and *errored*).
4. Does **not** retry skipped derivatives on subsequent runs unless the `.skip` file is
   deleted manually or the derivative has `overwrite: true`.

**`SkipDerivative` vs a plain exception:**

| Situation | What to raise |
|---|---|
| Condition genuinely absent from this recording | `SkipDerivative` |
| Missing file / corrupt data / unexpected format | `ValueError` / `RuntimeError` → becomes `.error` |
| Bug in your node code | Let it propagate naturally → becomes `.error` |

**Inspecting skipped files:** the `.skip` marker contains the reason string, so you can
audit which subjects were skipped and why:

```bash
find derivatives/ -name "*.skip" -exec cat {} \;
```

## Built-in Node Registry API

```python
from neurodags.nodes import register_node, get_node, list_nodes, iter_nodes

# List all registered nodes
print(list_nodes())

# Retrieve a node by name
fn = get_node("basic_preprocessing")

# Iterate all (name, function) pairs
for name, fn in iter_nodes():
    print(name, fn)
```

## Using NeuroDAGs Beyond EEG/MEG

Any per-file analysis pipeline works. You need:

1. A glob pattern that discovers your input files (`file_pattern` in `datasets.yml`)
2. Nodes that load and process your data and return a `NodeResult`

The `SourceFile` pseudo-derivative gives each node the raw file path as a string — your loader node reads it however it likes.

Example: CSV time-series files → per-file statistics → assembled dataframe.

```python
# custom_nodes.py
import pandas as pd
import xarray as xr
from neurodags.nodes import register_node
from neurodags.definitions import Artifact, NodeResult


@register_node
def load_csv_timeseries(file_path: str) -> NodeResult:
    df = pd.read_csv(file_path, index_col="time")
    da = xr.DataArray(df.values, dims=("time", "channel"),
                      coords={"time": df.index, "channel": df.columns})
    return NodeResult(
        artifacts={".nc": Artifact(item=da, writer=lambda p: da.to_netcdf(p))}
    )


@register_node
def channel_statistics(data) -> NodeResult:
    da = data.artifacts[".nc"].item
    stats = da.mean("time").to_dataset(name="mean")
    stats["std"] = da.std("time")
    return NodeResult(
        artifacts={".nc": Artifact(item=stats, writer=lambda p: stats.to_netcdf(p))}
    )
```

```yaml
# datasets.yml
my_study:
  name: MyStudy
  source_root: data
  file_pattern: data/**/*.csv
  derivatives_path: derivatives/
```

```yaml
# pipeline.yml
naming_version: 1
new_definitions: custom_nodes.py

DerivativeList:
  - Timeseries
  - Stats

DerivativeDefinitions:
  Timeseries:
    nodes:
      - id: 0
        derivative: SourceFile
      - id: 1
        node: load_csv_timeseries
        args:
          file_path: id.0

  Stats:
    for_dataframe: true
    nodes:
      - id: 0
        derivative: Timeseries.nc
      - id: 1
        node: channel_statistics
        args:
          data: id.0
```

Everything else — caching, dependency ordering, joblib parallelism, SLURM templates, dataframe assembly — works identically to an EEG pipeline.
