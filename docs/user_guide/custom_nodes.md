# Custom Nodes

NeuroDAGs lets you define new computation nodes without modifying or forking the package. Custom nodes are registered at runtime via a Python module pointed to by `new_definitions` in `pipeline.yml`.

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
