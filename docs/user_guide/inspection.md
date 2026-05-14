# Inspection and Visualization

## Dry Run Mode

Inspect planned computations without executing any nodes. Returns a dataframe describing what would run and whether cached outputs already exist.

```python
from neurodags.loaders import load_configuration
from neurodags.orchestrators import iterate_derivative_pipeline, run_pipeline

config = load_configuration("pipeline.yml")

# Dry-run all derivatives in DerivativeList
plan = run_pipeline(config, dry_run=True)
print(plan)

# Or dry-run a single derivative
plan = iterate_derivative_pipeline(config, "BandPower", dry_run=True)
print(plan)
```

CLI equivalent:

```bash
neurodags dry-run pipeline.yml --derivative BandPower --output dry_run_results.csv
```

The dry-run plan shows each step, whether its output is cached, and where it would write. Use this to:

- Verify the DAG structure before a long run
- Identify which files need recomputation
- Debug path resolution issues

## Failure Behavior

When a node fails during execution:

1. The error is logged with full traceback, derivative name, step id, and file path.
2. A `.error` marker file is written at the expected output location (e.g. `sub-01@MyDerivative.error`). Its content is a human-readable summary of the failure — useful for post-hoc inspection without digging through logs.
3. By default (`raise_on_error=False`) execution continues to the next file; all failures are collected in the log.
4. The failed file **will be retried** on the next run — the `.error` file is excluded from the cached-artifact check, so NeuroDAGs treats the output as missing and re-runs it.

To inspect which files failed after a run, look for `*.error` files in the derivatives directory:

```bash
find derivatives/ -name "*.error"
```

Each `.error` file contains:

```
Derivative 'MyDerivative' step id=2 node='my_node' failed:
<exception message>
```

### Automatic Cleanup of Stale Error Markers

When a file that previously failed is re-run and **succeeds**, NeuroDAGs automatically deletes the stale `.error` marker. This keeps the derivatives directory clean and ensures dry-run plans accurately reflect current state — a file that succeeded on re-run will not appear as `has_error_marker: true` on subsequent runs.

### Skipping Previously Failed Files

To skip files that already have a `.error` marker (and avoid retrying known failures), use `skip_errors=True`:

```python
run_pipeline(config, skip_errors=True)
```

CLI:

```bash
neurodags run pipeline.yml --skip-errors
neurodags dry-run pipeline.yml --skip-errors   # shows which files would be skipped
```

Skipped files are logged and reported in the dry-run plan with `has_error_marker: true`. To retry a skipped file, delete its `.error` marker:

```bash
rm derivatives/sub-01/ses-SE0/sub-01_ses-SE0_task-rest.vhdr@MyDerivative.error
```

## DAG Visualization (Mermaid)

NeuroDAGs can render pipeline and derivative graphs as interactive
[Mermaid](https://mermaid.js.org/) diagrams saved to standalone HTML files.

**Pipeline-level overview** — one node per derivative, edges show inter-derivative
dependencies:

```python
import yaml
from neurodags.mermaid import pipeline_to_html

with open("pipeline.yml") as f:
    config = yaml.safe_load(f)

pipeline_to_html(config, output_path="pipeline_dag.html", auto_open=True)
```

**Derivative-level detail** — every computation node and data reference inside
one derivative:

```python
from neurodags.mermaid import derivative_to_html

derivative_to_html(
    config["DerivativeDefinitions"]["BandPower"],
    "BandPower",
    output_path="bandpower_dag.html",
    auto_open=True,
)
```

Node shapes used in derivative diagrams:

| Shape | Meaning |
|-------|---------|
| Circle `(((...)))` | `SourceFile` — raw input file |
| Cylinder `[(...)]` | Upstream derivative artifact (cached on disk) |
| Rectangle `[...]` | Computation node |

To get the raw Mermaid string (e.g. for embedding in Jupyter or custom HTML):

```python
from neurodags.mermaid import pipeline_to_mermaid, derivative_to_mermaid

print(pipeline_to_mermaid(config))
print(derivative_to_mermaid(config["DerivativeDefinitions"]["BandPower"], "BandPower"))
```

CLI:

```bash
neurodags dag pipeline.yml
neurodags dag pipeline.yml --html pipeline_dag.html
neurodags dag pipeline.yml --derivative BandPower --html bandpower_dag.html
```

See the {doc}`../auto_examples/plot_mermaid_visualization` example for a
complete walkthrough.

## Interactive File Explorer

Built-in Dash-Plotly explorer for `.fif` (MNE) and `.nc` (NetCDF/xarray) files:

```bash
neurodags view path/to/file.fif
neurodags view path/to/file.nc

# Alternative module entry point
python -m neurodags.visualization path/to/file.fif
python -m neurodags.visualization path/to/file.nc
```

Features:
- Variable selector — when the `.nc` file is a multi-variable Dataset, a dropdown lets you switch between variables interactively without restarting
- Dimension-aware UI — per-variable dropdowns to slice along any dimension
- Plot types: Line, Scatter, Bar, Heatmap
- Works with both `xr.DataArray` and `xr.Dataset` files produced by NeuroDAGs
