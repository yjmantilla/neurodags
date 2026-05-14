# Inspection, Visualization, and Parallelism

## Dry Run Mode

Inspect planned computations without executing any nodes. Returns a dataframe describing what would run and whether cached outputs already exist.

```python
from neurodags.loaders import load_configuration
from neurodags.orchestrators import iterate_derivative_pipeline

config = load_configuration("pipeline.yml")
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

## Error Markers

When a node fails, NeuroDAGs writes a `.error` marker file alongside the expected output. On subsequent runs, the presence of this marker prevents silent retries — you must delete or handle it explicitly before retrying that file.

This avoids wasting compute on repeatedly failing files in large batches.

## Visualization

### DAG Visualization (Mermaid)

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

See the {doc}`../auto_examples/plot_mermaid_visualization` example for a
complete walkthrough.

### Interactive File Explorer

Built-in Dash-Plotly explorer for `.fif` (MNE) and `.nc` (NetCDF/xarray) files:

```bash
neurodags view path/to/file.fif
neurodags view path/to/file.nc

# Alternative module entry point
python -m neurodags.visualization path/to/file.fif
python -m neurodags.visualization path/to/file.nc
```

Features:
- Dimension-aware UI — dropdown to select which axis to plot along
- Plot types: Line, Scatter, Bar, Heatmap
- Works with any `.nc` file produced by NeuroDAGs, regardless of array shape

## Parallel Execution

NeuroDAGs uses joblib for file-level parallelism. Each file is an independent job.

### Via pipeline.yml

```yaml
n_jobs: 4           # -1 = all available cores, 1 or null = serial
joblib_backend: loky
joblib_prefer: processes
```

### Via Python

```python
iterate_derivative_pipeline(
    config,
    "MyDerivative",
    n_jobs=4,
    joblib_backend="loky",
    joblib_prefer="processes",
)
```

### Via CLI

```bash
neurodags run pipeline.yml --derivative MyDerivative --n-jobs 4 --joblib-backend loky
```

## CLI DAG Export

Generate Mermaid output directly from the command line:

```bash
neurodags dag pipeline.yml
neurodags dag pipeline.yml --html pipeline_dag.html
neurodags dag pipeline.yml --derivative BandPower --html bandpower_dag.html
```

## Subset Execution

Process only a subset of files by index:

```python
# Single file
iterate_derivative_pipeline(config, "MyDerivative", only_index=0)

# Multiple files
iterate_derivative_pipeline(config, "MyDerivative", only_index=[0, 2, 5])

# Limit files per dataset
iterate_derivative_pipeline(config, "MyDerivative", max_files_per_dataset=10)
```

## Error Handling

By default, errors are caught per-file and logged; processing continues. To stop on the first failure:

```python
iterate_derivative_pipeline(config, "MyDerivative", raise_on_error=True)
```

## HPC Tips

1. Set `mount_point: hpc` in `pipeline.yml` for cluster paths. If needed, override it by modifying the config dict before calling the API.
2. Use `n_jobs: -1` to use all available cores on a compute node.
3. Submit separate jobs per derivative to exploit SLURM array jobs — each job processes all files for one derivative.
4. Caching ensures that if a job partially completes, re-running only processes remaining files.
