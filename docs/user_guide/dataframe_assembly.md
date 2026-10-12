# Dataframe Assembly

NeuroDAGs can aggregate derivative artifacts across all files into a single dataframe — ready for statistical analysis or ML pipelines. It leverages xarray metadata to handle dimensions and coordinates automatically.

## Marking Derivatives for Inclusion

Set `for_dataframe: True` on any derivative you want collected:

```yaml
BandPower:
  for_dataframe: True
  nodes:
    - id: 0
      derivative: PowerSpectrum.nc
    - id: 1
      node: bandpower
      args:
        psd_like: id.0
        bands:
          alpha: [8.0, 13.0]
          beta: [13.0, 30.0]
```

## Building the Dataframe

```python
from neurodags.orchestrators import build_derivative_dataframe

df = build_derivative_dataframe("pipeline.yml", output_format="wide")
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pipeline_configuration` | dict or str | — | Pipeline config dict or path to YAML |
| `include_derivatives` | list[str] | None | Restrict to specific derivative names |
| `max_files_per_dataset` | int | None | Limit files per dataset |
| `only_index` | int or list[int] | None | Process only specific file indices |
| `output_format` | `"wide"` or `"long"` | `"wide"` | Shape of output dataframe |
| `preserve_complex_values` | bool | False | Keep nested structures instead of flattening |
| `raise_on_error` | bool | False | Re-raise exceptions instead of skipping |

## Output Formats

### Wide Format (`output_format="wide"`)

One row per source file. Each collected derivative becomes one or more columns.
`file_path` remains the original input file path; derivative identity is encoded
in the column names. Best for file-level features.

```
file_path               BandPower.nc@freqbands-alpha  BandPower.nc@freqbands-beta  Entropy.json
sub-01_task-rest.vhdr   0.32                          0.18                         1.24
sub-02_task-rest.vhdr   0.27                          0.21                         1.11
```

Columns are named `<derivative name><artifact suffix>`, using the derivative's YAML name (not its
on-disk CamelCase name). A `.nc` `DataArray` is flattened into one column per element, adding
`@<dim>-<value>_<dim>-<value>...` (dimensions sorted alphabetically, coordinate values escaped:
`-`→`~`, `_`→`|`, `@`→`$`, `,`→`.`). See
{ref}`Dataframe column names <naming-dataframe-columns>` for how to read a column back.

### Long Format (`output_format="long"`)

One row per collected value. `file_path` still refers to the source file, while
the collected derivative value is identified in the `derivative` column. No
column synthesis is performed. Best for multi-dimensional derivatives.

```
file_path               derivative                     value
sub-01_task-rest.vhdr   BandPower.nc@freqbands-alpha   0.32
sub-01_task-rest.vhdr   BandPower.nc@freqbands-beta    0.18
sub-01_task-rest.vhdr   Entropy.json                   1.24
sub-02_task-rest.vhdr   BandPower.nc@freqbands-alpha   0.27
```

## Selecting Derivatives

Only collect specific derivatives (must have `for_dataframe: True`):

```python
df = build_derivative_dataframe(
    "pipeline.yml",
    include_derivatives=["BandPower", "SpectralEntropy"],
    output_format="wide",
)
```

## With `save: False`

Derivatives with `save: False` are computed but not written to disk. They can still be marked `for_dataframe: True` — their artifacts are collected in memory during assembly.

```yaml
BandPowerMean:
  save: False
  for_dataframe: True
  nodes:
    - id: 0
      derivative: BandPower.nc
    - id: 1
      node: aggregate_across_dimension
      args:
        xarray_data: id.0
        dim: epochs
        operation: mean
```

## Practical Pattern: Compute → Aggregate → Collect

A common pattern for EEG features:

1. Compute a full derivative (e.g. per-epoch band power) with `save: True`.
2. Aggregate across epochs with `save: False, for_dataframe: True`.
3. Call `build_derivative_dataframe` to get one row per file.

```yaml
BandPower:
  for_dataframe: False
  nodes: ...          # compute per-epoch band power

BandPowerMean:
  save: False
  for_dataframe: True
  nodes:
    - id: 0
      derivative: BandPower.nc
    - id: 1
      node: aggregate_across_dimension
      args:
        xarray_data: id.0
        dim: epochs
        operation: mean
```

### Non-finite values when aggregating

`aggregate_across_dimension` reduces with xarray's default `skipna=True`, so NaN values
along the aggregated dimension (e.g. epochs where a feature failed to compute) are dropped
from the reduction. By default the node logs a warning with the dropped count; two `args`
control this:

```yaml
    - id: 1
      node: aggregate_across_dimension
      args:
        xarray_data: id.0
        dim: epochs
        operation: mean
        on_dropped: raise      # "warn" (default) | "raise" (fail fast) | "ignore"
        emit_counts: true      # attach n_used / n_dropped coords per aggregated value
```

Use `on_dropped: raise` when a derivative must aggregate complete data, and
`emit_counts: true` to carry the per-value `n_used`/`n_dropped` counts into the dataframe
so the reliability of each aggregate is queryable.
