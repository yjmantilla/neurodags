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
index  dataset  file_path              BandPower.nc@band-alpha_channel-Fp1  BandPower.nc@band-beta_channel-Fp1  Entropy.json
0      demo     sub-01_task-rest.vhdr  0.32                                 0.18                                1.24
1      demo     sub-02_task-rest.vhdr  0.27                                 0.21                                1.11
```

### Long Format (`output_format="long"`)

One row per collected value, with the columns `index`, `dataset`, `file_path`, `derivative`
and `value`. `derivative` holds the full wide column name, so both formats use the same
identity. No column synthesis is performed. Best for multi-dimensional derivatives.

```
file_path               derivative                           value
sub-01_task-rest.vhdr   BandPower.nc@band-alpha_channel-Fp1  0.32
sub-01_task-rest.vhdr   BandPower.nc@band-beta_channel-Fp1   0.18
sub-01_task-rest.vhdr   Entropy.json                         1.24
sub-02_task-rest.vhdr   BandPower.nc@band-alpha_channel-Fp1  0.27
```

(dataframe-column-names)=

## Column Names

Every generated column is built by {func}`neurodags.naming.format_column`:

```text
<name><artifact suffix>                                   # one column per artifact
<name><artifact suffix>@<dim>-<value>_<dim>-<value>...    # one column per DataArray element
<name>.error                                              # the derivative failed for this file
```

- `<name>` is the derivative's YAML name, verbatim; `<artifact suffix>` is the artifact key
  (`.nc`, `.json`, `.report.html`).
- A `.nc` artifact holding an N-dimensional `xarray.DataArray` is flattened into one column
  per element. Each element contributes one `dim-value` pair per dimension, joined by `_` and
  **sorted by encoded dimension name**. 0-d arrays get no `@` part.
- Dimension names are percent-encoded except `A-Z`, `a-z`, `0-9`; coordinate values are
  percent-encoded except `A-Z`, `a-z`, `0-9`, `.`, `+`, `~`, `-`. Encoding is UTF-8 with
  uppercase hex, so `_`, `@`, `%`, `,`, spaces and non-ASCII characters never appear literally.
  Non-string coordinates are written with `str()` (`1.5`, `-1`, `None`).
- Metadata columns (`index`, `dataset`, `file_path`) never contain `.`; generated columns always
  do, so the two can never collide.

| Coordinate | Column |
|------------|--------|
| `band="alpha"`, `channel="Fp1"` | `BandPower.nc@band-alpha_channel-Fp1` |
| `band="8-12"` | `BandPower.nc@band-8-12` |
| `freq=-1.5` | `Psd.nc@freq--1.5` |
| `channel="EEG 001"` | `Psd.nc@channel-EEG%20001` |
| `cond="a_b"` | `Erp.nc@cond-a%5Fb` |
| dimension `freq_band`, value `alpha` | `Psd.nc@freq%5Fband-alpha` |

The encoding is reversible: every column maps back to exactly one derivative, artifact and set
of `(dimension, value)` pairs. To read one by hand, split at the first `@`, then the head at its
first `.` (name and suffix), then the coordinates on `_` and each pair at its first `-`, and
percent-decode. {func}`neurodags.naming.parse_column` does exactly this.

**Collisions are errors.** If two coordinates, artifacts or derivatives would produce the same
column (possible only with a non-unique coordinate index, or with mixed types such as `1` and
`"1"` in one coordinate), the build stops with {class}`neurodags.naming.ColumnCollisionError`
naming the column and both origins. This happens even with `raise_on_error=False`: a collision
is a problem of the pipeline's outputs, not of one file, so it is never recorded in a
`name.error` column. With `n_jobs > 1` the error is re-raised in the calling process; the
`neurodags dataframe` CLI prints `neurodags: error: Dataframe column collision: ...` and exits
with code `2`.

**Errors.** With `raise_on_error=False`, a derivative that fails for a file leaves its message in
the `<name>.error` column of that row (wide) or in a row with `derivative = "<name>.error"`
(long).

(dataframe-splitting-columns)=

## Splitting Column Names

The dataframe is **easy to split, not pre-split**: neurodags guarantees that every generated
column name splits unambiguously, but does not add columns for the pieces. Which pieces you need
(one column per dimension, parameters encoded in derivative names, pivots) depends on the
analysis, so you split them yourself with the public helpers in {mod}`neurodags.naming`:
{func}`~neurodags.naming.parse_column`, {func}`~neurodags.naming.decode_dim`,
{func}`~neurodags.naming.decode_value`, {func}`~neurodags.naming.parse_entities` and
{func}`~neurodags.naming.decode_number`.

On the long format, parse the `derivative` column and expand the coordinates into one column per
dimension. If your derivative names follow the key-value convention (`Fit_alpha-0p5_w-6`: base
`Fit`, parameters `alpha=0p5`, `w=6`), {func}`~neurodags.naming.parse_entities` turns the name
into parameter columns and {func}`~neurodags.naming.decode_number` reads `0p5` as `0.5`:

<!-- tested by tests/test_dataframe_columns.py::test_docs_split_recipe -->

```python
import pandas as pd

from neurodags.naming import decode_number, parse_column, parse_entities

# df = build_derivative_dataframe("pipeline.yml", output_format="long")
parsed = df["derivative"].map(parse_column)
df["name"] = parsed.map(lambda p: p.name)
df["suffix"] = parsed.map(lambda p: p.suffix)  # ".error" for error rows
coords = pd.DataFrame([dict(p.coords) for p in parsed], index=df.index)
df = df.join(coords)  # one column per dimension; values are strings ("8-12", "-1.5")

# Optional: parameters encoded in derivative names (key-value convention).
entities = df["name"].map(lambda n: (parse_entities(n) or (None, {}))[1])
df = df.join(pd.DataFrame(entities.tolist(), index=df.index))
df["alpha"] = df["alpha"].map(decode_number, na_action="ignore")  # "0p5" -> Decimal("0.5")
```

Coordinate values stay strings: convert them with `pd.to_numeric` where that makes sense. If a
dimension name clashes with an existing column (for example a dimension called `value`), pass
`rsuffix=` to `join` or rename the columns first. On the wide format, parse the column labels
instead: `{c: parse_column(c) for c in df.columns}` maps metadata columns to `None`.

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
