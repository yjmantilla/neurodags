# pipeline.yml Reference

`pipeline.yml` is the top-level declarative configuration. It defines all derivative computations, links to datasets, and controls execution.

## Top-Level Keys

```yaml
naming_version: 1               # required since 0.4.0
datasets: datasets.yml          # path to datasets.yml (or an inline mapping)
mount_point: local              # active mount point (matches key in datasets.yml)
new_definitions: custom_nodes.py  # optional: custom node module(s)

n_jobs: null                    # optional: parallelism (null = serial, -1 = all cores)
joblib_backend: loky            # optional: joblib backend
joblib_prefer: processes        # optional: joblib prefer hint

DerivativeDefinitions:
  <DerivativeName>:
    ...

DerivativeList:
  - <DerivativeName>
  - ...
```

## `naming_version`

**Required since 0.4.0.** Must be exactly `naming_version: 1`. It declares that the
configuration follows the 0.4.0 naming, output-layout and completion rules. A configuration
without it, or with another value, is refused before anything runs by every entry point
(all CLI subcommands, the TUI, `run_pipeline`, `iterate_derivative_pipeline`,
`build_derivative_dataframe`, ...):

```text
neurodags: error: pipeline configuration 'pipeline.yml' has no top-level 'naming_version'. neurodags 0.4.0 requires 'naming_version: 1'. To upgrade a neurodags 0.3.x configuration, run `neurodags migrate` (available in 0.4.0).
```

## `datasets`

Path to `datasets.yml`, or an inline mapping of datasets (same fields as in
{doc}`datasets_yaml`). A relative path is resolved from the pipeline YAML location.

Relative paths *inside* the datasets definitions (`source_root`, `file_pattern`,
`exclude_pattern`, `derivatives_path`) are resolved against the directory of the YAML file
that contains them: the datasets YAML, or `pipeline.yml` for an inline mapping (changed in
0.4.0; they used to be relative to the current working directory). For a configuration
passed as a dict to the Python API, pass `base_dir=` to `run_pipeline` /
`iterate_derivative_pipeline` / `build_derivative_dataframe`; relative dataset paths without
it are an error.

## `mount_point`

Selects which environment's paths to use from `datasets.yml`. Must match a key in each dataset's `source_root` / `file_pattern` / `derivatives_path` maps.

## `new_definitions`

Path (or list of paths) to Python modules that register custom nodes. Loaded before any derivatives execute.

```yaml
new_definitions: custom_nodes.py

# or multiple files:
new_definitions:
  - custom_nodes/nodes_a.py
  - /abs/path/to/nodes_b.py
```

Relative paths resolved from the pipeline YAML location. Each module is executed once on import.

---

## DerivativeDefinitions

Each key is a derivative name (CamelCase by convention, see [Derivative names](#derivative-names)). The value is a derivative definition:

```yaml
DerivativeDefinitions:
  MyDerivative:
    save: True             # default True
    overwrite: False       # default False
    for_dataframe: False   # default False
    nodes:
      - id: 0
        ...
      - id: 1
        ...
```

(derivative-names)=

### Derivative names

Outputs are written as `<source>@<OnDiskName><artifact suffix>`. A name without `_` is used as
is (`BandPower` → `@BandPower`, `psd-welch` → `@psd-welch`); a name with `_` is converted to
CamelCase (`fit_w_l06` → `@FitWL06`, `band_power` → `@BandPower`), keeping the derivative name a
single token without `_` (a separator in many naming conventions, including BIDS). Everything else — `DerivativeList`, reuse steps
(`derivative: fit_w_l06.nc`), dataframe columns — uses the YAML name. `neurodags validate` shows
each on-disk name that differs from its YAML name, and names that collide on disk (e.g.
`band_power` and `BandPower`) or contain `.`, `@`, `/`, `\`, `*`, `?`, `[`, `]` are rejected
before anything runs.

See **[File Naming Conventions](naming.md)** for all naming rules: output files, marker files,
references, the output layout, `&` in chained pipelines and dataframe column names.

### Derivative Flags

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `save` | bool | `True` | Persist output artifacts to disk. `False` = compute but don't write. |
| `overwrite` | bool | `False` | Force recompute even if cached output exists. |
| `for_dataframe` | bool | `False` | Include this derivative when calling `build_derivative_dataframe`. |

### Node Steps

Each entry in `nodes` is a step with a unique `id`. Steps execute in topological order. A step is either a **compute step** (runs a node function) or a **reuse step** (loads a derivative from disk).

#### Compute Step

```yaml
- id: 1
  node: my_node_name
  args:
    input_data: id.0     # reference to step 0's output
    param_a: 42
    param_b: [1, 40]
```

- `node`: name of a registered node function
- `args`: keyword arguments passed to the node; values of the form `id.<N>` resolve to the artifact produced by step `N`

#### Reuse Step (load derivative from disk or in-memory)

```yaml
- id: 0
  derivative: CleanedEEG.fif
```

- `derivative`: `<DerivativeName>.<ext>` — resolves the named derivative for the current file; use the YAML name, not the on-disk name (see {ref}`Derivative references <naming-references>`)
- Use `SourceFile` to load the raw input file

If the upstream derivative is already cached on disk, the matching file is loaded. If the upstream derivative has not yet been written (it is being computed in the same run), the in-memory `NodeResult` is used directly — both paths produce identical results.

##### Selecting one artifact from a multi-artifact derivative

When an upstream derivative is a **splitter** — a node that returns one artifact per condition, segment type, or split key — the same `<DerivativeName>.<ext>` syntax selects a specific artifact:

```yaml
- id: 0
  derivative: EpochSplitter.EO_baseline.fif   # selects only the EO_baseline artifact
```

This works whether `EpochSplitter` is cached on disk or in memory.  If the requested extension is absent from the upstream derivative's artifacts, a warning is logged and the full `NodeResult` is passed as a fallback.

```yaml
# Splitter example
DerivativeDefinitions:
  EpochSplitter:
    nodes:
      - id: 0
        derivative: SourceFile
      - id: 1
        node: split_by_condition    # returns .EO_baseline.fif, .EC_baseline.fif, …
        args:
          mne_object: id.0

  EO_Features:
    nodes:
      - id: 0
        derivative: EpochSplitter.EO_baseline.fif   # only EO_baseline epochs
      - id: 1
        node: compute_features
        args:
          epochs: id.0
```

#### id.N References

`id.<N>` in args resolves to the output of step `N`. When a step produces multiple artifacts, the artifact matching the node's expected argument type is selected by the `_unwrap_for_arg` heuristic (first `.fif` for MNE objects, first artifact otherwise). To select a specific artifact from a multi-artifact upstream derivative, use the `derivative: Name.ext` reuse step (see above) rather than relying on `id.<N>` heuristics.

#### Field access: `id.N.field`

When step `N` produces a **mapping** (a dict — e.g. a [parser node](parser_nodes.md) such as `bids_parse`), a downstream arg can pull a single field with `id.<N>.<field>`:

```yaml
- id: 1
  node: bids_parse
  args: {path: id.0}          # -> {"subject": "001", "task": "rest", ...}
- id: 2
  node: my_processing_node
  args:
    raw:     id.0
    subject: id.1.subject      # "001"
    task:    id.1.task         # "rest"
```

**Field-key constraint.** `.` is the reference separator, so an addressable `field` must be a **plain identifier** — letters, digits and underscores, not starting with a digit. Dotted, dashed, or nested paths (`id.1.a.b`, `id.1.some-key`) are **not** supported and raise a clear error; keep parser output flat with identifier keys (BIDS entities already comply). Values are unconstrained. To consume the whole mapping instead, reference `id.N` (no field) and index it inside the node.

---

## Sharing Arguments with Anchors and Merge Keys

When many derivatives share the same node arguments, define them once with a YAML anchor
(`&name`) and pull them into each `args` mapping with a merge key (`<<: *name`):

```yaml
_defaults: &fitargs
  n_iter: 100
  alphas: [1.0, 1000.0, 20]

DerivativeDefinitions:
  FitA:
    nodes:
      - {id: 1, node: my_fit, args: {<<: *fitargs, space: a}}
  FitB:
    nodes:
      - id: 1
        node: my_fit
        args:
          <<: *fitargs
          n_iter: 5        # overrides the merged value
          space: b
```

Standard YAML merge semantics apply:

- Keys written in the mapping itself override merged ones (`FitB` above runs with `n_iter: 5`).
- `<<: [*a, *b]` merges several anchors; when they share a key, the earlier one wins.

The anchor can live under any top-level key that neurodags does not use; `_defaults` is a
good convention. Unknown top-level keys are ignored by the loader and by `neurodags validate`,
so the helper key needs no extra configuration. An anchor must be defined before it is used,
so put the helper key above `DerivativeDefinitions`.

Duplicate keys are still rejected: writing the same key twice in one mapping (including inside
the anchored mapping) raises `Duplicate key '<key>' at line L, column C`. Overriding a key that
came from a merge, as in `FitB`, is not a duplicate.

---

## DerivativeList

Controls which derivatives execute and in what order. Derivatives not listed here are defined but never run. Comment out entries to skip without removing definitions:

```yaml
DerivativeList:
  - CleanedEEG
  - CrossSpectralDensity
  - PowerSpectrum
  # - SpectralEntropy     # skip this one
  - BandPower
```

---

## Complete Example

```yaml
naming_version: 1
datasets: datasets.yml
mount_point: local
new_definitions: custom_nodes.py

DerivativeDefinitions:
  CleanedEEG:
    nodes:
      - id: 0
        derivative: SourceFile
      - id: 1
        node: basic_preprocessing
        args:
          mne_object: id.0
          resample: 256
          filter_args:
            l_freq: 0.5
            h_freq: 110

  CrossSpectralDensity:
    nodes:
      - id: 0
        derivative: CleanedEEG.fif
      - id: 1
        node: welch_csd
        args:
          data: id.0
          n_fft: 1024

  Coherence:
    nodes:
      - id: 0
        derivative: CrossSpectralDensity.nc
      - id: 1
        node: compute_coherence
        args:
          csd: id.0
          bands:
            alpha: [8, 12]

  PowerSpectrum:
    for_dataframe: True
    nodes:
      - id: 0
        derivative: CrossSpectralDensity.nc
      - id: 1
        node: extract_auto_spectra
        args:
          csd: id.0

  SpectralEntropy:
    overwrite: True
    nodes:
      - id: 0
        derivative: PowerSpectrum.nc
      - id: 1
        node: spectral_entropy
        args:
          psd: id.0

  BandPower:
    save: False
    nodes:
      - id: 0
        derivative: PowerSpectrum.nc
      - id: 1
        node: extract_bands
        args:
          psd: id.0
          bands:
            alpha: [8, 12]
            beta: [13, 30]

  AlphaNetworkCoupling:
    nodes:
      - id: 0
        derivative: BandPower.nc
      - id: 1
        derivative: Coherence.nc
      - id: 2
        node: correlate_power_connectivity
        args:
          bandpower: id.0
          coherence: id.1
          band: alpha

DerivativeList:
  - CleanedEEG
  - CrossSpectralDensity
  - PowerSpectrum
  # - SpectralEntropy
  - Coherence
  - BandPower
  - AlphaNetworkCoupling
```

This pipeline corresponds to the following DAG:

```
SourceFile
  └─ CleanedEEG
       └─ CrossSpectralDensity
            ├─ PowerSpectrum (for_dataframe)
            │    ├─ SpectralEntropy (overwrite=True)
            │    └─ BandPower (save=False)
            │         └─ AlphaNetworkCoupling ←─ Coherence
            └─ Coherence
```
