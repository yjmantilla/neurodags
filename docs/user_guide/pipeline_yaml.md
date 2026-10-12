# pipeline.yml Reference

`pipeline.yml` is the top-level declarative configuration. It defines all derivative computations, links to datasets, and controls execution.

## Top-Level Keys

```yaml
datasets: datasets.yml          # path to datasets.yml
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

## `datasets`

Path to `datasets.yml`. Relative paths are resolved from the pipeline YAML location.

## `mount_point`

Selects which environment's paths to use from `datasets.yml`. Must match a key in each dataset's `file_pattern` / `derivatives_path` maps.

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

Each key is a derivative name (CamelCase by convention). The value is a derivative definition:

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

- `derivative`: `<DerivativeName>.<ext>` — resolves the named derivative for the current file
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
