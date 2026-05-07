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

#### Reuse Step (load derivative from disk)

```yaml
- id: 0
  derivative: CleanedEEG.fif
```

- `derivative`: `<DerivativeName>.<ext>` — loads the named derivative for the current file from disk
- Use `SourceFile` to load the raw input file

#### id.N References

`id.<N>` in args resolves to the output of step `N`. When a step produces multiple artifacts (multiple extensions), the first artifact is returned unless the extension is specified.

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
