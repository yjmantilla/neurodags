# Core Concepts

## Directed Acyclic Graph (DAG)

A pipeline is a DAG of computation nodes. Each node takes inputs (from previous nodes or cached derivatives) and produces outputs (artifacts). Edges encode data dependencies; cycles are not allowed.

```
SourceFile → CleanedEEG → CrossSpectralDensity → PowerSpectrum → BandPower
                                               ↘ Coherence ↗
```

## Derivative

A **derivative** is a named, reusable computation result associated with an input file. Derivatives are stored on disk with `@` and the derivative name, exactly as written in the YAML, appended to their source file's name:

```
sub-1.fif                     → source
sub-1.fif@CleanedEEG.fif      → derivative
sub-1.fif@PowerSpectrum.nc    → derivative
```

Derivatives mirror the directory structure below each dataset's `source_root` (inside `derivatives_path` when it is set, otherwise next to the sources) — NeuroDAGs is agnostic to how your data is organized. A derivative named `power_spectrum` writes `@power_spectrum` files; names are never converted. See **[File Naming and Output Layout](naming.md)** for every naming rule and restriction.

### Chaining pipelines (neurodags output as source)

When a derivative produced by one pipeline run is used as the source file in a subsequent pipeline run, its filename already contains `@` (e.g. `sub-1.fif@CleanedEEG.fif`). NeuroDAGs automatically replaces that `@` with `&` when building the reference base, so the one-`@`-per-filename invariant is preserved:

```
sub-1.fif@CleanedEEG.fif      → source (a previous derivative)
sub-1.fif&CleanedEEG.fif@PowerSpectrum.nc  → new derivative
```

This means downstream tooling can always split on the single `@` (`basename.rsplit("@", 1)`) to separate the source identity from the derivative name; the original source name is kept in the unit's `.done` manifest. Set the downstream dataset's `source_root` to the upstream `derivatives_path` (or a folder of it): discovery never returns the dataset's own output tree, nor the `code/` and `.neurodags/` folders (see {ref}`Output layout <naming-output-layout>`).

Avoid `@` and `&` in raw source file names: two sources that differ only by `@`/`&` (`a@b.vhdr`, `a&b.vhdr` in one folder) map to the same outputs and are refused before running.

## Node

A **node** is a Python function decorated with `@register_node`. It receives data as keyword arguments and returns a `NodeResult`. Preprocessing nodes, spectral analysis nodes, entropy nodes, and custom nodes are all treated identically.

## NodeResult and Artifact

```python
class Artifact(NamedTuple):
    item: Any                                      # the data object
    writer: Callable[[str], list[str] | None]      # saves it; returns the files written (None = exactly the path)

class NodeResult(NamedTuple):
    artifacts: dict[str, Artifact]   # artifact suffix → Artifact
```

A node returns one or more artifacts keyed by **artifact suffix** (`.nc`, `.report.html`, `.EO_baseline.fif`; see {ref}`Output file names <naming-output-files>` for the grammar). Example: `{".nc": Artifact(...), ".report.html": Artifact(...)}`. The last node in a derivative's chain saves each artifact as `<source>@<DerivativeName><suffix>`; the writer must report every file it writes (see {ref}`Writer contract <naming-writer-contract>`).

### Multi-Artifact (Splitter) Nodes

A node can return several artifacts in one `NodeResult`, each under a distinct extension key. This is the **splitter** pattern: one artifact per condition, segment type, or split key.

```python
from neurodags.writers import save_mne

@register_node
def split_by_condition(epochs) -> NodeResult:
    artifacts = {}
    for cond in epochs.event_id:
        artifacts[f".{cond}.fif"] = Artifact(
            item=epochs[cond],
            writer=lambda path, e=epochs[cond]: save_mne(e, path),  # reports every file written
        )
    return NodeResult(artifacts=artifacts)
```

Downstream derivatives select one artifact using the `derivative: SplitterName.condA.fif` syntax in the pipeline YAML: the text after the name must equal one of the artifact suffixes exactly (see {ref}`Derivative references <naming-references>`). The selection is applied identically whether the splitter's output is already done on disk or is still in memory from the current run.

## Pipeline Steps

Each derivative definition has a `nodes` list. A step is either:

- **`node`**: execute a registered Python function
- **`derivative`**: resolve an upstream derivative by name and artifact suffix (`Name.nc` selects exactly the `.nc` artifact); a bare `Name` is allowed only when the derivative has exactly one artifact, and for a splitter `derivative: Name.condA.fif` selects a single artifact

Steps reference earlier results via `id.<N>` (e.g., `id.0` = result of step 0).

## File Independence

NeuroDAGs processes each input file in isolation. Every derivative is computed independently per file — there are no cross-file operations within the framework. This is a deliberate design constraint that enables trivial parallelism (each file is an independent job) and caching.

The consequence: operations that require information from multiple files — group-level ICA, normalization to a group mean, atlas registration using a subject-average template — cannot be expressed as NeuroDAGs derivatives. These must be done outside the pipeline, either as a post-processing step or by dropping to plain Python/NumPy after running `build_derivative_dataframe`.

If your workflow needs cross-file operations mid-pipeline, consider Snakemake or Pydra instead.

## Caching

When a derivative finishes for a file (a *unit*), NeuroDAGs writes a small `.done` manifest next to its outputs listing every file it wrote. If that manifest is valid (every listed file still exists) and `overwrite: False` (the default), the entire derivative computation is skipped. This allows you to resume interrupted pipelines and avoid redundant computation in large studies; a unit interrupted mid-write has no `.done` and is recomputed. Output files without a valid `.done` (including outputs written by 0.3.x and not migrated) are never trusted. `neurodags status` reports every unit as `done`, `done (migrated)`, `skipped`, `errored`, `incomplete` or `missing` (see {ref}`Unit states <naming-unit-states>` and {ref}`Completion manifest <naming-done-manifest>`).

**Cache invalidation is manifest-based, not code-based.** If you change a node's implementation, NeuroDAGs has no way to know the cached output is stale — it only checks the manifest and the files it lists. To force recomputation after modifying a node, either set `overwrite: true` on that derivative or delete the relevant `@<name>.done` files (or all `@<name>.*` files) manually.

## SourceFile

`SourceFile` is a built-in pseudo-derivative that resolves to the raw input file. Use it as the first node in any derivative chain that starts from raw data.

```yaml
nodes:
  - id: 0
    derivative: SourceFile
  - id: 1
    node: my_preprocessing_node
    args:
      data: id.0
```

## Derivative Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `save` | bool | `True` | Write artifacts to disk. `False` = compute in memory only. |
| `overwrite` | bool | `False` | Recompute even if the unit is done (valid `.done` manifest). |
| `for_dataframe` | bool | `False` | Include in `build_derivative_dataframe` output. |

## DerivativeList

`DerivativeList` in `pipeline.yml` controls which derivatives are executed and in what order. Comment out a derivative name to skip it without removing its definition:

```yaml
DerivativeList:
  - CleanedEEG
  - PowerSpectrum
  # - SpectralEntropy   # skipped
  - BandPower
```

## Mount Points

`datasets.yml` supports environment-specific path resolution. The `mount_point` key in `pipeline.yml` selects which path set to use:

```yaml
# pipeline.yml
naming_version: 1    # required since 0.4.0
mount_point: local   # or: hpc
```

```yaml
# datasets.yml (relative paths are relative to this file's folder)
my_dataset:
  name: MyDataset
  source_root:
    local: data
    hpc: /cluster/BIDS
  file_pattern:
    local: data/**/*.vhdr
    hpc: /cluster/BIDS/**/*.vhdr
  derivatives_path:
    local: outputs/
    hpc: /cluster/scratch/out
```

This makes pipelines portable across workstations and HPC clusters without editing the pipeline YAML.

## Dataset Variables (`vars`)

Dataset entries can declare a `vars:` block. Any pipeline node arg whose string value matches `$identifier` is substituted with the corresponding value at runtime.

```yaml
# datasets.yml
eeg_study_EO:
  name: EEG study, eyes open
  source_root: { local: data }
  file_pattern: { local: data/**/*@clean_prep.fif }
  derivatives_path: { local: outputs/features/EO_baseline }
  vars:
    condition_name: EO_baseline
    epoch_duration: 2.0
```

```yaml
# pipeline.yml
CleanedPrep:
  save: False
  nodes:
    - id: 0
      derivative: SourceFile
    - id: 1
      node: extract_condition_epochs
      args:
        condition_name: $condition_name   # resolved from active dataset entry
        epoch_duration: $epoch_duration
```

The substitution runs after `id.N` reference resolution and before node invocation. Only whole-string values are substituted — embedded `$` in paths or other strings is ignored. Variables can be any YAML scalar or collection type (string, int, float, bool, list).

**Key benefit**: the dataset entry becomes the single source of truth for a run. Switching the active entry (via `skip: true/false`) changes both `derivatives_path` and all `$var_name` args simultaneously — no pipeline YAML edits needed.

See [datasets_yaml.md](datasets_yaml.md) for full reference including multi-site and cohort-specific use cases.
