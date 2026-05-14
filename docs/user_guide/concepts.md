# Core Concepts

## Directed Acyclic Graph (DAG)

A pipeline is a DAG of computation nodes. Each node takes inputs (from previous nodes or cached derivatives) and produces outputs (artifacts). Edges encode data dependencies; cycles are not allowed.

```
SourceFile → CleanedEEG → CrossSpectralDensity → PowerSpectrum → BandPower
                                               ↘ Coherence ↗
```

## Derivative

A **derivative** is a named, reusable computation result associated with an input file. Derivatives are stored on disk with a `@DerivativeName` suffix next to their source file:

```
sub-1.fif               → source
sub-1@CleanedEEG.fif    → derivative
sub-1@PowerSpectrum.nc  → derivative
```

Derivatives mirror the input directory structure — NeuroDAGs is agnostic to how your data is organized.

## Node

A **node** is a Python function decorated with `@register_node`. It receives data as keyword arguments and returns a `NodeResult`. Preprocessing nodes, spectral analysis nodes, entropy nodes, and custom nodes are all treated identically.

## NodeResult and Artifact

```python
class Artifact(NamedTuple):
    item: Any                        # the data object
    writer: Callable[[str], None]    # how to save it to disk

class NodeResult(NamedTuple):
    artifacts: dict[str, Artifact]   # extension → Artifact
```

A node returns one or more artifacts keyed by file extension. Example: `{".nc": Artifact(...), ".report.html": Artifact(...)}`. The last node in a derivative's chain saves under the `@DerivativeName` prefix.

## Pipeline Steps

Each derivative definition has a `nodes` list. A step is either:

- **`node`**: execute a registered Python function
- **`derivative`**: load a cached derivative from disk (enables reuse without recomputing)

Steps reference earlier results via `id.<N>` (e.g., `id.0` = result of step 0).

## File Independence

NeuroDAGs processes each input file in isolation. Every derivative is computed independently per file — there are no cross-file operations within the framework. This is a deliberate design constraint that enables trivial parallelism (each file is an independent job) and caching.

The consequence: operations that require information from multiple files — group-level ICA, normalization to a group mean, atlas registration using a subject-average template — cannot be expressed as NeuroDAGs derivatives. These must be done outside the pipeline, either as a post-processing step or by dropping to plain Python/NumPy after running `build_derivative_dataframe`.

If your workflow needs cross-file operations mid-pipeline, consider Snakemake or Pydra instead.

## Caching

If a derivative's final artifact already exists on disk and `overwrite: False` (the default), the entire derivative computation is skipped. This allows you to resume interrupted pipelines and avoid redundant computation in large studies.

**Cache invalidation is existence-based, not code-based.** If you change a node's implementation, NeuroDAGs has no way to know the cached output is stale — it only checks whether the output file exists. To force recomputation after modifying a node, either set `overwrite: true` on that derivative or delete the relevant `@DerivativeName` files manually.

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
| `overwrite` | bool | `False` | Recompute even if output exists. |
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
mount_point: local   # or: hpc
```

```yaml
# datasets.yml
my_dataset:
  file_pattern:
    local: data/**/*.vhdr
    hpc: /cluster/BIDS/**/*.vhdr
  derivatives_path:
    local: outputs/
    hpc: /cluster/scratch/out
```

This makes pipelines portable across workstations and HPC clusters without editing the pipeline YAML.
