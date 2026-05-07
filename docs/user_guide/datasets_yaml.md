# datasets.yml Reference

`datasets.yml` decouples data sources from processing logic, so the same dataset definitions can be reused across multiple pipelines.

## Structure

```yaml
<dataset_key>:
  name: <string>
  file_pattern: <string or mount-point map>
  derivatives_path: <string or mount-point map>
  exclude_pattern: <glob string>   # optional
  skip: <bool>                     # optional, default False
```

## Fields

### `name`

Human-readable dataset identifier. Used in logging and dataframe output.

### `file_pattern`

Glob pattern for finding source files. Supports mount-point maps for environment-specific paths:

```yaml
# Single path
file_pattern: data/**/*.vhdr

# Environment-specific (mount points)
file_pattern:
  local: data/**/*.vhdr
  hpc: /cluster/BIDS/**/*.vhdr
```

The active mount point is set by `mount_point` in `pipeline.yml`.

### `derivatives_path`

Root directory where derivative files are written. Mirrors the source file's subdirectory structure under this root:

```yaml
derivatives_path:
  local: outputs/
  hpc: /cluster/scratch/out
```

### `exclude_pattern`

Optional glob to exclude files matching the pattern from processing.

### `skip`

Set `skip: True` to temporarily disable a dataset without removing it.

## Multiple Datasets

List any number of datasets. All are processed when running `iterate_derivative_pipeline`:

```yaml
dataset1:
  name: StudyA
  file_pattern:
    local: data/studyA/**/*.vhdr
  derivatives_path:
    local: outputs/studyA/

dataset2:
  name: StudyB
  file_pattern:
    local: data/studyB/**/*.vhdr
  derivatives_path:
    local: outputs/studyB/
```

## Referencing in pipeline.yml

```yaml
# pipeline.yml
datasets: datasets.yml    # path to datasets file
mount_point: local        # which mount point to activate
```

`datasets` can be an absolute or relative path. Relative paths are resolved from the pipeline YAML location.
