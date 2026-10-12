# datasets.yml Reference

`datasets.yml` decouples data sources from processing logic, so the same dataset definitions can be reused across multiple pipelines.

## Structure

```yaml
<dataset_key>:
  name: <string>
  file_pattern: <string or mount-point map>
  derivatives_path: <string or mount-point map>
  exclude_pattern: <glob string>             # optional
  drop_split_continuations: <bool>           # optional, default True
  skip: <bool>                               # optional, default False
  vars: {<key>: <value>, ...}               # optional, dataset-level variables
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

See "Environment variables and `~`" below for using
`$SCRATCH`-style paths.

### `derivatives_path`

Root directory where derivative files are written. Mirrors the source file's subdirectory structure under this root:

```yaml
derivatives_path:
  local: outputs/
  hpc: /cluster/scratch/out
```

### `exclude_pattern`

Optional glob to exclude files matching the pattern from processing. It is
expanded like `file_pattern` (see below).

### Environment variables and `~`

`file_pattern`, `derivatives_path` and `exclude_pattern` (and the `datasets:`
reference in `pipeline.yml`) expand a leading `~` and environment variables
written as `$NAME` or `${NAME}` (`%NAME%` is also accepted on Windows):

```yaml
derivatives_path:
  local: ~/neurodags-out
  hpc: $SCRATCH/myproject/derivatives
```

- Only the value for the **active** mount point is expanded. Other entries may
  reference variables that exist only on other machines (e.g. `$SCRATCH` on a
  laptop).
- A referenced variable that is **unset or empty is an error**, raised before
  anything is written. neurodags never creates a directory literally named
  `$SCRATCH` or `~`. `neurodags validate` reports the resolved paths (or the
  error) without running anything.
- A `$` not followed by a variable name (`a$1`, `foo$`) is kept literally.
  There is no escape syntax for a literal `$NAME`.
- Relative paths are still resolved against the current working directory, as
  before; expansion does not change that.
- Skipped datasets (`skip: True`) are not checked.

This is unrelated to `vars` (below): `vars` are dataset-level values substituted
into pipeline node args (`$condition_name`), not environment variables, and node
args are never expanded from the environment.

### `drop_split_continuations`

Optional, default `True`. MNE writes recordings larger than ~2 GB as a set of
split `.fif` files, where only the first (*entry*) file is an independent
recording — `mne.io.read_raw_fif(entry)` stitches the continuations back in
automatically. With this enabled, the scanner keeps the entry file of each split
set and drops the continuations, so the pipeline does not process partial
recordings as if they were separate source files. Both naming conventions are
recognised:

- **BIDS** (mne-bids): keeps `..._split-01_meg.fif`, drops `..._split-02_meg.fif`+
- **plain mne**: keeps `name.fif`, drops `name-1.fif`, `name-2.fif`, …

Detection is filename-based (no files are opened during scanning), and the
plain-mne rule only drops a `-N` file when its entry file is also present, so
legitimately named files such as `sub-01_..._run-2_meg.fif` are never dropped.
Set to `False` to disable and return every matched file. The number of dropped
continuations is logged.

### `skip`

Set `skip: True` to temporarily disable a dataset without removing it.

### `vars`

Dataset-level variables that are substituted into pipeline node args at runtime. Any string arg value matching `$identifier` is replaced by the corresponding value from `vars`.

```yaml
eeg_adhd_epilepsy_EO_baseline:
  name: eeg_adhd_epilepsy_EO_baseline
  file_pattern:
    local: data/**/*@CleanedPrepRaw.fif
  derivatives_path:
    local: outputs/features_conditions/EO_baseline
  vars:
    condition_name: EO_baseline
    epoch_duration: 2.0
```

Then in the pipeline YAML:

```yaml
CleanedPrep:
  save: False
  nodes:
    - id: 0
      derivative: SourceFile
    - id: 1
      node: extract_condition_epochs
      args:
        condition_name: $condition_name   # → "EO_baseline" from the active dataset entry
        epoch_duration: $epoch_duration   # → 2.0
```

**Substitution rules:**
- Only whole-string values are substituted: `$condition_name` → substituted; `/path/$HOME/file` → untouched.
- Variables may be any YAML type: string, int, float, bool, list.
- Referencing an undefined variable raises a `KeyError` at runtime with the list of available vars.
- `vars` only applies to node `args`; it has no effect on `derivative:` step references or other YAML keys.

**Primary use case — condition-per-dataset-entry**: Each dataset entry encodes a different condition. The active entry drives both the `derivatives_path` (where features are written) and the `$condition_name` arg (which condition to epoch). Activating a different entry changes both in one step, with no pipeline YAML edits needed.

```yaml
# step-1_dataset.yml
eeg_adhd_epilepsy_EO_baseline:
  vars: { condition_name: EO_baseline }
  derivatives_path: { local: outputs/features/EO_baseline }
  ...

eeg_adhd_epilepsy_EC_baseline:
  skip: true
  vars: { condition_name: EC_baseline }
  derivatives_path: { local: outputs/features/EC_baseline }
  ...
```

**Other uses**: multi-site studies (different `line_freq` per site), cohort-specific parameters (different `epoch_duration` for pediatric vs adult), or any parameter that logically belongs to the dataset rather than the pipeline logic.

## Multiple Datasets

List any number of datasets. All are processed when running `run_pipeline`:

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
