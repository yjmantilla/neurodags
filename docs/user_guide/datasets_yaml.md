# datasets.yml Reference

`datasets.yml` decouples data sources from processing logic, so the same dataset definitions can be reused across multiple pipelines.

```{admonition} Changed in 0.4.0
:class: important

- `source_root` is **required** for every dataset (see below).
- Relative paths in a datasets YAML (`source_root`, `file_pattern`, `exclude_pattern`,
  `derivatives_path`) are resolved against **the directory of that datasets YAML**. In 0.3.x
  they were resolved against the current working directory.
- New optional `derivatives_subdir`; two active datasets resolving to the same
  `derivatives_path` must use different `derivatives_subdir` values.
- The pipeline YAML must declare `naming_version: 1` (see the
  [pipeline.yml reference](pipeline_yaml.md)).
- Outputs mirror the tree below `source_root` (see {ref}`Output layout <naming-output-layout>`),
  and discovered files are sorted (see [Discovery and file order](#datasets-discovery-order)).

To convert a 0.3.x datasets YAML (and its outputs), use `neurodags migrate`
({doc}`migrating_to_0_4`).
```

## Structure

```yaml
<dataset_key>:
  name: <string>
  source_root: <string or mount-point map>   # required (0.4.0)
  file_pattern: <string or mount-point map>
  derivatives_path: <string or mount-point map>
  derivatives_subdir: <relative path>        # optional (0.4.0)
  exclude_pattern: <glob string>             # optional
  drop_split_continuations: <bool>           # optional, default True
  skip: <bool>                               # optional, default False
  vars: {<key>: <value>, ...}               # optional, dataset-level variables
```

## Fields

### `name`

Required. Human-readable display label. The dataset **key** (`<dataset_key>`), not `name`, is
what identifies the dataset in the `dataset` column of dataframes, in `inputs.tsv`, in `.done`
manifests and in `status --include-files`.

### `source_root`

**Required since 0.4.0**, for every dataset (also skipped ones). The root folder of the
dataset's source files, as a string or a mount-point map with an entry for the active mount
point:

```yaml
source_root:
  local: ~/data/study/rawdata
  hpc: $SCRATCH/study/rawdata
file_pattern:
  local: ~/data/study/rawdata/**/*.vhdr
  hpc: $SCRATCH/study/rawdata/**/*.vhdr
```

- For an active dataset it must be an existing directory; every file matched by
  `file_pattern` must lie strictly below it (a file outside it is a configuration error naming
  the file).
- Choose the folder your recordings are organised under (e.g. the BIDS root), not a
  subject folder: neurodags will mirror the tree below `source_root` inside
  `derivatives_path`, so narrowing `file_pattern` for a test run (`rawdata/sub-01/**/*.vhdr`)
  does not move the outputs.
- Across mount points, the files below `source_root` must have the same relative paths
  when the data is the same (neurodags cannot verify this; `neurodags validate` prints the
  resolved root for the active mount).
- When it is missing, `neurodags validate` prints a suggestion: the leading folders of the
  expanded `file_pattern` before the first component containing `*`, `?` or `[` (a pattern
  without wildcards suggests its parent folder). No suggestion is made when that would be a
  filesystem root (`/`, `C:\`).

### `file_pattern`

Glob pattern for finding source files. Supports mount-point maps for environment-specific paths:

```yaml
# Single path (relative to this datasets YAML's folder)
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

Root directory where derivative files are written. Below it (and below `derivatives_subdir`,
if set) the outputs mirror each source file's folders relative to `source_root` (see
{ref}`Output layout <naming-output-layout>`). Without `derivatives_path`, outputs are written
next to the sources and no `code/` snapshot is made:

```yaml
derivatives_path:
  local: outputs/
  hpc: /cluster/scratch/out
```

The `code/` snapshot of the configuration is written to `<derivatives_path>/code/`.

### `derivatives_subdir`

Optional (0.4.0), default unset. A relative subfolder of `derivatives_path` for this
dataset's outputs (`/`-separated, no `..`, not absolute, not starting with `code` or
`.neurodags`; used literally, without `~`/`$VAR` expansion).

If two active datasets in one configuration resolve to the same `derivatives_path`, their
`derivatives_subdir` values must differ (unset counts as one value); otherwise `validate`
and every run refuse the configuration. Adding a second dataset therefore never moves the
first one's outputs: only the new dataset needs a subfolder.

```yaml
study_a:
  name: StudyA
  source_root: /data/studyA
  file_pattern: /data/studyA/**/*.vhdr
  derivatives_path: /data/derivatives          # outputs of study_a (unchanged)

study_b:
  name: StudyB
  source_root: /data/studyB
  file_pattern: /data/studyB/**/*.vhdr
  derivatives_path: /data/derivatives
  derivatives_subdir: study_b                  # required: shares derivatives_path
```

Outputs of `study_b` go to `/data/derivatives/study_b/<folders below source_root>/`; the
`code/` snapshot stays at `/data/derivatives/code/`. Sharing a `derivatives_path` across
*separate* pipeline configurations cannot be detected; avoid it.

### `exclude_pattern`

Optional glob to exclude files matching the pattern from processing. It is
expanded like `file_pattern` (see below).

(relative-paths)=

### Relative paths (changed in 0.4.0)

This is the reference table for every relative path neurodags reads. `~` and `$VAR` are
expanded first; a path that is absolute after expansion is used as is.

| Path | Resolved against |
|---|---|
| paths typed on the command line (pipeline file, `-d/--datasets`, `--inputs`, `-o`) | the current working directory |
| the `datasets:` key and `new_definitions` in `pipeline.yml` | the directory of `pipeline.yml` |
| `source_root`, `file_pattern`, `exclude_pattern`, `derivatives_path` in a datasets YAML (`datasets: datasets.yml`, or `-d datasets.yml`) | the directory of **that datasets YAML** (0.3.x: the current working directory) |
| the same keys in an inline `datasets:` mapping inside `pipeline.yml` | the directory of `pipeline.yml` |
| the same keys in a dict passed to the Python API | the `base_dir=` argument; relative paths without it are an error |

The result does not depend on where you run the command from, and workers always receive
absolute paths. The 0.3.5 fallback that looked a missing `-d` file up next to the pipeline
YAML is removed in 0.4.0.

### Environment variables and `~`

`source_root`, `file_pattern`, `derivatives_path` and `exclude_pattern` (and the `datasets:`
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
- Expansion happens first; a path that is still relative afterwards is resolved as
  described in "Relative paths" above.
- Skipped datasets (`skip: True`) are not expanded or checked, except that they must still
  declare `source_root`.

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

Set `skip: True` to temporarily disable a dataset without removing it. A skipped dataset
contributes no files (and no indices), but must still declare `source_root`.

(datasets-discovery-order)=

## Discovery and file order

For each active dataset, neurodags globs `file_pattern` and keeps only inputs: files (a
symbolic link to a file is kept under its own path) and recording containers, directories that
are one recording each, named `*.ds` (CTF) or `*.mff` (EGI), case-insensitively (the list is
`neurodags.iterators.RECORDING_CONTAINER_EXTENSIONS`). A container is one input: nothing
below a folder with such a name (checked on the path below `source_root`, nested `.ds`
folders included) is ever an input, whatever the pattern, so `raw/**/*` yields
`sub-01_meg.ds` once and `raw/**/*.bin` does not return the files of an `.mff` recording. A
frozen input list naming a path inside a container is refused.
Other directories are never inputs (select a BrainVision recording by its `.vhdr`, not its
folder). Discovery then normalizes each path lexically (`raw/./sub//x` is `raw/sub/x`, and on
POSIX a leading `//` is `/`; symbolic links are not resolved) and removes duplicates (a pattern
with several `**` can match a file more than once), removes `exclude_pattern` matches (normalized the same way, so another
spelling of the same path is still excluded), drops MNE split continuations, removes the
pipeline's own output trees (the dataset's
`derivatives_path[/derivatives_subdir]` when it lies inside `source_root`, and `code/` /
`.neurodags/` under any `derivatives_path`), checks that every file lies below `source_root`, and
**sorts the files by their path relative to `source_root`** (`/` separators, code-point order:
`sub-10` sorts before `sub-2`, so zero-pad labels such as `sub-02`). `max_files_per_dataset`
keeps the first N of that order. Datasets are then concatenated in YAML order and numbered from
`0`: that number is the file index used by `--only-index`, `status` and the SLURM templates.
To keep indices fixed while files are added, freeze the list with `neurodags inputs` (see
{doc}`hpc`).

### `vars`

Dataset-level variables that are substituted into pipeline node args at runtime. Any string arg value matching `$identifier` is replaced by the corresponding value from `vars`.

```yaml
eeg_adhd_epilepsy_EO_baseline:
  name: eeg_adhd_epilepsy_EO_baseline
  source_root:
    local: data
  file_pattern:
    local: data/**/*@clean_prep.fif
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

List any number of datasets. All active ones are processed when running `run_pipeline`, in
YAML order (their files are numbered one after the other). Datasets that share a
`derivatives_path` need distinct `derivatives_subdir` values; here each has its own folder:

```yaml
dataset1:
  name: StudyA
  source_root:
    local: data/studyA
  file_pattern:
    local: data/studyA/**/*.vhdr
  derivatives_path:
    local: outputs/studyA/

dataset2:
  name: StudyB
  source_root:
    local: data/studyB
  file_pattern:
    local: data/studyB/**/*.vhdr
  derivatives_path:
    local: outputs/studyB/
```

## Referencing in pipeline.yml

```yaml
# pipeline.yml
naming_version: 1         # required since 0.4.0
datasets: datasets.yml    # path to datasets file
mount_point: local        # which mount point to activate
```

`datasets` can be an absolute or relative path. Relative paths are resolved from the pipeline
YAML location; paths inside the datasets YAML are then resolved from the datasets YAML's
location.
