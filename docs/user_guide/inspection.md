# Inspection and Visualization

```{note}
**Output streams.** The inspection commands (`status`, `validate`, `dag`) write
their result — the table, `--format json`, or the Mermaid text — to **stdout**,
while all framework logs go to **stderr**. So the deliverable is always clean and
pipeable, e.g. `neurodags status pipeline.yml --format json 2>/dev/null` yields a
single valid JSON document. Increase verbosity with `--log-level DEBUG` (still on
stderr).
```

## Dry Run Mode

Inspect planned computations without executing any nodes. Returns a dataframe describing what would run and whether cached outputs already exist.

```python
from neurodags.orchestrators import iterate_derivative_pipeline, run_pipeline

# Pass the path: relative paths then resolve against the YAML files. A dict loaded
# in Python needs base_dir=... for relative dataset paths (0.4.0).
config = "pipeline.yml"

# Dry-run all derivatives in DerivativeList
plan = run_pipeline(config, dry_run=True)
print(plan)

# Or dry-run a single derivative
plan = iterate_derivative_pipeline(config, "BandPower", dry_run=True)
print(plan)
```

CLI equivalent:

```bash
neurodags dry-run pipeline.yml --derivative BandPower --output dry_run_results.csv
```

The dry-run plan shows each step, whether its output is cached, and where it would write. Use this to:

- Verify the DAG structure before a long run
- Identify which files need recomputation
- Debug path resolution issues

## Failure Behavior

When a node fails during execution (or anything else inside the unit: a derivative it depends
on, preparing a node's arguments, selecting a referenced artifact, writing an artifact):

1. The error is logged with full traceback, derivative name, step id, and file path.
2. A `.error` marker file is written at the expected output location (e.g. `sub-01_task-rest_eeg.vhdr@MyDerivative.error`).  Any `.done` of the unit is removed, so the unit is `errored`, never `done`. Its content is a human-readable summary of the failure — useful for post-hoc inspection without digging through logs.
3. By default (`raise_on_error=False`) execution continues to the next file; all failures are collected in the log.
4. The failed file **will be retried** on the next run — only a valid `.done` manifest makes a unit cached, and a failure removes it, so NeuroDAGs re-runs the unit.

To inspect which files failed after a run, look for `*.error` files in the derivatives directory:

```bash
find derivatives/ -name "*.error"
```

Each `.error` file contains:

```
Derivative 'MyDerivative' step id=2 node='my_node' failed:
<exception message>
```

### Automatic Cleanup of Stale Error Markers

When a file that previously failed is re-run and **succeeds**, NeuroDAGs automatically deletes the stale `.error` marker once the new `.done` manifest is published (a unit carries at most one of `.done`, `.error`, `.skip`). This keeps the derivatives directory clean and ensures dry-run plans accurately reflect current state — a file that succeeded on re-run will not appear as `has_error_marker: true` on subsequent runs.

### Skipping Previously Failed Files

To skip files that already have a `.error` marker (and avoid retrying known failures), use `skip_errors=True`:

```python
run_pipeline(config, skip_errors=True)
```

CLI:

```bash
neurodags run pipeline.yml --skip-errors
neurodags dry-run pipeline.yml --skip-errors   # shows which files would be skipped
```

Skipped files are logged and reported in the dry-run plan with `has_error_marker: true`. To retry a skipped file, delete its `.error` marker:

```bash
rm derivatives/sub-01/ses-SE0/sub-01_ses-SE0_task-rest.vhdr@MyDerivative.error
```

## Pipeline Status

`neurodags status` gives a quick at-a-glance summary of how many units (source file × derivative) are done, skipped, incomplete, missing (not yet computed), or errored — without having to open a CSV. The states are the ones defined in {ref}`Unit states <naming-unit-states>`.

```bash
neurodags status pipeline.yml
```

Example output:

```
config: /abs/path/pipeline.yml
files:  42

Derivative                total    done  (migrated)   skipped  incomplete   missing   errored
─────────────────────────────────────────────────────────────────────────────────────────────
Alpha                        42      29          20         0           1        10         2
Beta                         42      25          20         0           0        15         2
─────────────────────────────────────────────────────────────────────────────────────────────
Total                        84      54          40         0           1        25         4

40 of the done unit(s) are done (migrated): their .done was written by 'neurodags migrate' from 0.3.x outputs (column '(migrated)').

4 error(s) found.  Run with --list-errors for details.
1 derivative(s) incomplete (no valid .done manifest).  Run with --list-missing for details.
25 derivative(s) missing.  Run with --list-missing for details.
Use --format indices to get the file indices for --only-index resubmission.
```

The `(migrated)` column appears only when some done units were certified by
`neurodags migrate`; it is a subset of `done`, not a separate state. The table also prints a
note explaining the columns (omitted above).

**Status definitions:**

| Status | Meaning |
|--------|---------|
| `done` | The `.done` manifest is valid: it matches the derivative and naming version, and every file it lists exists |
| `done (migrated)` | `done`, with a manifest written by `neurodags migrate` (counted in `done` and in the `(migrated)` column) |
| `skipped` | A `.skip` marker exists (the derivative does not apply to this file) |
| `incomplete` | Output files (or an invalid `.done`) exist without a valid manifest: an interrupted or partial write, a deleted split part, or 0.3.x outputs not yet migrated with `neurodags migrate`. Recomputed by the next run |
| `missing` | Nothing on disk for this file — not yet computed |
| `errored` | A `.error` marker file exists from a previous failed run |
| `locked` | A `.lock` file exists and there is no valid `.done`: a job is computing the unit right now, or one was killed while computing it. Runs refuse the unit until the lock is gone. Not a table column: the table is followed by the list of lock files, with a hint to delete them once no job is running for those units (see {ref}`Unit lock <naming-unit-lock>`) |

Files that match an output prefix but are not listed in its valid `.done` manifest are ignored;
the table prints how many there are and `--include-files` lists them per unit (`unlisted`). See
{ref}`Completion manifest <naming-done-manifest>`.

**Flags:**

```bash
# show only specific derivatives
neurodags status pipeline.yml --derivative Alpha --derivative Beta

# print the file index and path of each errored file
neurodags status pipeline.yml --list-errors

# print the file index and path of each incomplete and missing file
neurodags status pipeline.yml --list-missing

# combine both
neurodags status pipeline.yml --list-errors --list-missing

# parallelize the underlying dry-run (same semantics as 'run')
neurodags status pipeline.yml --n-jobs 4
neurodags status pipeline.yml --n-jobs -1   # all cores
```

Each listed file is prefixed with its global **file index** — the same index that
`neurodags run --only-index` and the SLURM array templates consume — and files are sorted by
index within each derivative. With `--list-errors` the `.error` marker file path is shown
alongside each failed input, making it easy to inspect or delete:

```
Errored files:
  [Alpha]
    [idx 0] /data/sub-01/ses-01/sub-01_task-rest.vhdr
      error file: /derivatives/sub-01/ses-01/sub-01_task-rest.vhdr@Alpha.error

Missing files:
  [Alpha]
    [idx 7] /data/sub-02/ses-01/sub-02_task-rest.vhdr
    [idx 12] /data/sub-05/ses-01/sub-05_task-rest.vhdr
```

The index is a position in the input list: datasets in config order, files sorted by their
path below `source_root` (code-point order, so `sub-10` before `sub-2`). Without `--inputs`
it identifies the same file only while the dataset configuration and the set of input files
are unchanged — regenerate it right before resubmitting, don't store it. With a frozen list
(`neurodags status pipeline.yml --inputs inputs.tsv`, see {doc}`hpc`) the indices are those
of the list; the report then also says how many files on disk are not in the list, and
lists listed files that no longer exist as `missing-source` (they are never part of
`--format indices`). If an index cannot be determined it is shown as `[idx ?]`.

**Exit codes:** `0` only when all derivatives are complete (no missing, no incomplete, no errored, no locked unit, no `missing-source` input); `1` when any derivative is missing, incomplete, locked or has a `.error` marker, or a listed input is `missing-source`. `--format indices` does not list locked units (resubmitting them would only hit the lock again). `done`, `done (migrated)` and `skipped` units count as complete. This makes `status` usable in shell scripts and CI dependency chains:

```bash
# fail fast if anything is incomplete or errored
neurodags status pipeline.yml || sbatch resubmit.sh
```

**Machine-readable output:** `--format json` emits a structured JSON object instead of the table:

```bash
neurodags status pipeline.yml --format json
```

```json
{
  "config": "pipeline.yml",
  "n_files": 42,
  "derivatives": {
    "Alpha": {"total": 42, "done": 29, "skipped": 0, "incomplete": 1, "missing": 10,
              "errored": 2, "missing-source": 0, "locked": 0, "migrated": 20}
  },
  "grand_total": {"total": 42, "done": 29, "skipped": 0, "incomplete": 1, "missing": 10,
                  "errored": 2, "missing-source": 0, "locked": 0, "migrated": 20},
  "complete": false,
  "missing_indices": [3, 7, 12, 15, 18, 21, 25, 30, 33, 40],
  "incomplete_indices": [5],
  "errored_indices": [0, 9],
  "locked_indices": [],
  "unlisted_files": 0,
  "inputs": null,
  "missing_source_indices": [],
  "not_in_list": 0
}
```

`migrated` counts the done units whose manifest was written by `neurodags migrate` (a subset of
`done`). `inputs` is the
absolute path of the `--inputs` list (`null` without one); `missing_source_indices` and
`not_in_list` are only non-empty / non-zero with `--inputs`.

`missing_indices` / `incomplete_indices` / `errored_indices` are the sorted, de-duplicated
file indices with at least one missing / incomplete / errored derivative (across the selected derivatives; narrow with
`--derivative`). A file can appear in several lists if it is missing for one derivative and
errored for another (or in all three).

Add `--include-files` (only valid with `--format json`) for a per-file breakdown: a flat
`files` list with one record per *(source file, derivative)* pair, sorted by `(index,
derivative)`:

```bash
neurodags status pipeline.yml --format json --include-files
```

```json
  "files": [
    {"index": 0, "dataset": "my_dataset", "file_path": "/data/sub-01/ses-01/sub-01_task-rest.vhdr",
     "derivative": "Alpha", "status": "errored", "label": "errored",
     "error_path": "/derivatives/sub-01/ses-01/sub-01_task-rest.vhdr@Alpha.error",
     "unlisted": []},
    {"index": 1, "dataset": "my_dataset", "file_path": "/data/sub-02/ses-01/sub-02_task-rest.vhdr",
     "derivative": "Alpha", "status": "done", "label": "done (migrated)", "error_path": null,
     "unlisted": []}
  ]
```

`label` is `status`, except `done (migrated)` for a done unit certified by `neurodags migrate`
(`status` stays `done`, so consumers counting `done` keep working). `lock_path` is the unit's
`.lock` file when one exists, else `null`. `status` is one of `done`, `skipped`, `incomplete`, `missing`, `errored`, `locked`, `no-save`, `unknown`, or
`missing-source` (with `--inputs`: listed, but no longer on disk). When no
input files are found, `--format json` still emits a valid document (zero counts, empty
index lists, `"complete": true`).

**Index-only output:** `--format indices` prints nothing but the sorted, space-separated file
indices that are missing, incomplete or errored — ready to paste into `--only-index` or
`sbatch --array`. Narrow it with `--list-missing` (missing and incomplete: both are recomputed
the same way) or `--list-errors` (errored only); with neither or both flags it prints the union.
A complete pipeline prints an empty line.

```bash
neurodags status pipeline.yml --format indices                  # e.g. "0 3 7 9 12"
neurodags status pipeline.yml --format indices --list-errors    # errored only: "0 9"
neurodags status pipeline.yml --derivative Alpha --format indices
```

The exit code keeps its meaning in every format: `1` whenever something is missing,
incomplete or errored (or, with `--inputs`, a listed input is `missing-source`). Under `set -e`, capturing it with `$(...)`
would abort the script precisely when there is something to resubmit, so add `--exit-zero`:

```bash
IDX=$(neurodags status pipeline.yml --format indices --exit-zero)
[ -n "$IDX" ] && neurodags run pipeline.yml --only-index $IDX
```

With `--exit-zero`, `status` exits `0` whenever the report was produced, but real failures
(missing or invalid config, unset environment variable, unreadable `-d` file) still exit
non-zero, so `set -e` still stops on a broken config. Avoid the older `|| true` guard: it
also swallows those real failures, leaving `IDX` silently empty.
See {doc}`hpc` for resubmitting through SLURM.

## DAG Visualization (Mermaid)

NeuroDAGs can render pipeline and derivative graphs as interactive
[Mermaid](https://mermaid.js.org/) diagrams saved to standalone HTML files.

**Pipeline-level overview** — one node per derivative, edges show inter-derivative
dependencies:

```python
import yaml
from neurodags.mermaid import pipeline_to_html

with open("pipeline.yml") as f:
    config = yaml.safe_load(f)

pipeline_to_html(config, output_path="pipeline_dag.html", auto_open=True)
```

**Derivative-level detail** — every computation node and data reference inside
one derivative:

```python
from neurodags.mermaid import derivative_to_html

derivative_to_html(
    config["DerivativeDefinitions"]["BandPower"],
    "BandPower",
    output_path="bandpower_dag.html",
    auto_open=True,
)
```

Node shapes used in derivative diagrams:

| Shape | Meaning |
|-------|---------|
| Circle `(((...)))` | `SourceFile` — raw input file |
| Cylinder `[(...)]` | Upstream derivative artifact (cached on disk) |
| Rectangle `[...]` | Computation node |

To get the raw Mermaid string (e.g. for embedding in Jupyter or custom HTML):

```python
from neurodags.mermaid import pipeline_to_mermaid, derivative_to_mermaid

print(pipeline_to_mermaid(config))
print(derivative_to_mermaid(config["DerivativeDefinitions"]["BandPower"], "BandPower"))
```

**Layout engines:**

HTML output uses the ELK layout engine by default — orthogonal edge routing with active crossing minimisation, significantly cleaner than curved edges for dense pipelines. ELK loads its bundle from the CDN so internet access is required when opening the HTML. Use `--layout dagre` (or `layout="dagre"`) for offline use; dagre renders with right-angle step edges and has no CDN dependency.

```python
# ELK layout via Python API
pipeline_to_html(config, output_path="pipeline_dag.html", layout="elk")
derivative_to_html(deriv_def, "BandPower", output_path="bp.html", layout="elk")
```

CLI:

```bash
neurodags dag pipeline.yml
neurodags dag pipeline.yml --html pipeline_dag.html
neurodags dag pipeline.yml --derivative BandPower --html bandpower_dag.html
neurodags dag pipeline.yml --html pipeline_dag.html --layout elk   # ELK for dense graphs
```

See the {doc}`../auto_examples/plot_mermaid_visualization` example for a
complete walkthrough.

## Interactive File Explorer

Built-in Dash-Plotly explorer for `.fif` (MNE) and `.nc` (NetCDF/xarray) files:

```bash
neurodags view path/to/file.fif
neurodags view path/to/file.nc

# Alternative module entry point
python -m neurodags.visualization path/to/file.fif
python -m neurodags.visualization path/to/file.nc
```

Features:
- Variable selector — when the `.nc` file is a multi-variable Dataset, a dropdown lets you switch between variables interactively without restarting
- Dimension-aware UI — per-variable dropdowns to slice along any dimension
- Plot types: Line, Scatter, Bar, Heatmap
- Works with both `xr.DataArray` and `xr.Dataset` files produced by NeuroDAGs
