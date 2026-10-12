# CLI Reference

NeuroDAGs installs a unified `neurodags` command with several subcommands. Every subcommand that accepts a pipeline file also accepts `-d/--datasets <path>` to override the datasets YAML defined inside the pipeline file — useful when the same pipeline is run against different dataset collections. See *Per-subcommand Dataset Override* at the end of this page for how that path is resolved.

```{important}
**Which directory a relative path is resolved against:**

- **Paths you type on the command line** (the pipeline file, `-d/--datasets`): the current working directory.
- **The `datasets:` key inside `pipeline.yml`**: the directory of `pipeline.yml`.
- **Paths inside a datasets YAML** (`source_root`, `file_pattern`, `exclude_pattern`, `derivatives_path`): the directory of **that datasets YAML** (changed in 0.4.0; in 0.3.x they were relative to the current working directory).
- **The same keys in an inline `datasets:` mapping inside `pipeline.yml`**: the directory of `pipeline.yml`.
- **The same keys in a dict passed to the Python API**: the `base_dir=` argument (relative paths without it are an error).

`~` and `$VARS` are expanded first; a path that is absolute after expansion is used as is. Full table: {ref}`Relative paths <relative-paths>`.
```

```{admonition} Changed in 0.4.0
:class: important

Every pipeline YAML must declare `naming_version: 1`. A configuration without it (or with
another value) is refused by every command, the TUI and the Python API with a one-line error
(exit code `2` on the CLI); to upgrade a 0.3.x configuration, run `neurodags migrate`
(see {doc}`migrating_to_0_4`). See {doc}`pipeline_yaml`.
```

## Global Options

These flags apply to every subcommand and must be placed *before* the subcommand name:

```bash
neurodags --log-level WARNING run pipeline.yml       # suppress INFO output
neurodags --log-level DEBUG run pipeline.yml         # verbose output
neurodags --log-file run.jsonl run pipeline.yml      # also write logs to JSONL file
neurodags --log-level WARNING --log-file run.jsonl run pipeline.yml
```

| Flag | Default | Description |
|------|---------|-------------|
| `--log-level LEVEL` | `INFO` (or `$LOG_LEVEL`) | Console verbosity: `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `--log-file PATH` | none | Write all log events to PATH in JSONL format (one JSON object per line) |

The JSONL log file can be loaded directly as a dataframe:

```python
import pandas as pd
df = pd.read_json("run.jsonl", lines=True)
# columns: event, level, logger, timestamp, ... plus any bound context keys
```

## Validation

Load and summarise the configuration without running anything. Prints, for each active dataset, the active mount point and the resolved `source_root`, `file_pattern`, `exclude_pattern`, `derivatives_path` (and `derivatives_subdir` with the resulting output folder), then every derivative *defined* in the config, and the **effective run set** — the selected derivatives plus the intermediates that will be computed as their dependencies (what a `run` actually produces). Derivatives that are defined but not part of the run are listed separately, and `for_dataframe` outputs are flagged.

```bash
neurodags validate pipeline.yml                       # summary for the default run set (DerivativeList)
neurodags validate pipeline.yml -d alt.yml            # override datasets
neurodags validate pipeline.yml --derivative BandPower  # effective set for a specific selection
```

Pass `--derivative` (repeatable) to preview exactly what a `run --derivative …` would compute: dependencies are auto-resolved into `computed_with_dependencies`, while anything else defined shows under `not computed by this run`.

An unversioned configuration or invalid derivative names stop `validate` like every other command (one `neurodags: error: …` message listing every problem, exit code `2`); name warnings (names over 64 characters, repeated key-value keys) are printed without failing. Dataset configuration errors are printed to stderr, one `error: …` line each, and `validate` exits with code `1`: a missing `source_root` (with a suggested value derived from `file_pattern` when one exists), a `source_root` that is not an existing directory, a path that does not resolve for the active mount point, an invalid `derivatives_subdir`, or two active datasets sharing a `derivatives_path` with the same `derivatives_subdir`. `run`, `dry-run`, `status`, `count-inputs` and `dataframe` refuse the same configurations before touching any file (exit code `2`).

```text
  study:
    mount: local
    source_root: None
    file_pattern: /data/study/rawdata/**/*.vhdr
    derivatives_path: /data/study/derivatives
...
error: datasets.study.source_root is required (since 0.4.0): set it to the root folder of the dataset's source files; suggested for mount 'local': /data/study/rawdata
```

See {doc}`pipeline_yaml` for all pipeline keys and {doc}`datasets_yaml` for dataset fields.

## Execution

Run derivatives defined in `DerivativeList`. When no `--derivative` flag is given, all derivatives in `DerivativeList` are run in dependency order.

```bash
neurodags run pipeline.yml                          # run all derivatives in DerivativeList
neurodags run pipeline.yml --derivative CleanedEEG  # run one derivative
neurodags run pipeline.yml --derivative A --derivative B  # run several
```

**Parallelism:**

```bash
neurodags run pipeline.yml --n-jobs 4          # 4 parallel workers
neurodags run pipeline.yml --n-jobs -1         # all cores
neurodags run pipeline.yml --n-jobs 4 --joblib-backend loky --joblib-prefer processes
```

**Subset / error control:**

```bash
neurodags run pipeline.yml --max-files-per-dataset 10   # first 10 files (sorted) per dataset
neurodags run pipeline.yml --only-index 0 5 12   # process only files at these indices (unknown index = error)
neurodags run pipeline.yml --inputs inputs.tsv   # use a frozen input list instead of discovery
neurodags run pipeline.yml --skip-errors          # skip files that already have a .error marker
neurodags run pipeline.yml --raise-on-error       # stop immediately on first failure
```

**Config snapshot (automatic provenance):**

Every `neurodags run` call automatically copies the pipeline configuration files into a
`code/` subdirectory inside each dataset's `derivatives_path` before any derivatives are
executed.  This gives you a record of exactly what was run and with which version of the
pipeline.

Files written to `derivatives_path/code/`:

| File | Description |
|------|-------------|
| `<pipeline>.yml` | The pipeline YAML passed to `run` |
| `<new_definitions>.py` | Any `new_definitions:` Python file(s) listed in the pipeline |
| `<datasets>.yml` | The resolved datasets YAML (from the pipeline or `-d` override) |
| `neurodags_env.json` | Installed neurodags version, git commit of the source repo (if available), and UTC timestamp |

Example `neurodags_env.json`:

```json
{
  "snapshot_time": "2026-05-21T08:00:00.000000+00:00",
  "neurodags_version": "0.4.0",
  "neurodags_git_commit": "a1b2c3d4..."
}
```

The snapshot runs unconditionally and overwrites any prior snapshot in `code/` — it always
reflects the config that was active for the most recent `run` call.  Snapshot failures
(e.g. read-only filesystem) are logged as warnings and never block derivative execution.
The snapshot is **skipped** for `neurodags dry-run`.

See {doc}`parallelism` for full details on parallel execution and error handling.

## Dry Run

Inspect the planned execution without running any nodes. Reports each file, derivative, and whether the output is already cached. Useful for verifying the DAG before a long run, checking which files need recomputation, or debugging path issues.

```bash
neurodags dry-run pipeline.yml                             # all derivatives
neurodags dry-run pipeline.yml --derivative CleanedEEG    # one derivative
neurodags dry-run pipeline.yml --output plan.csv          # save to CSV
neurodags dry-run pipeline.yml --output plan.parquet      # or Parquet
neurodags dry-run pipeline.yml --n-jobs 4                 # parallel file resolution
neurodags dry-run pipeline.yml --skip-errors              # exclude errored files
```

See {doc}`inspection` for the plan format and how to interpret cached / missing / errored states.

## Status

Quick at-a-glance summary of done / skipped / incomplete / missing / errored counts per derivative — no CSV required. The states are defined in {ref}`Unit states <naming-unit-states>`.

```bash
neurodags status pipeline.yml                        # summary table for all derivatives
neurodags status pipeline.yml --derivative Alpha     # filter to one derivative
neurodags status pipeline.yml --list-errors          # errored files: [idx N] path + .error path
neurodags status pipeline.yml --list-missing         # incomplete and missing files: [idx N] path
neurodags status pipeline.yml --list-errors --list-missing
neurodags status pipeline.yml --n-jobs 4             # parallelize the underlying dry-run
neurodags status pipeline.yml --format json          # machine-readable JSON output
neurodags status pipeline.yml --format json --include-files   # + per-file records
neurodags status pipeline.yml --format indices       # missing/incomplete/errored indices for --only-index
neurodags status pipeline.yml --inputs inputs.tsv    # check against a frozen input list
```

The table has one row per derivative with the columns `total`, `done`, `skipped`,
`incomplete`, `missing` and `errored` (plus `(migrated)` when some done units were certified
by `neurodags migrate`); see {doc}`inspection` for a full example and the JSON format.

Exit code `0` only when all derivatives are complete (nothing missing, incomplete or errored, and with `--inputs` no `missing-source` input); `1` otherwise — suitable for CI and shell dependency chains:

```bash
neurodags status pipeline.yml || sbatch resubmit.sh
```

Pass `--exit-zero` to use `status` purely as a report: it exits `0` whenever the report was produced, even if files are missing, incomplete or errored, while real failures (missing or invalid config, unset environment variable, unreadable `-d` file) still exit non-zero. This is what a report job chained with `--dependency=afterok` wants (see {doc}`hpc`):

```bash
neurodags status pipeline.yml --exit-zero                  # report only; 0 even if incomplete
neurodags status pipeline.yml --format indices --exit-zero # works with every --format
```

See {doc}`inspection` for status definitions and `.error` marker behaviour.

## File Indices and Input Lists

Every command enumerates its inputs once: per dataset, the discovered files are sorted by their
path relative to `source_root` (`/` separators, code-point order: `sub-10` sorts before `sub-2`,
so zero-pad labels such as `sub-02` for a natural order), `--max-files-per-dataset` keeps the
first N, and the datasets are concatenated in YAML order (`skip: true` ones left out). The
position in that list is the file **index** used by `--only-index`, `status` and the SLURM
array templates. A negative or unknown `--only-index` value is an error (exit code `2`).

`neurodags inputs` freezes that list in a file, so the indices cannot change while you work
with it (new files on disk are ignored):

```bash
neurodags inputs pipeline.yml -o inputs.tsv        # write the list (omit -o to print it)
neurodags inputs pipeline.yml -d other.yml -o inputs.tsv
```

The file starts with `# neurodags-inputs naming_version=1`, followed by one
`index<TAB>dataset<TAB>rel` line per input (`rel`: path relative to the dataset's
`source_root`, `/` separators), so the same list works on every mount point.

`run`, `dry-run`, `status`, `count-inputs` and `dataframe` accept `--inputs FILE` (relative to
the current working directory; Python API: `inputs=`). The list replaces discovery; files on
disk that it does not contain are reported and skipped, and listed files that no longer exist
are reported as `missing-source` at their index (`status` shows them, `run` skips them,
`--only-index` on them is an error). `--inputs` cannot be combined with
`--max-files-per-dataset`. See {doc}`hpc` for the array-job workflow.

## Source File Count

Print the number of input entries, i.e. the size of the `--only-index` range (`0` to `N-1`). Useful for sanity-checking datasets before a long run and for sizing SLURM arrays. Note: this counts input entries, not output files — one input file may produce multiple output files depending on the derivatives, and a file listed under two dataset entries counts twice.

```bash
neurodags count-inputs pipeline.yml                        # number of input entries (discovery)
neurodags count-inputs pipeline.yml --inputs inputs.tsv    # number of entries in a frozen list
```

`--derivative` is still accepted but does not change the count.

## Dataframe Assembly

Collect derivatives marked `for_dataframe: True` into a flat CSV or Parquet file, one row per file (wide) or one row per value (long).

```bash
neurodags dataframe pipeline.yml --format wide --output features.csv
neurodags dataframe pipeline.yml --format long --output features.parquet
neurodags dataframe pipeline.yml --include-derivative PowerSpectrum --include-derivative BandPower
neurodags dataframe pipeline.yml --max-files-per-dataset 5
neurodags dataframe pipeline.yml --n-jobs 4    # parallel file-level collection
neurodags dataframe pipeline.yml --n-jobs -1   # all cores
```

Parallelism is per-file using separate processes (loky backend). Threading is intentionally avoided because NetCDF4/HDF5 is not thread-safe — concurrent thread access to `.nc` files causes `[Errno -101] HDF error`.

See {doc}`dataframe_assembly` for format details and how to mark derivatives for dataframe inclusion.

## DAG Visualization

Render the pipeline or a single derivative as a Mermaid diagram. The pipeline-level view shows one node per derivative with inter-derivative edges; the derivative-level view shows every computation node inside one derivative.

```bash
neurodags dag pipeline.yml                                        # print Mermaid text to stdout
neurodags dag pipeline.yml --html pipeline_dag.html               # export to standalone HTML
neurodags dag pipeline.yml --html pipeline_dag.html --open        # export and open in browser
neurodags dag pipeline.yml --derivative CleanedEEG --html d.html  # single-derivative DAG
neurodags dag pipeline.yml --html pipeline_dag.html --layout elk  # ELK layout for dense graphs
```

HTML output uses the ELK layout engine by default — orthogonal edge routing with active crossing minimisation, significantly cleaner than curved edges for dense pipelines. ELK requires internet access to load its bundle from the CDN. Use `--layout dagre` for offline use (right-angle step edges, no CDN dependency).

See {doc}`inspection` for a full walkthrough of DAG visualization.

## File Explorer

Launch an interactive Dash-Plotly browser for `.fif` (MNE) and `.nc` (NetCDF/xarray) files.

```bash
neurodags view path/to/file.fif   # MNE raw / epochs explorer
neurodags view path/to/file.nc    # xarray DataArray / Dataset explorer
```

Features: variable selector for multi-variable Datasets, dimension-aware slicing dropdowns, plot types: Line, Scatter, Bar, Heatmap.

See {doc}`inspection` for the full feature list.

## SLURM / HPC Scripts

Generate ready-to-submit SLURM array job scripts. Three submission patterns are available:

| Pattern | Description |
|---------|-------------|
| `per-file` (default) | One array job per pipeline run; each task processes one file across all derivatives |
| `flat` | One array job where each task is a unique (file, derivative) pair |
| `chained` | One array job per derivative, chained with `--dependency=afterok` in topological order |

```bash
neurodags slurm-script pipeline.yml                              # per-file (default)
neurodags slurm-script pipeline.yml --pattern flat
neurodags slurm-script pipeline.yml --pattern chained
neurodags slurm-script pipeline.yml --output run_array.sh        # write to file
neurodags slurm-script pipeline.yml --derivative CleanedEEG      # restrict to one derivative
```

See {doc}`hpc` for full details on each pattern and how to submit.

## Migrating a 0.3.x Project

A 0.3.x configuration is refused until it is migrated. `neurodags migrate` adds `naming_version: 1` and `source_root`, rewrites relative dataset paths, renames and moves the existing outputs to the 0.4.0 names and layout, and writes `.done` manifests for them:

```bash
neurodags migrate pipeline.yml --report plan.json        # preview: prints the plan, changes nothing
neurodags migrate pipeline.yml --apply                   # rewrite config, move outputs, write .done
neurodags migrate pipeline.yml --mount hpc --apply       # then each other mount's output tree
```

Exit code `2` when the plan is refused (nothing is changed). See {doc}`migrating_to_0_4`.

## TUI (Terminal User Interface)

Requires `pip install neurodags[tui]`. Provides tabs for configuration, execution, dry-run, status, dataframe assembly, DAG visualization, and file inspection — all without leaving the terminal.

```bash
neurodags tui                          # launch empty, load config interactively
neurodags tui pipeline.yml             # launch with config pre-loaded
neurodags tui pipeline.yml -d alt.yml  # with datasets override
```

See {doc}`tui` for a full walkthrough.

## Per-subcommand Dataset Override

All subcommands that take a pipeline file also accept:

| Flag | Description |
|------|-------------|
| `-d / --datasets <path>` | Override the `datasets` key in the pipeline YAML. The path is relative to the current working directory |

How the `-d` path is resolved (`validate`, `run`, `dry-run`, `dataframe`, `count-inputs`, `inputs`, `status`, `slurm-script`, `migrate`, `tui`, and the legacy `neurodags-tui`):

1. `~` and `$VARS` are expanded, exactly as for the `datasets:` key (an unset or empty variable is an error).
2. A relative path is resolved against the **current working directory** — the directory you run `neurodags` from, like every other CLI path.
3. If the file does not exist there, the command stops with a short error naming the path it tried and exits with code `2` (no traceback). **Removed in 0.4.0:** the YAML-relative fallback (deprecated in 0.3.5) that looked the file up next to the pipeline YAML.

Paths written *inside* the datasets file passed with `-d` are resolved against that file's own directory (0.4.0; `neurodags migrate` reads a 0.3.x file with the 0.3.x rule, relative to `--legacy-cwd`). The `datasets:` key written *inside* `pipeline.yml` is unaffected: it stays relative to the pipeline YAML.

```bash
# run from the repository root; the pipeline lives in pipelines/x/
neurodags run pipelines/x/pipeline.yml -d pipelines/x/datasets_alt.yml   # cwd-relative (correct)
neurodags run pipelines/x/pipeline.yml -d datasets_alt.yml               # error since 0.4.0 (not in the cwd)
```

In the Python API (`run_pipeline`, `iterate_derivative_pipeline`, `build_derivative_dataframe`), a relative `datasets_configuration` string keeps the library rule — relative to the pipeline YAML's directory, like the `datasets:` key (relative to `base_dir=` for an in-memory pipeline). Pass an absolute path to avoid ambiguity. A `datasets_configuration` dict with relative paths needs `base_dir=`.
