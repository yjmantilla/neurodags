# CLI Reference

NeuroDAGs installs a unified `neurodags` command with several subcommands. Every subcommand that accepts a pipeline file also accepts `-d/--datasets <path>` to override the datasets YAML defined inside the pipeline file — useful when the same pipeline is run against different dataset collections. See *Per-subcommand Dataset Override* at the end of this page for how that path is resolved.

```{important}
**CLI paths are relative to where you run the command; paths inside the YAML are relative to the YAML.** `-d/--datasets` is resolved against the current working directory (like any other CLI path), while the `datasets:` key *inside* `pipeline.yml` is resolved against the directory of `pipeline.yml`.
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

Load and summarise the configuration without running anything. Prints the resolved datasets, every derivative *defined* in the config, and the **effective run set** — the selected derivatives plus the intermediates that will be computed as their dependencies (what a `run` actually produces). Derivatives that are defined but not part of the run are listed separately, and `for_dataframe` outputs are flagged.

```bash
neurodags validate pipeline.yml                       # summary for the default run set (DerivativeList)
neurodags validate pipeline.yml -d alt.yml            # override datasets
neurodags validate pipeline.yml --derivative BandPower  # effective set for a specific selection
```

Pass `--derivative` (repeatable) to preview exactly what a `run --derivative …` would compute: dependencies are auto-resolved into `computed_with_dependencies`, while anything else defined shows under `not computed by this run`.

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
neurodags run pipeline.yml --max-files-per-dataset 10
neurodags run pipeline.yml --only-index 0 5 12   # process only files at these indices
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
  "neurodags_version": "0.1.0",
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

Quick at-a-glance summary of done / missing / errored counts per derivative — no CSV required.

```bash
neurodags status pipeline.yml                        # summary table for all derivatives
neurodags status pipeline.yml --derivative Alpha     # filter to one derivative
neurodags status pipeline.yml --list-errors          # errored files: [idx N] path + .error path
neurodags status pipeline.yml --list-missing         # missing files: [idx N] path
neurodags status pipeline.yml --list-errors --list-missing
neurodags status pipeline.yml --n-jobs 4             # parallelize the underlying dry-run
neurodags status pipeline.yml --format json          # machine-readable JSON output
neurodags status pipeline.yml --format json --include-files   # + per-file records
neurodags status pipeline.yml --format indices       # missing/errored indices for --only-index
```

Example output:

```
config: /abs/path/pipeline.yml
files:  42

Derivative               total   done  missing  errored
───────────────────────────────────────────────────────
Alpha                       42     30       10        2
Beta                        42     25       15        2
───────────────────────────────────────────────────────
Total                       84     55       25        4

2 error(s) found.  Run with --list-errors for details.
4 derivative(s) missing.  Run with --list-missing for details.
Use --format indices to get the file indices for --only-index resubmission.
```

Exit code `0` only when all derivatives are complete (no missing, no errored); `1` otherwise — suitable for CI and shell dependency chains:

```bash
neurodags status pipeline.yml || sbatch resubmit.sh
```

Pass `--exit-zero` to use `status` purely as a report: it exits `0` whenever the report was produced, even if files are missing or errored, while real failures (missing or invalid config, unset environment variable, unreadable `-d` file) still exit non-zero. This is what a report job chained with `--dependency=afterok` wants (see {doc}`hpc`):

```bash
neurodags status pipeline.yml --exit-zero                  # report only; 0 even if incomplete
neurodags status pipeline.yml --format indices --exit-zero # works with every --format
```

See {doc}`inspection` for status definitions and `.error` marker behaviour.

## Source File Count

Print the number of unique source (input) files the pipeline will process. Useful for sanity-checking datasets before a long run. Note: this counts input files, not output files — one input file may produce multiple output files depending on the derivatives.

```bash
neurodags count-inputs pipeline.yml                          # number of source files across all derivatives
neurodags count-inputs pipeline.yml --derivative CleanedEEG  # count for a specific derivative
```

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

How the `-d` path is resolved (`validate`, `run`, `dry-run`, `dataframe`, `count-inputs`, `status`, `slurm-script`, `tui`, and the legacy `neurodags-tui`):

1. `~` and `$VARS` are expanded, exactly as for the `datasets:` key (an unset or empty variable is an error).
2. A relative path is resolved against the **current working directory** — the directory you run `neurodags` from, like every other CLI path.
3. **Deprecated fallback:** if the file does not exist there but does exist relative to the pipeline YAML's directory (the behaviour up to 0.3.4), that file is used and one warning line is printed to stderr with the cwd-relative path to pass instead, e.g. `neurodags: warning: -d 'datasets_alt.yml' was found relative to the pipeline YAML; this lookup is deprecated and will be removed in a future release. Pass '-d pipelines/x/datasets_alt.yml' instead.` This fallback will be removed in a future release.
4. If neither exists, the command stops with a short error naming both paths it tried and exits with code `2` (no traceback).

The `datasets:` key written *inside* `pipeline.yml` is unaffected: it stays relative to the pipeline YAML.

```bash
# run from the repository root; the pipeline lives in pipelines/x/
neurodags run pipelines/x/pipeline.yml -d pipelines/x/datasets_alt.yml   # cwd-relative (correct)
neurodags run pipelines/x/pipeline.yml -d datasets_alt.yml               # YAML-relative: still works, with a deprecation warning
```

In the Python API (`run_pipeline`, `iterate_derivative_pipeline`, `build_derivative_dataframe`), a relative `datasets_configuration` string keeps the library rule — relative to the pipeline YAML's directory, like the `datasets:` key. Pass an absolute path to avoid ambiguity.
