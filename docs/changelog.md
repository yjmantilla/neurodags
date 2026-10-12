# Changelog

## Unreleased

### Fixed

- **Relative paths broke with `n_jobs>1` after an in-process `chdir`** (#35): loky reuses
  worker processes, and a reused worker keeps the cwd it was started with. After an
  `os.chdir` in the parent (notebooks, tests, scripts calling `run_pipeline` more than once),
  workers resolved relative source files, `derivatives_path` and `common_root` against the
  stale cwd: inputs were not found, outputs (or "cached" hits) landed in the previous
  directory, while `n_jobs=1` worked. The parent now makes the source file path, the dataset
  path fields (`file_pattern`, `derivatives_path`, `exclude_pattern`) and the common root
  absolute before dispatch, for `run_pipeline` / `iterate_derivative_pipeline` and
  `build_derivative_dataframe`. The derivative layout is unchanged (the path relative to the
  common root is the same), and the reported `file_path` (logs, dataframe column, dry-run
  `file_path` column) keeps the discovered spelling.
  **Behaviour change:** nodes and derivatives now receive the absolute source path (e.g. the
  `SourceFile` path and dry-run `file` entries), and outputs of datasets without
  `derivatives_path` are addressed by absolute path; the files land in the same place.
- **Parallel dataframe workers lacked `new_definitions` custom nodes** (#36):
  `build_derivative_dataframe` loaded custom nodes only in the parent, so with `n_jobs>1` a
  fresh worker computing an uncached or `save: False` derivative that used a custom node failed
  with `Unknown node`. The absolute custom definition paths are now sent to each worker and
  loaded there, as `run_pipeline` already did.

- **Inspection commands no longer create output directories** (#33): `neurodags status`,
  `dry-run`, `count-inputs` (and `run_pipeline(..., dry_run=True)`) created
  `derivatives_path` and its per-file subdirectories as a side effect of computing output
  paths. Computing the reference base is now pure; directories are created only when
  something is about to be written — by real (non-dry) runs before each file is processed,
  and by `run_derivative` just before it computes. `build_derivative_dataframe` therefore
  only creates directories when it has to compute a derivative, not when it reads cached
  outputs. (`orchestrators._resolve_reference_base`, `dag.run_derivative`)

- **Single-file datasets write inside `derivatives_path`** (#34): for a dataset matching
  exactly one file, `find_unique_root(..., mode="maximal")` returned the file itself, so
  outputs were written as siblings of the derivatives directory
  (`<parent>/derivatives@Meta.json`). The root is now always a strict ancestor of every
  input (the file's parent directory for a single file), so outputs land at
  `<derivatives_path>/<file name>@Meta.json` — the same layout as files sharing a
  directory in a multi-file dataset. A reference base that would still resolve to `.` is
  also mapped to the file name. (`utils.find_unique_root`, `orchestrators._resolve_reference_base`)

  **Behaviour change (single-file datasets only):** outputs previously written as
  `<derivatives_path>@<Derivative>.*` (next to `derivatives_path`, named after it) are not
  found any more and will be recomputed; move them to
  `<derivatives_path>/<source file name>@<Derivative>.*` to reuse them. Output paths for
  datasets with two or more files are unchanged, so existing caches keep working.

## 0.3.2 — 2026-10-12

### Added

- **Parser nodes + `id.N.field` field access** (#8): a parser node takes a file path and
  returns a flat dict of properties; downstream args can now pull a single field with
  `id.N.field` (e.g. `subject: id.1.subject`) instead of each node re-parsing the path.
  New built-in `bids_parse` extracts BIDS entities via `mne_bids.get_entities_from_fname`.
  The arg resolver supports `id.N.field` alongside the existing whole-result `id.N`.
  **Field keys must be plain identifiers** — `.` is the reference separator, so
  dotted/dashed/nested paths (`id.1.a.b`, `id.1.some-key`) are rejected with a clear error;
  parser output must stay flat with identifier keys (values are unconstrained). Field access
  also works when the parser derivative is served from cache (`derivative: Name.json`): the
  cached `.json` artifact is loaded back into a mapping; field access on any other cached
  artifact raises. Mermaid diagrams draw edges for `id.N.field` references. See the new
  *Parser Nodes* user guide. (`nodes.parsers.bids_parse`, `dag._resolve_refs`)

  **Behaviour change:** a literal string arg that starts with `id.<digits>.` (e.g.
  `"id.1.subject"` or `"id.2.a.b"`) is no longer passed through verbatim — it now resolves
  as a field reference or raises an error.

- **`neurodags status` reports file indices for `--only-index` resubmission** (#29): the
  global file index — the one `--only-index` and the SLURM array templates consume — was
  dropped from the status rows, so there was no direct way from "which files are
  missing/errored" to "resubmit exactly those".
  - `--list-missing` / `--list-errors` now print `[idx N] path`, sorted by index within each
    derivative (the `error file:` lines are kept).
    **Behaviour change:** scripts that scraped paths from these text lists see the new
    `[idx N]` prefix; use `--format json` for machine consumption.
  - `--format json` always adds `missing_indices` and `errored_indices` (sorted,
    de-duplicated ints). All existing keys are unchanged.
  - New `--include-files` (requires `--format json`) adds a flat `files` list with one
    `{index, dataset, file_path, derivative, status, error_path}` record per
    (source file, derivative) pair, sorted by `(index, derivative)`.
  - New `--format indices` prints only the space-separated, sorted missing/errored indices
    (narrowed by `--list-missing` / `--list-errors`), ready for `--only-index` or
    `sbatch --array`. The exit code keeps its meaning (`1` when anything is missing or
    errored), so guard `$(...)` with `|| true` under `set -e`.
  - See the new "Resubmitting missing/errored files" section in the HPC guide (retry workers
    must not use `skip_errors=True`; flat-array task IDs are not file indices).

### Fixed

- **`status --format json` with no input files emitted `No files found.`** (invalid JSON);
  it now emits a valid document with zero counts, empty index lists and `"complete": true`.

## 0.3.1 — 2026-10-11

### Fixed

- **Slow CLI startup** (#30): `import neurodags.cli` took ~7.5 s because node discovery
  imported `antropy` (which JIT-compiles numba kernels at import), `neurokit2`, `matplotlib`
  and `fooof`, and `datasets.py` imported `mne.export`. These are now imported the first time
  a node that needs them runs (and `mne.export` only inside `save_dummy_vhdr`);
  `nodes/spectral.py` also uses postponed annotations so its `mne.io.BaseRaw` signatures no
  longer load `mne.io`/`mne.viz`/matplotlib at import. CLI startup drops from ~7.5 s to
  ~1.3 s; the saving also applies to every parallel worker and SLURM task that does not run
  those nodes. `matplotlib.use("Agg")` is no longer a side effect of importing `neurodags`;
  it now happens when a neurokit node first executes.
  **Behaviour change:** if `antropy`, `neurokit2` or `fooof` is missing, the corresponding
  nodes are now still registered and raise an actionable `ImportError` when called (recorded
  as a per-file error), instead of being silently skipped at discovery and failing with
  `KeyError: Unknown node`. (`nodes/antropy.py`, `nodes/neurokit.py`, `nodes/spectral.py`,
  `datasets.py`)

- **`~` and `$VARS` in dataset paths are now expanded** (#28): `file_pattern`,
  `derivatives_path`, `exclude_pattern` and the `datasets:` reference in `pipeline.yml` expand
  a leading `~` and `$NAME` / `${NAME}` (`%NAME%` on Windows) for the active mount point.
  Previously they were used verbatim, so `derivatives_path: $SCRATCH/out` silently wrote
  derivatives (and the `code/` snapshot) into a literal `./$SCRATCH` directory. The `datasets:`
  reference is now expanded *before* being joined to the pipeline directory, and
  `derivatives_path` is resolved in the parent process before dispatching parallel workers.
  (`utils.expand_path`, `utils.get_path`, `utils.resolve_dataset_paths`)

  **New error:** a referenced environment variable that is unset or empty now raises
  `ValueError` naming the field and variable. `run_pipeline` / `neurodags run` / `dry-run` /
  `status`, `build_derivative_dataframe` / `neurodags dataframe` and `neurodags validate` check
  every active (non-skipped) dataset before anything is written, so the error is not
  swallowed into per-file error records. Inactive mount-point entries are not checked.
  `neurodags validate` now also prints the resolved paths for each dataset.

- **Docs site now shows the released version**: the Docs workflow checked out without tags, so
  hatch-vcs fell back to `0.1.dev1+g<hash>` on the published documentation. The workflow now
  fetches full history and tags (`fetch-depth: 0`), and `docs/conf.py` displays the nearest
  release tag (falling back to the installed dist version), so the site shows e.g.
  `neurodags 0.3.0 (<commit>)`. (`.github/workflows/docs.yml`, `docs/conf.py`)

## 0.3.0 — 2026-07-13

### Added

- **`neurodags status` command**: new inspection subcommand reporting, per derivative, how
  many source files are *done* / *skipped* / *missing* / *errored* — without opening a CSV.
  Supports `--derivative` selection, `--list-missing` / `--list-errors` to print the offending
  paths, `--format json` for machine-readable output (`config`, `n_files`, per-derivative
  counts, `grand_total`, and a `complete` boolean), and `--n-jobs` for parallel status
  collection. The *skipped* column reports derivatives with a `.skip` marker (see
  `SkipDerivative`) distinctly from *missing*. (`cli._cmd_status`, `cli._status_classify`)

- **Parallel processing (`--n-jobs`)**: `run`, `dry-run`, `dataframe`, and `status` process
  files in parallel via joblib. `--n-jobs N` (or `-1` for all cores; `1`/`None` stays serial),
  with optional `--joblib-backend` / `--joblib-prefer`. Per-file parallelism uses separate
  processes (loky) to avoid HDF5/NetCDF4 thread-safety issues. Also exposed as `n_jobs=` on
  `iterate_derivative_pipeline` / `run_pipeline` / `build_derivative_dataframe`.

- **Global logging options `--log-level` / `--log-file`**: set console verbosity
  (`DEBUG`/`INFO`/`WARNING`/`ERROR`, defaulting to `$LOG_LEVEL` or `INFO`) and optionally tee
  all log events to a JSONL file (one JSON object per line, loadable with
  `pandas.read_json(path, lines=True)`). (`cli`, `loggers.configure_logging`)

- **`SkipDerivative` exception**: nodes can now raise `SkipDerivative` to signal that a
  source file is intentionally not processable by a given derivative — distinct from an
  unexpected error. neurodags catches this, writes a `.skip` marker file alongside where
  the artifact would have been saved, and propagates the skip to all parent derivatives
  that depend on it (each writing their own `.skip` marker). Skipped derivatives are not
  retried on subsequent runs unless the `.skip` file is deleted or `overwrite: true` is
  set. **Motivation:** in multi-condition studies, some subjects may not have undergone
  every condition; without `SkipDerivative`, their missing conditions showed as *missing*
  in `neurodags status` — indistinguishable from derivatives that simply had not run yet,
  which made pipeline completion state ambiguous. (`definitions.SkipDerivative`,
  `dag.run_derivative`, `neurodags.SkipDerivative`)

- **`neurodags dag --layout`**: new flag for HTML DAG output selecting the layout engine.
  `elk` (default) uses orthogonal routing via ELK — requires CDN access. `dagre` uses
  right-angle step edges with no CDN dependency — suitable for offline environments.
  Also available as `layout=` in the Python API (`pipeline_to_html`, `derivative_to_html`,
  `save_mermaid_html`).

- **Dataset-level variables (`vars:`)**: dataset entries in `datasets.yml` can now
  declare a `vars:` block of arbitrary key-value pairs.  Any pipeline node arg whose
  string value matches `$identifier` is substituted with the corresponding value from
  the active dataset entry's `vars` at runtime, after `id.N` reference resolution.
  Only whole-string values are substituted — embedded `$` in paths or other strings
  is left untouched.  Variables may be any YAML type (string, int, float, bool,
  list).  Referencing an undefined variable raises `KeyError` with the list of
  available vars.  Primary use case: encoding a condition name (or any
  dataset-specific parameter) in the dataset entry so that activating a different
  entry changes both `derivatives_path` and pipeline behaviour in one step, with no
  pipeline YAML edits required.  (`definitions.DatasetConfig.vars`,
  `dag._resolve_vars`, `dag._prep_kwargs`)

- **In-memory multi-artifact selection**: when a node returns a `NodeResult` with
  multiple artifacts (e.g. a splitter that produces one artifact per condition),
  downstream derivatives can now select a specific artifact using the existing
  dot-extension syntax — `derivative: SplitterName.condA.fif` — even when the
  splitter has not yet been written to disk.  Previously this selection only worked
  for on-disk (cached) artifacts; the in-memory path passed the full `NodeResult`
  and relied on the `_unwrap_for_arg` heuristic, which returned the first matching
  artifact regardless of the requested suffix.  The fix applies the same suffix
  filter to the in-memory `NodeResult` that was already applied to on-disk
  candidates, making both paths consistent.  A warning is logged when the requested
  suffix is absent from the splitter's artifacts.  (`dag.run_derivative`)

- **Config snapshot on `neurodags run`**: before executing any derivatives, the pipeline
  YAML, `new_definitions` file(s), and datasets YAML are copied to
  `derivatives_path/code/`.  A `neurodags_env.json` file is also written with the
  installed neurodags version, git commit of the source repo (when installed from a
  checkout), and a UTC timestamp.  Skipped on dry runs; failures are warnings, never
  errors.  (`orchestrators._snapshot_pipeline_config`)

- **TUI Status tab**: the Textual TUI gained a Status tab that displays per-derivative
  pipeline status and errors, mirroring the `neurodags status` command. (`tui`)

- **`neurodags view` renders neurokit2 figures**: the interactive `.nc`/`.fif` explorer now
  renders neurokit2 `figure_png_hex` / `figure_rgba` artifact variables as images.
  (`visualization`)

### Changed

- **`neurodags validate` reports the effective run set instead of a misleading
  `derivatives_enabled`** (#17): the old output printed `derivatives_enabled` equal to the
  full `DerivativeList`, which read as "what a run computes" but ignored `--derivative`
  selection and dependency resolution. `validate` now takes an optional `--derivative`
  (repeatable, defaults to `DerivativeList`) and prints `run_set` (the selection),
  `computed_with_dependencies` (the selection plus the intermediates auto-computed as its
  dependency closure — what a run actually produces), which of those are `for_dataframe`
  outputs, and `not computed by this run` for the remaining defined derivatives.
  (`cli._cmd_validate`, `cli._dependency_closure`)

- **`neurodags status` exit code**: exits `1` when any derivatives are missing or errored
  (not only errored). Enables use in CI and shell dependency chains:
  `neurodags status pipeline.yml || sbatch resubmit.sh`.

- **DAG HTML visualization uses ELK layout by default**: Mermaid diagrams now use the ELK
  layout engine (orthogonal edge routing, crossing minimisation) instead of dagre with bezier
  curves. Significantly cleaner for dense pipelines. Use `--layout dagre` for offline use.
  The raw Mermaid text output (`neurodags dag` without `--html`) is unchanged.

- **`neurodags count` renamed to `neurodags count-inputs`**: clarifies that the command
  counts source (input) files the pipeline will process, not output files or derivative
  instances. One input file may produce multiple output files depending on the derivatives.
  All generated SLURM templates, documentation, and tests updated accordingly.

### Fixed

- **Inspection subcommands no longer pollute stdout with logs** (#15): `status`, `validate`,
  and `dag` write their deliverable (table / `--format json` / Mermaid) to **stdout**, but
  framework logs — including the per-file INFO from `status`'s internal dry-run and the
  import-time built-in derivative registration — were written to stdout too, so the result
  was buried and `--format json` was not pipeable (even with stderr redirected, because the
  chatter was on stdout). structlog output now routes to **stderr** (TTY/JSON detection uses
  stderr), and a lightweight import-time bootstrap installs a stderr-routed, level-filtered
  default *before* the built-in registration runs, so `neurodags status … --format json
  2>/dev/null` yields a single clean JSON document. (`loggers.configure_logging`,
  `loggers._bootstrap_quiet_default`)

- **No more `fooof` DeprecationWarning on every CLI invocation** (#16): importing the node
  registry pulled in `fooof`, whose `__init__` calls `warnings.simplefilter('always')` and
  then warns about its rename to `specparam` — printing the notice to stderr on `status`,
  `validate`, `dag`, `run`, etc. The import in `nodes/spectral.py` is now wrapped in
  `warnings.catch_warnings(record=True)`, which swallows the notice and restores the prior
  warning-filter state (fooof's `'always'` reset no longer leaks process-wide). neurodags
  still depends on fooof; migration to `specparam` is tracked separately.
  (`nodes.spectral`)

- **Parallel workers can now resolve uncached inter-derivative references** (#18): YAML
  `DerivativeDefinitions` were registered only in the main process, so with `--n-jobs > 1`
  a fresh (loky) worker that had to *compute* a referenced sub-derivative — rather than
  read it from disk cache — raised `ValueError: Unknown derivative '<Name>.nc'`. The bug
  was masked whenever the intermediate was already cached (the disk-cache path needs no
  registry) and disappeared under `--n-jobs 1`. The full `DerivativeDefinitions` map is
  now threaded into each `_FileJob` and re-registered at the top of `_process_file_job`
  (mirroring how `custom_node_paths` re-registers nodes), and `_collect_dataframe_file`
  re-registers the map it already receives — so both the `run` and `dataframe` paths
  resolve fresh nested derivatives in parallel. (`orchestrators._FileJob`,
  `orchestrators._process_file_job`, `orchestrators._collect_dataframe_file`)

- **`aggregate_across_dimension` no longer silently drops non-finite values** (#19): a
  reduction such as `mean`/`std` runs with xarray's default `skipna=True`, so NaN values
  along the aggregated dimension were dropped with no record — a per-channel mean over 2
  surviving epochs looked identical downstream to one over 13, and since missingness is
  often not-at-random (e.g. failed FOOOF fits concentrating in one condition) this was a
  data-quality/leakage hazard. The node now takes `on_dropped` (`"warn"` default →
  `log.warning` with the dropped count; `"raise"` to fail fast; `"ignore"` for the old
  behaviour) and an opt-in `emit_counts` flag that attaches `n_used`/`n_dropped`
  coordinates to the aggregated array so per-value reliability is queryable. Fires only
  when a reduction actually skips NaN (float data, `skipna` in effect); pipelines with no
  non-finite values are unchanged. (`operations.aggregate_across_dimension`)

- **Split-FIF continuations no longer scanned as separate source files**: MNE splits
  recordings larger than ~2 GB across multiple `.fif` files, of which only the first
  (*entry*) file is an independent recording — the continuations are stitched back in
  transparently by `mne.io.read_raw_fif(entry)`. The file scanner used a bare
  `glob.glob`, so every continuation was returned as its own source file and the pipeline
  ran derivatives on partial data, emitting duplicate/garbage rows. The scanner now drops
  continuations while keeping the entry, for both conventions: BIDS `_split-01_` (kept) vs
  `_split-02_`+ (dropped, zero-padding aware), and plain-mne `name.fif` (kept) vs
  `name-1.fif`/`name-2.fif`+ (dropped). Detection is filename-based and cheap (no file
  reads); the plain-mne `-N` rule only fires when the entry file is also present in the
  scan, so legitimately named files such as `sub-01_..._run-2_meg.fif` are never dropped.
  Controlled by the new `drop_split_continuations` flag (default `True`) — a per-dataset
  field on `DatasetConfig` (opt out one dataset) and a parameter on
  `get_files_from_pattern` / `get_all_files_across_datasets` /
  `get_all_files_from_pipeline_configuration` (global switch). The number of dropped
  continuations is logged. (`iterators.find_split_continuations`,
  `iterators.get_files_from_pattern`, `definitions.DatasetConfig.drop_split_continuations`)

- **Sub-derivative cache respected when parent has `overwrite: True`**: previously,
  a derivative with `overwrite: True` forced `cached_here = False` for all of its
  sub-derivative inputs, causing them to re-execute even when they had `overwrite: False`
  and valid cached files on disk.  The cache check now uses the *child* derivative's own
  `overwrite` flag rather than the parent's, so only the derivative that explicitly sets
  `overwrite: True` is recomputed.  (`dag.run_derivative`)

- **`{"cached": [...]}` dict no longer leaks into node arguments**: when a sub-derivative
  early-returns its internal `{"cached": [path, ...]}` sentinel (e.g. because it hit its
  own cache), the parent previously stored that dict raw in `store[sid]` and passed it
  on to downstream node functions as if it were a real value — causing `AttributeError`
  at runtime.  The parent now resolves the cached dict to the matching path string before
  storing, so downstream nodes always receive a proper path or `NodeResult`.
  (`dag.run_derivative`)

- **Custom nodes available during `save=False` dataframe assembly**: `build_derivative_dataframe`
  now loads the pipeline's `new_definitions` node modules, so `for_dataframe` derivatives that
  use custom nodes resolve during dataframe collection instead of raising "unknown node".
  (`orchestrators`)

- **Custom node modules registered in `sys.modules` before exec**: prevents the double
  registration that occurred when a split node-definition module was imported both by path
  and by name. (`nodes.loader`)

## 0.1.0

- Initial release of the template.
