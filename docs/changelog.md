# Changelog

## Unreleased (0.4.0)

0.4.0 gives every output one identity: derivative names are used verbatim on disk, output
folders depend only on the configuration and each file's path below the new `source_root`,
and a unit counts as done only when its `.done` manifest lists every file it wrote. File
discovery is sorted (indices change once) and dataframe column names are reversible. These
changes are breaking: 0.3.x configurations are refused until they declare `naming_version: 1`.
`neurodags migrate` converts a 0.3.x configuration and its existing outputs in one previewed
pass; start with the upgrade checklist in {doc}`user_guide/migrating_to_0_4`.

### Breaking

- **`naming_version: 1` is required** at the top level of every pipeline configuration. A
  configuration without it, or with another value, is refused before anything runs by every
  entry point: all CLI subcommands (one `neurodags: error: ...` line, exit code `2`, no
  traceback), the TUI (`neurodags tui`, `neurodags-tui`) and the Python API (`run_pipeline`,
  `iterate_derivative_pipeline`, `build_derivative_dataframe`,
  `get_datasets_and_mount_point_from_pipeline_configuration`, ...; raises
  `neurodags.loaders.ConfigVersionError`). The message points to `neurodags migrate`. The
  generated `slurm-script` templates load the pipeline with
  `neurodags.loaders.load_pipeline_configuration` and pass `base_dir=` to `run_pipeline`.
- **`source_root` is required for every dataset** (active or skipped; string or mount map).
  For an active dataset it must resolve for the active mount point and be an existing
  directory, and every discovered file must lie strictly below it (a file outside it is a
  `DatasetConfigError` naming the file). `neurodags validate` suggests a value from the fixed
  part of `file_pattern` (never a filesystem root).
- **Relative paths inside a datasets YAML are resolved against that YAML's directory**
  (`source_root`, `file_pattern`, `exclude_pattern`, `derivatives_path`; in 0.3.x: the current
  working directory). In an inline `datasets:` mapping they resolve against the pipeline YAML's
  directory, and in a dict passed to the Python API against the new `base_dir=` argument
  (relative paths without it are an error). `~`/`$VAR` expansion is unchanged and happens
  first. Discovered file paths (and the dataframe `file_path` column) are therefore absolute
  when the pattern was relative.
- **The YAML-relative `-d/--datasets` fallback is removed** (deprecated in 0.3.5): `-d` is
  resolved against the current working directory only; a missing file is a clean error
  (exit `2`). `neurodags.cli.resolve_cli_datasets_path` takes only the `-d` value now.
- **Derivative names are used verbatim on disk** (#49): outputs are
  `<source>@<name><suffix>` with the name exactly as written in `DerivativeDefinitions`; the
  0.3.x CamelCase conversion is gone (`band_power` writes `@band_power`, was `@BandPower`).
  Names follow a stricter grammar, `^[A-Za-z](?:[A-Za-z0-9_-]*[A-Za-z0-9])?$`, must be YAML
  strings (quote `Yes`, `On`, `1_000`, ... keys), unique ignoring case, not `SourceFile`, and
  `X` and `X-<digits>` cannot both be defined. Every `DerivativeDefinitions` key,
  `DerivativeList` entry, reference and `--derivative` value is checked when the pipeline is
  loaded (so by every CLI command, the TUI and the Python API), all problems together, before
  anything is written (`neurodags.naming.NamingError`; exit code `2` on the CLI, including
  `validate`). This replaces the 0.3.5 on-disk collision check and its `validate` "files named
  @X" listing; `validate` prints name warnings (long names, repeated key-value keys) instead.
- **Artifact suffixes are validated before the first writer runs** (#49): every
  `NodeResult.artifacts` key must match `^(?:\.[A-Za-z0-9][A-Za-z0-9_-]*)+$` (`.nc`,
  `.report.html`, `.EO_baseline.fif`), must not end in `.done`/`.error`/`.skip`, and two keys
  must not differ only in case. An invalid key fails the unit with an `.error` marker and
  nothing is written.
- **Source file names are kept verbatim in output names; the last `@` separates source and
  derivative.** 0.3.x wrote every `@` of a source basename as `&`, so a chained output was
  `sub-1.fif&CleanedEEG.fif@PowerSpectrum.nc`; it is now
  `sub-1.fif@CleanedEEG.fif@PowerSpectrum.nc`. `a@b.vhdr` and `a&b.vhdr` in one folder are
  distinct units (no longer refused as a collision). Parse output names at the last `@`
  (`neurodags.naming.parse_output_basename`, `rsplit("@", 1)`); scripts that split on the
  first `@` get the wrong source for chained names. A broad chained pattern such as
  `*@CleanedEEG*` now also matches downstream outputs in the same tree; anchor it on the
  artifact suffix (`*@CleanedEEG.fif`). `neurodags migrate` renames 0.3.x `&` outputs; until
  then `run`, `dry-run`, `status` and `dataframe` stop with an error pointing to it when
  the output tree holds a file named like an output of one of the pipeline's derivatives whose
  source part contains `&` and that no `.done` claims (0.3.x never wrote `.done`) and no real
  source of that spelling owns (one scan of each
  output tree per command). `status` no longer counts a downstream
  unit's files (`x@Clean.fif@Power.nc`) as unlisted or incomplete files of the upstream unit
  `x@Clean`.
- **Output folders are computed from `source_root`** (#50):
  `<derivatives_path>[/<derivatives_subdir>]/<dirname(rel)>/<basename>@<name><suffix>`
  where `rel` is the source path below `source_root` (next to the source without
  `derivatives_path`). Locations no longer depend on the common root of the discovered files
  (`find_unique_root`), so adding files, narrowing `file_pattern`, limits, indices or input
  lists never move outputs; a single-file dataset keeps its folders below `source_root`. Trees
  written by 0.3.x may sit at other folders (pilot-era roots; `neurodags migrate` moves them).
  Two sources mapping to the same outputs (datasets sharing a tree, names differing only in
  case) are refused before running (`DatasetConfigError`). `InputEnumeration.common_roots` is replaced
  by `source_roots`, and the third value returned by `get_all_files_across_datasets` /
  `get_all_files_from_pipeline_configuration` is now the resolved `source_root` per dataset.
  `neurodags.utils.find_unique_root` remains as a utility but no longer decides the layout.
- **Two active datasets resolving to the same `derivatives_path` must use different
  `derivatives_subdir` values** (#50; unset counts as one value); otherwise `validate` and
  every run, dry run, `status`, `count-inputs` and `dataframe` refuse the configuration
  (`neurodags.utils.DatasetConfigError`).
- **Discovery excludes output trees** (#50): a dataset never discovers files below its own
  `derivatives_path[/derivatives_subdir]` (when that lies inside its `source_root`), nor in
  `code/` or `.neurodags/` under any `derivatives_path` of the configuration, so chained
  stages do not take their own outputs as inputs.
- **`.done` manifests define "done"** (#27): a unit (source file × derivative) is cached only
  when `<prefix>.done` (JSON: `manifest_version`, `naming_version`, `derivative`, `source`,
  `artifacts`, `companions`, `written`, `origin`) parses, matches the derivative and naming
  version, and every file it lists exists. This replaces the "any `prefix.*` file exists"
  check in execution, dry runs, `status`, dataframe collection and reuse steps; files matching
  the prefix but not listed are ignored. **0.3.x outputs have no manifest: they are
  `incomplete` and recomputed** unless converted with `neurodags migrate`. On (re)compute the
  old `.done` is deleted before any artifact is written and the new one is published last,
  atomically (temp file + `os.replace`); `.done`, `.error` and `.skip` are mutually exclusive.
  The dry-run plan's `derivative_output` step gains `state`, `artifacts`, `origin`,
  `manifest_problem` and `unlisted`.
- **`neurodags status` reports `incomplete` units** (#27; outputs without a valid `.done`): a
  new table column, an `incomplete` count per derivative and in `grand_total`,
  `incomplete_indices` and `unlisted_files` in `--format json`, `unlisted` per file with
  `--include-files`. Incomplete units make `status` exit `1` and are included in
  `--format indices` (and in `--list-missing`). The TUI status table gains the same column.
- **Exact artifact references; no fallback** (#49): `derivative: Name.suffix` selects the
  artifact whose suffix key equals `.suffix` exactly (string equality), whether `Name` is done
  on disk (from its `.done` manifest), computed in the same run, or kept in memory
  (`save: false`). The 0.3.x `endswith` matching on file names is gone: `Clean.fif` no longer
  matches a `.report.html` or `.EO_baseline.fif` artifact, so write `Splitter.EO_baseline.fif`
  in full. A missing suffix no longer falls back to another file or to the whole `NodeResult`:
  it fails the consuming unit with the new `neurodags.dag.ArtifactSelectionError` (a
  `LookupError`; `.error` marker when the consumer saves), with the same message on every
  path, listing the available suffixes. The dry-run plan reports it as `selection_problem` on
  the reuse step.
- **A bare `derivative: Name` is allowed only when `Name` has exactly one artifact**; it then
  behaves like `Name<suffix>`. With several (or no) artifacts it raises
  `ArtifactSelectionError` (in 0.3.x: the first file in sorted order, or the whole
  `NodeResult`).
- **Dataframe columns use a new, reversible coordinate encoding** (#49,
  `neurodags.naming.format_column`): dimension names and coordinate values are
  percent-encoded (UTF-8, uppercase hex) instead of the lossy 0.3.x character swaps, and dims
  are always sorted by encoded name. Wide column names and long-format `derivative` values
  change wherever a coordinate contained `-`, `_`, `@`, `,`, a space or a non-ASCII character,
  or a dimension name contained anything outside `[A-Za-z0-9]`: `band-8~12` → `band-8-12`,
  `freq-~1.5` → `freq--1.5`, `cond-a|b` → `cond-a%5Fb`, `ref-a$b` → `ref-a%40b`, `x-1.5`
  (from `1,5`) → `x-1%2C5`, `channel-EEG 001` → `channel-EEG%20001`, `freq_band-alpha` →
  `freq%5Fband-alpha`, and an empty coordinate `dim-NA` → `dim-`. Columns split back with
  `neurodags.naming.parse_column`; the guide shows a pandas recipe (the dataframe stays "easy
  to split, not pre-split": no extra columns). The `sort_flattened_dims` option of
  `neurodags.dag.collect_derivative_for_dataframe` is removed.
- **Error columns are `<name>.error`** (were `<name>__error`); long-format error rows have
  `derivative = "<name>.error"`.
- **Dataframe column collisions are errors** (were warnings in 0.3.5, with the earlier value
  overwritten): two coordinates of one artifact (non-unique index, or mixed types such as `1`
  and `"1"`), two artifacts or two derivatives producing the same column raise
  `neurodags.naming.ColumnCollisionError` naming the column and both origins, even with
  `raise_on_error=False` (never recorded as a per-file `.error`). `neurodags dataframe` prints
  one `neurodags: error: ...` line and exits `2`.
- **Sorted discovery: file indices change once** (#41). Per dataset, discovered files are
  deduplicated and sorted by their path relative to `source_root` (`/` separators, code-point
  order: `sub-10` sorts before `sub-2`; zero-pad labels for a natural order), instead of the
  filesystem's glob order. `max_files_per_dataset` keeps the first N after sorting (a limited
  run is a prefix of the full run) and is applied before indexing, so with a limit
  `--only-index` refers to the limited list (in 0.3.x `run` indexed the unlimited list).
  Datasets are concatenated in YAML order. Regenerate any stored index lists and SLURM scripts
  after upgrading.
- **`--only-index` with an unknown or negative index is an error** (#41; was a warning, the
  index being ignored): a clean `neurodags: error: ...` line and exit code `2` on the CLI;
  `neurodags.iterators.IndexSelectionError` (a `ValueError`) from `run_pipeline`,
  `iterate_derivative_pipeline` and `build_derivative_dataframe`, raised before anything is
  written.
- **`count-inputs` counts the enumeration directly** (#41; number of index entries, as in
  0.3.6) instead of dry-running a derivative; `--derivative` is accepted but no longer
  affects it.
- **`slurm-script` templates use a frozen input list** (#41): the submit instructions run
  `neurodags inputs pipeline.yml -o inputs.tsv` and size the array with
  `count-inputs --inputs inputs.tsv`, and the workers pass `inputs=os.path.abspath("inputs.tsv")`.
- **Removed APIs**: `neurodags.utils.snake_to_camel` (kept privately for `neurodags migrate`),
  `utils.derivative_file_names`, `utils.check_derivative_names` (use
  `neurodags.naming.check_derivative_names`), `utils.analyze_bids_filename`,
  `dag._is_step_cached`, `dag._artifact_candidates_for` and the orchestrator helpers
  `_sanitize_reference_base` / `_resolve_reference_base` (use `neurodags.outputs`).
  **Deprecated**: `utils.replace_bids_suffix` warns (`DeprecationWarning`) and is removed in
  0.5.

### Added

- **`neurodags migrate pipeline.yml [-d DATASETS] [--mount NAME] [--apply] [--report plan.json]
  [--map OLD=NEW] [--legacy-cwd DIR] [--no-synthesize] [--source-root DATASET[:MOUNT]=PATH]
  [--upstream-map OLDTOKEN=NEWNAME]`** (#49, #50, #27): upgrades a 0.3.x project in one
  pass, previewing by default. The configuration gets `naming_version: 1`, a `source_root`
  per dataset and mount (suggested from `file_pattern`; a filesystem root is never
  suggested) and relative dataset paths rewritten from the 0.3.x working directory to the
  YAML's directory; files are edited in place (comments kept) and written last, with
  `.bak-0.3` backups. Existing outputs are found by scanning each `derivatives_path` (0.3.x
  CamelCase tokens, pilot-era flat trees, the pre-0.3.3 `<derivatives_path>@Deriv.*` form,
  `&` source names, matched by their 0.3.x spelling) and moved with `os.replace` to the
  verbatim 0.4.0 names and folders (`x.vhdr&A.fif@B.nc` becomes `x.vhdr@A.fif@B.nc`);
  `.error`/`.skip` markers move with their unit; orphans are left in place, but chained
  outputs whose upstream source is not discovered are refused (*unresolved*) rather than
  recomputed, also when the dataset discovers no source at all. A separate datasets YAML is
  written before the pipeline YAML, so a run interrupted between the two completes when
  re-run. MNE split
  recordings are re-saved with MNE under the new name and verified before the old parts are
  deleted. Units whose artifact set is known get a `.done` with `"origin": "migrate"`.
  Ambiguous mappings, occupied targets, two old copies of one unit with different content
  (typically a pilot-era flat copy next to the full-run copy), unwritable folders,
  cross-filesystem moves and invalid 0.4.0 names (rename with `--map`) are refused before
  anything changes. `--apply` keeps a journal
  (`<derivatives_path>/.neurodags/migrate-journal.json`), so an interrupted run resumes, and
  re-running an applied plan is a no-op. `--report` writes the plan as JSON with a dataframe
  column map (`name__error` → `name.error`; best-effort coordinate re-encoding,
  `neurodags.migrate.legacy_column_to_new`). See the new guide
  {doc}`user_guide/migrating_to_0_4`.
- **`done (migrated)` in `neurodags status`** (#27): the table gets a `(migrated)` column when
  some done units were certified by `neurodags migrate`; `--format json` adds `migrated`
  counts per derivative and in `grand_total`, and a per-file `label` (`done (migrated)`).
- **`neurodags inputs pipeline.yml [-d DATASETS] [-o inputs.tsv]`** (#41): writes the frozen
  input list, `# neurodags-inputs naming_version=1` followed by `index<TAB>dataset<TAB>rel`
  lines (`rel`: path below `source_root`, POSIX separators), so one list works on every mount.
- **`--inputs FILE`** (#41) on `run`, `dry-run`, `status`, `count-inputs` and `dataframe` (path
  relative to the working directory), and `inputs=` on `run_pipeline`,
  `iterate_derivative_pipeline` and `build_derivative_dataframe`: the list replaces discovery
  for selection and indexing. Files on disk not in the list are counted and reported (logged;
  printed by `status`) but not processed; listed files that no longer exist are reported as
  `missing-source` at their original index (`status` shows them; `run` skips them with a
  per-file error record; `--only-index` on such an index fails). The list is validated
  (header/version, known non-skipped datasets, no duplicate indices, indices contiguous from
  0); combining it with `max_files_per_dataset` is an error. `count-inputs --inputs` prints
  the number of entries. `neurodags status --format json` gains `inputs`,
  `missing_source_indices` and `not_in_list` keys, and a `missing-source` count per
  derivative and in `grand_total`.
- **`derivatives_subdir`** dataset field (#50, optional): a relative subfolder of
  `derivatives_path` for the dataset (validated: no `..`, not absolute, not `code/` or
  `.neurodags/`). Outputs of the dataset are written below it. The `code/` snapshot stays at
  `<derivatives_path>/code/`.
- **`base_dir=`** argument on `run_pipeline`, `iterate_derivative_pipeline`,
  `build_derivative_dataframe`, `get_all_files_from_pipeline_configuration` and
  `get_datasets_and_mount_point_from_pipeline_configuration`: the directory relative paths of
  in-memory dataset mappings (and, for a dict pipeline, a relative `datasets:` reference and
  `new_definitions`) resolve against.
- **`neurodags validate` dataset report**: per active dataset, the active mount point and the
  resolved `source_root`, `file_pattern`, `exclude_pattern`, `derivatives_path`,
  `derivatives_subdir` and resulting output folder; every dataset problem (with the
  `source_root` suggestion) as an `error:` line on stderr, exit code `1`. It also warns when
  a dataset's `source_root` contains its own outputs without an `exclude_pattern` covering
  them (`neurodags.iterators.output_tree_warnings`).
- **Writer contract** (#27): `Artifact.writer(path)` may return the list of files it wrote (or
  `None` = exactly `path`); the `.done` manifest is built from these reports and every
  reported file must exist. New `neurodags.writers.save_mne(obj, path, **kwargs)` returns
  every part of an MNE split recording (`X.fif`, `X-1.fif`, ...), used by the built-in
  preprocessing nodes.
- **`neurodags.naming`** (#49, #50): pure functions that define the 0.4.0 naming rules in one
  place: derivative-name grammar and validation (`check_derivative_names`: string keys,
  reserved `SourceFile`, case-insensitive uniqueness, the `X`/`X-1` split-file rule), the
  optional key-value convention (`Fit_alpha-0p5_w-6`, `alpha-0p5_w-6_Fit`) with
  `parse_entities` and `encode_number`/`decode_number` (`0p5`, `m1`), artifact suffix
  validation, output basenames (`<source>@<name><suffix>`), reference parsing, and reversible
  dataframe columns (`format_column`, `parse_column`, `decode_dim`, `decode_value`,
  `ColumnCollisionError`).
- **`neurodags.outputs`** (#27, #50): the single layout and completion module —
  `output_reference_base`, `entry_reference_base`, `output_prefix`,
  `check_output_collisions`, `writer_paths`, `build_manifest`, `write_manifest`,
  `read_manifest`, `manifest_artifacts`, `unit_state` (`done`, `error`, `skipped`,
  `incomplete`, `missing`), `publish_marker`, `remove_markers`.
- **Enumeration API** in `neurodags.iterators` (#41): `discover_inputs`, `load_inputs`,
  `enumerate_inputs`, `enumerate_pipeline_inputs`, `read_inputs_file`, `write_inputs_file`,
  `format_inputs`, and the `InputEnumeration` / `InputEntry` records (each entry exposes
  `index`, `dataset`, `rel`, `path` and `missing`). `run_pipeline` and
  `iterate_derivative_pipeline` accept a precomputed `enumeration=`.
  `get_all_files_across_datasets` / `get_all_files_from_pipeline_configuration` keep their
  return shapes on top of it (see Breaking for their third value).
- `neurodags.loaders.load_pipeline_configuration` (loads a pipeline and checks its version
  and names) and `neurodags.loaders.check_pipeline_names(config)` (validates every name and
  returns the warnings).
- Helpers in `neurodags.utils`: `resolve_config_path`, `resolve_dataset_field`,
  `suggest_source_root`, `relative_to_source_root` (lexical check that a file lies strictly
  below `source_root`), `check_derivatives_subdir`, `effective_derivatives_base` and
  `inspect_dataset_paths`.

### Changed

- **One enumeration per invocation** (#41): `run_pipeline` discovers the inputs once and shares
  them with every derivative (it rediscovered per derivative), and `build_derivative_dataframe`
  scans once (it scanned twice).
- Documentation: new guides {doc}`user_guide/naming` (rewritten for 0.4.0: names, layout,
  manifests, unit states, references, column names) and {doc}`user_guide/migrating_to_0_4`
  (upgrade checklist, reading the preview, the pilot-copy refusal, refusals table); every
  guide, the README, the comparison page, the gallery examples and `example_pipelines/` use
  the 0.4.0 configuration surface. Every complete pipeline and datasets YAML shown in the
  docs is validated by the test suite (`tests/test_docs_yaml_snippets.py`).

### Fixed

- **Outputs could be recomputed or written to a new folder when the set of discovered files
  changed** (#50): 0.3.x placed outputs below the common root of the discovered files, so a
  pilot run on one subject and a later full run wrote the same unit to two folders, and the
  full run recomputed it. Locations now depend only on `source_root` and the file's own path.
- **A partially written unit counted as done** (#27): any file with the output prefix made a
  unit cached, so an interrupted multi-artifact write or a missing MNE split part was never
  recomputed. A unit is now done only when its `.done` manifest lists every file.
- **File indices depended on the filesystem's glob order** (#41), so the same `--only-index`
  could select different files on two machines or after files were added; discovery is now
  sorted, and `neurodags inputs` freezes the list.
- **Lossy dataframe column names** (#49): distinct coordinates such as `a-b` and `a~b`, or
  `a_b` and `a|b`, produced the same column; the new encoding is reversible.
- `datasets_psycho.yml`'s `fir` `file_pattern` was a bare directory and is now a `**/*.vhdr`
  glob.
- `neurodags migrate` (pre-release fixes): `--apply` never replaces an existing file (moves
  and staged, verified MNE re-saves are published with `os.link`) and holds
  `<derivatives_path>/.neurodags/migrate.lock`, never taken over automatically, also for
  configuration-only runs; `--map A=B` onto another definition is refused; compressed split
  recordings (`.fif.gz`, `.fif-N.gz`) stay one artifact and re-saves use MNE's default split
  size; interrupted split moves resume after any number of interruptions, and FIF recordings
  are loaded in full before their `.done`; a unit's `.done` lists all its artifacts; a split
  rewrite into a new folder works; re-running after keeping an identical duplicate (also a
  split recording whose kept copy was re-saved) is a no-op.

## 0.3.6 — 2026-10-12

### Fixed

- **`count-inputs` undercounted when a file is listed under more than one dataset entry**: it
  printed the number of unique file paths, while `--only-index` numbers (dataset entry, file)
  pairs. With the same recordings in two dataset entries (e.g. one entry per condition), a
  SLURM array sized with `N=$(neurodags count-inputs ...)` was too small and the last indices
  never ran. It now prints the number of indices. Pipelines where each file appears in only
  one dataset entry get the same number as before.

### Documentation

- The CLI reference said "paths inside the YAML are relative to the YAML", which is only true
  for the `datasets:` key in `pipeline.yml`. Paths inside a datasets YAML (`file_pattern`,
  `derivatives_path`, `exclude_pattern`) are resolved against the current working directory;
  the page now lists each case.

## 0.3.5 — 2026-10-12

### Added

- **`neurodags status --exit-zero`** (#45): exit `0` whenever the report was produced, even
  if derivatives are missing or errored, so `status` can run as a pure report — e.g. a short
  job chained with `--dependency=afterok`, where exit `1` would cancel everything after it.
  Works with every `--format` (`table`, `json`, `indices`). Real failures (missing or invalid
  config, unset environment variable, unreadable `-d` file) still exit non-zero. The default
  exit codes are unchanged (`1` when anything is missing or errored). The HPC and inspection
  guides now recommend `--exit-zero` instead of `|| true`, which also hid config errors.

- **"File Naming Conventions" docs page** (`user_guide/naming.md`, #42), linked from the
  pipeline YAML guide, Core Concepts and Dataframe Assembly: the `@` / `&` output-name rules,
  the derivative-name conversion and the rejected names, artifact suffixes (must start with
  `.`), `.error` / `.skip` markers, how `derivative: Name.ext` references are parsed (YAML name,
  split at the first `.`), reserved arg strings (`id.N`, `id.N.field`, `$var`, `eval%`), the
  output layout (common root, single-file datasets, no `derivatives_path`, `code/` snapshot) and
  dataframe column names (YAML name, flattened `@dim-value_...` suffixes and their escaping, how
  to read a column back), with a worked example tree and a Do / Don't table. Core Concepts'
  output example now shows the source extension (`sub-1.fif@CleanedEEG.fif`), and Dataframe
  Assembly's example columns now match the real format (`BandPower.nc@freqbands-alpha`). The
  page notes that the collision check is per pipeline config (two pipelines sharing a
  `derivatives_path` are not checked against each other), that `snake_to_camel` lowercases the
  rest of each `_`-separated part (`ICA_cleaned` → `@IcaCleaned`), and that a naming redesign is
  planned for 0.4.0. The `snake_to_camel` docstring explains why the conversion exists.
- **Comparison page wording** (#42): NeuroDAGs is described as input-naming-convention agnostic
  (input names kept, no BIDS validation, optional `bids_parse` node) instead of
  "BIDS-Preserving", and the output strategy as appending `@<Derivative><suffix>` to the source
  name, with the layout mirroring the input below the discovered files' common root, instead of
  an unconditional "perfect mirror".

### Changed

- **`-d/--datasets` is resolved relative to the current working directory** (#43): every
  subcommand that takes `-d` (`validate`, `run`, `dry-run`, `dataframe`, `count-inputs`,
  `status`, `slurm-script`, `tui`, and the legacy `neurodags-tui`) now resolves the path
  against the directory the command is run from, after expanding `~` and `$VARS` like the
  `datasets:` key. Previously it was joined onto the pipeline YAML's directory, so
  `neurodags run pipelines/x/pipeline.yml -d pipelines/x/alt.yml` looked for
  `pipelines/x/pipelines/x/alt.yml`. The rule is now: CLI paths are relative to where you run
  the command; paths inside the YAML (the `datasets:` key) stay relative to the YAML.
  **Behaviour change:** a relative `-d` path that exists relative to the working directory is
  now used even if a different file with that name exists next to the pipeline YAML. For
  backward compatibility, a path that only exists relative to the pipeline YAML is still used,
  with a one-line `neurodags: warning: ...` on stderr showing the cwd-relative path to pass
  instead (a `FutureWarning` from `neurodags.cli.resolve_cli_datasets_path`); this YAML-relative
  fallback will be removed in a future release. The Python API is unchanged: a relative
  `datasets_configuration` string passed to `run_pipeline` / `iterate_derivative_pipeline` /
  `build_derivative_dataframe` is still resolved relative to the pipeline YAML (now
  documented in their docstrings).

- **On-disk derivative names are shown** (#42): `neurodags validate` lists each derivative whose
  on-disk name differs from its YAML name (e.g. `fit_w_l06 -> files named @FitWL06`). The naming
  rule itself is unchanged: it keeps the derivative name a single token without `_` (a
  separator in many naming conventions, including BIDS), and existing caches keep working.

### Fixed

- **YAML merge keys (`<<: *anchor`) failed to load** (#44): the duplicate-key-checking loader
  (`UniqueKeySafeLoader`) overrode `construct_mapping` without flattening merge keys, so a
  pipeline such as `args: {<<: *fitargs, space: a}` failed with `could not determine a
  constructor for the tag 'tag:yaml.org,2002:merge'`, and shared node arguments had to be
  repeated in every derivative. The loader now flattens merges as PyYAML's `SafeConstructor`
  does, with standard semantics: `<<: *a` and `<<: [*a, *b]` are supported, keys written in
  the mapping override merged ones, and with a list of merges the earlier one wins (results
  match `yaml.safe_load`). Duplicate keys are still rejected with their line/column, both in
  a mapping that uses a merge and inside an anchored mapping; only keys written explicitly in
  the same mapping count, so overriding a merged key is not a duplicate.
- `register_derivatives_from_yaml` and the built-in `derivatives/pipeline.yml` now load
  through `load_configuration` too, so every pipeline YAML path rejects duplicate keys the
  same way.

- **Missing `-d/--datasets` file crashed with a traceback** (#43): when the file exists
  neither relative to the working directory nor relative to the pipeline YAML, or references
  an unset environment variable, the CLI now prints a one-line error naming the paths it tried
  and exits with code `2`, instead of an `OSError` traceback from the YAML loader.

- **Colliding or layout-breaking derivative names silently shared cached outputs** (#42):
  derivative names are converted with `snake_to_camel` to build file names
  (`<source>@<OnDiskName><artifact suffix>`), so two different derivatives such as `band_power`
  and `BandPower` both wrote and read `@BandPower` files; because caching is existence-based,
  the second one silently reused the first one's output. Every name in `DerivativeDefinitions`
  is now validated up front (not only the run set: dependencies are pulled in and all
  derivatives share the derivatives directory), raising a `ValueError` that lists each problem:
  - two names with the same on-disk name — the message names both YAML names and the shared
    `@<OnDiskName>` — or on-disk names equal up to case (the same file on case-insensitive
    filesystems, e.g. `Bandpower` / `band_power`);
  - names containing `.` (the artifact-suffix separator: `Fit.v2`'s files `@Fit.v2.*` matched
    `Fit`'s cache lookup `@Fit.*`, so `Fit` was never computed and its dataframe columns showed
    `Fit.v2`'s data), `@` (breaks the one-`@`-per-file-name rule), `/` or `\` (path separators:
    the write failed), or the glob characters `*`, `?`, `[`, `]` (the cache lookup globbed them:
    `F*` treated `@Fit…` files as its own);
  - the reserved name `SourceFile` and the empty name (`derivative: SourceFile` always means the
    raw input, so such a derivative could never be referenced).

  The check runs in `neurodags validate` (exit code 1) and before anything is written in
  `run_pipeline` / `iterate_derivative_pipeline` / `build_derivative_dataframe` (and hence
  `neurodags run`, `dry-run`, `status`, `count-inputs`, `dataframe`), so it is never swallowed
  into per-file error records. New helpers `neurodags.utils.derivative_file_names` and
  `check_derivative_names`.
  **Behaviour change:** pipelines with such names now fail fast; they were producing wrong
  results (or failing per file) before. Rename the derivative(s). On-disk naming of all valid
  names is unchanged.

- **Cached outputs were not found when a path contained `[`, `]`, `*` or `?`**: the cache
  lookup globbed `<output prefix>.*` with the literal output path, so glob metacharacters in
  the source file name or in **any** parent directory name (e.g. `study [2024]/sub-01.vhdr`)
  were interpreted as a pattern. Affected users: anyone whose source files, dataset
  directories or `derivatives_path` contain `[`, `]`, `*` or `?`. For them every derivative
  was silently recomputed on every run, `neurodags status` / dry-run reported those outputs
  as missing, `derivative: Name.ext` reuse steps recomputed the upstream derivative, and
  `for_dataframe` derivatives were skipped during dataframe collection. A bracketed name
  could also produce a wrong cache hit (`a[1].fif` matched the outputs of `a1.fif`). The
  literal prefix is now escaped (only the trailing `.*` is a pattern) and only regular files
  count as artifacts. Output paths are unchanged, so after upgrading existing outputs are
  found again — no recompute needed. `file_pattern` / `exclude_pattern` are still user glob
  patterns: a literal bracketed directory there must be escaped by the user (e.g.
  `study [[]2024]/**/*.vhdr`, see `glob.escape`).
- **Dataframe column collisions are now reported instead of silent**: when flattening an
  xarray artifact, coordinate values are encoded into column names (`-`→`~`, `_`→`|`,
  `@`→`$`, `,`→`.`), so distinct coordinates such as `a-b` and `a~b` map to the same column
  and the earlier value was overwritten without notice. A warning
  (`Dataframe column collision while flattening artifact ...`) now names the derivative,
  artifact suffix, generated column suffix, and both original coordinate sets. Likewise, when
  two derivatives produce the same column in a dataframe row, a warning
  (`Dataframe column collision: derivative produces columns already present in the row ...`)
  names the columns, the derivative being merged and the previous owner. Output (column
  names, last value wins) is unchanged; the encoding itself will be revisited in 0.4.0.
- **Mermaid diagrams with non-identifier derivative names** (e.g. `psd-welch`):
  `pipeline_to_mermaid` / `neurodags dag` used raw derivative names as Mermaid node ids, which
  produced invalid diagrams for names containing `-` or other punctuation. Such names now get
  a sanitized id (non-word characters → `_`, de-duplicated with a numeric suffix if needed)
  and keep the real name as the node label. Diagrams whose names are already valid ids are
  unchanged.

### Documentation

- The pipeline YAML guide has a new section on sharing node arguments across derivatives
  with anchors and merge keys, kept under a top-level helper key such as `_defaults`
  (unknown top-level keys are ignored by the loader and `neurodags validate`).

## 0.3.4 — 2026-10-12

### Added

- **`basic_preprocessing(average_psd=...)`**: with `extra_artifacts: true`, `average_psd: true`
  plots each report PSD averaged across channels (one trace) instead of one trace per channel,
  which makes filter roll-offs easy to see. Default `false` keeps the previous figures.
- **Quickstart preprocessing template** in `example_pipelines/`
  (`pipeline_quickstart_preprocessing.yml` + `datasets_quickstart_preprocessing.yml`):
  band-pass 0.1–75 Hz, 2 s epochs, resample to 256 Hz, plus an MNE Report — built-in nodes
  only, paths from `$RAW_DATA` / `$OUTPUT_DIR`. Documented in the Quickstart guide and run
  end-to-end in the test suite.

## 0.3.3 — 2026-10-12

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
