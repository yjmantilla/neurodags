# File Naming Conventions

NeuroDAGs has no database: an output's **file name is its identity**. Caching is
existence-based (a derivative is "done" for a file when a matching output file exists), chained
derivatives are found by name, and dataframe columns are built from names. This page lists every
naming rule and restriction in one place. Most of them are enforced: `neurodags validate`
reports problems in your derivative names before anything runs.

NeuroDAGs is **agnostic to your file-naming convention**: it keeps your source file names (BIDS or
not) and only appends its own markers — `@` (source/derivative separator), the derivative name
and the artifact suffix — plus `&`, which replaces `@` when a derivative is itself used as a
source. Whether outputs look BIDS-like depends on how your inputs are named.

```{note}
The naming rules on this page describe the current behaviour. A redesign is planned for 0.4.0
([#49](https://github.com/yjmantilla/neurodags/issues/49));
until then they are kept stable so existing caches stay valid.
```

## Rules at a glance

| Do | Don't |
|----|-------|
| Name derivatives in CamelCase (`BandPower`) or with `-` (`psd-welch`); both are used on disk unchanged. | Use `.`, `@`, `/`, `\`, `*`, `?`, `[`, `]` in derivative names, or name one `SourceFile` — rejected (see [Derivative names](#naming-derivative-names)). |
| If you use `_` (`band_power`), look for files under the CamelCase name (`@BandPower`). | Define two derivatives whose on-disk names match, even only up to case (`band_power` / `BandPower` / `Bandpower`) — rejected. |
| Reference derivatives by their YAML name: `derivative: band_power.nc`, `DerivativeList`, `--derivative`. | Reference the on-disk name (`derivative: BandPower.nc` for a YAML name `band_power`) — it is not an alias. |
| Glob outputs by on-disk name: `**/*@BandPower*`. | Glob by YAML name (`*@band_power*`) — matches nothing. |
| Start custom node artifact keys with `.` (`.nc`, `.report.html`, `.EO.fif`). | Use artifact keys without a leading `.` (`-epo.fif`) — the cache lookup (`@Name.*`) never finds them, so they recompute every run. |
| Keep raw source file names free of `@`, `&` and glob characters (`*`, `?`, `[`, `]`). | Have a source named with `[`/`]` (never seen as cached), or both `a@b.vhdr` and `a&b.vhdr` in one folder (same outputs). |
| Split dataframe column names with `neurodags.naming.parse_column` (see [Dataframe column names](#naming-dataframe-columns)). | Split columns by hand on `_` or `-` — dimension names and values are percent-encoded (`freq_band` → `freq%5Fband`), and values may contain `-` (`8-12`). |
| Treat `<derivatives_path>/code/` as reserved for the config snapshot. | Store your own files in `<derivatives_path>/code/` — each `neurodags run` rewrites its snapshot files there. |

(naming-output-files)=

## Output file names

Every saved artifact is written to

```text
<reference_base>@<OnDiskName><artifact suffix>
```

- **`<reference_base>`** identifies the source file (see [Output layout](#naming-output-layout)).
  It keeps the source's own extension: `sub-01_task-rest_eeg.vhdr@...`.
- **`@`** separates the source from the derivative. There is exactly one `@` per output file
  name, so `name.split("@", 1)` (or `rsplit`) always gives `(source, derivative part)`.
- **`<OnDiskName>`** is the derivative name converted for disk (see below).
- **`<artifact suffix>`** is the key of each artifact the last node returns, appended verbatim:
  `.fif`, `.nc`, `.json`, `.report.html`, `.EO_baseline.fif` (multi-dot suffixes are fine).
  It must start with `.`: a derivative counts as cached when any file matches
  `<reference_base>@<OnDiskName>.*`.

A derivative with several artifacts writes several files sharing the same prefix; the presence of
**any** of them counts as cached (so a run interrupted mid-write is not redone; use
`overwrite: true` or delete the files).

### Marker files

Two marker files share the output prefix and are never counted as cached outputs:

| File | Written when | Effect |
|------|--------------|--------|
| `<reference_base>@<OnDiskName>.error` | a node of the derivative raised | Contains the error. The derivative is retried on the next run unless `--skip-errors` / `skip_errors=True`; removed after a later successful save. `neurodags status` reports the file as *errored*. |
| `<reference_base>@<OnDiskName>.skip` | a node raised `SkipDerivative` (not applicable to this file) | The derivative (and derivatives depending on it) is skipped for this file until the marker is deleted or `overwrite: true`. |

`save: false` derivatives write neither outputs nor markers.

(naming-derivative-names)=

## Derivative names

The on-disk name is derived from the name in `DerivativeDefinitions`
(`neurodags.utils.snake_to_camel`):

- A name **without `_`** is used as is: `BandPower` → `@BandPower`, `psd-welch` → `@psd-welch`
  (`-` is left unchanged).
- A name **with `_`** is split on `_`, each part is title-cased (Python `str.title()`) and the
  parts are joined: `fit_w_l06` → `@FitWL06`, `band_power` → `@BandPower`, `Fit_A` → `@FitA`,
  `EO_Features` → `@EoFeatures` (note the lower-cased `o`).
- `str.title()` **lowercases the rest of each `_`-separated part**, so capitals inside a part are
  lost: `Ep_DetrendedFluctuation` → `@EpDetrendedfluctuation`, `ICA_cleaned` → `@IcaCleaned`.
  Check `neurodags validate` (or `snake_to_camel(name)`) for the exact name to search for.

The conversion keeps the derivative name a single token without `_`, which many file-naming
conventions (including BIDS) use as a separator. Names with `_` are accepted; just look for their
files under the converted name. The conversion is applied consistently for writing and reading,
so existing caches keep working.

**Everything except the file names uses the YAML name**: `DerivativeList`, `--derivative`,
reuse steps (`derivative: fit_w_l06.nc`), dataframe columns and log messages. The on-disk name is
not an alias: `derivative: FitWL06.nc` is an unknown derivative (it only matches by accident when
the files already exist).

`neurodags validate` lists every derivative whose on-disk name differs from its YAML name:

```text
on-disk names differing from YAML names (1):
  fit_w_l06 -> files named @FitWL06
```

### Rejected names

`neurodags validate`, `run`, `dry-run`, `status`, `count-inputs`, `dataframe` and the Python
entry points (`run_pipeline`, `iterate_derivative_pipeline`, `build_derivative_dataframe`) check
**every** name in `DerivativeDefinitions` (not only those selected to run) and fail with a
`ValueError` before anything is written when:

| Problem | Why |
|---------|-----|
| Two names map to the same on-disk name, e.g. `band_power` and `BandPower` (both `@BandPower`) | They would read and overwrite each other's cached files, silently returning the wrong data. |
| Two on-disk names differ only in case, e.g. `Bandpower` and `band_power` | Same file on case-insensitive filesystems (macOS, Windows). |
| The name contains `.` | `.` starts the artifact suffix: `derivative: Fit.v2.nc` would be read as derivative `Fit`, and `Fit`'s cache lookup (`@Fit.*`) would match `Fit.v2`'s files, so `Fit` would never be computed. |
| The name contains `@` | Breaks the one-`@`-per-file-name rule. |
| The name contains `/` or `\` | Path separator: outputs would go to a non-existent subdirectory. |
| The name contains `*`, `?`, `[` or `]` | Glob characters in the cache lookup: `F*` would treat `@Fit…` files as its own. |
| The name is `SourceFile` or empty | `derivative: SourceFile` always means the raw input file, so such a derivative could never be referenced. |

```{important}
The collision check is **per pipeline configuration**. Two *different* pipeline files writing to
the same `derivatives_path` are not checked against each other: if one defines `band_power` and the
other `BandPower`, both use `@BandPower` files and silently share cached outputs. Use distinct
names (or distinct `derivatives_path`s) across pipelines that share an output directory.
```

Other names (with `_`, `-`, spaces, non-ASCII, ...) are accepted. `&` is accepted too but best
avoided: it is the replacement for `@` in chained source names, so it makes those file names
harder to read back.

(naming-references)=

## Derivative references (`derivative: Name.ext`)

A reuse step value is split at its **first** `.`:

| Reference | Derivative | Artifact selected |
|-----------|-----------|-------------------|
| `SourceFile` | the raw input file | — |
| `CleanPrep` | `CleanPrep` | all artifacts (which one a node gets follows the [`id.N` heuristic](pipeline_yaml.md)) |
| `CleanPrep.fif` | `CleanPrep` | the `.fif` artifact |
| `Splitter.EO_baseline.fif` | `Splitter` | the `.EO_baseline.fif` artifact |

The part before the first `.` is the YAML name; the rest selects the artifact. When the derivative
is read from disk, the rest is matched against the **end** of the file names, so `CleanPrep.fif`
would also match a `.EO_baseline.fif` artifact; the first match in sorted order is used (with a
warning if several match). When it was just computed in memory, the artifact key must equal
`.<rest>` exactly. Use the full artifact suffix to get the same artifact either way.

### Other reserved strings in node `args`

| Form | Meaning |
|------|---------|
| `id.N` | output of step `N` of the same derivative |
| `id.N.field` | one field of a mapping returned by step `N` (`field` must be a plain identifier) |
| `$name` (whole string) | dataset variable from the active dataset's `vars:` block |
| `eval%<expression>` | Python expression (with `np`, `math`) for callable arguments of the factory nodes, e.g. `pure_function` |

Any other string is passed to the node unchanged. Strings starting with `id.<digits>.` that are not
a valid `id.N.field` raise an error instead of being passed through.

(naming-output-layout)=

## Output layout

- **With `derivatives_path`**: NeuroDAGs takes the **common root** of all files matched by the
  dataset (the deepest directory containing all of them) and mirrors each file's path relative to
  it under `derivatives_path`:
  `<derivatives_path>/<path relative to common root>@<OnDiskName><suffix>`. For a dataset matching
  a single file, the common root is that file's directory, so outputs land at
  `<derivatives_path>/<file name>@...`. The common root depends on which files match: changing
  `file_pattern` (or `exclude_pattern`) so the matched set has a different common root moves where
  outputs are expected (making this layout stable is tracked in
  [#50](https://github.com/yjmantilla/neurodags/issues/50)).
- **Without `derivatives_path`**: outputs are written next to the source file
  (`<source path>@<OnDiskName><suffix>`).
- **`@` in the source basename becomes `&`**: when the source file name already contains `@`
  (typically a NeuroDAGs output used as the input of another pipeline), every `@` in the
  **basename** (not in directory names) is replaced by `&`, keeping a single `@` per output name.
  This is not reversible: `a@b.vhdr` and `a&b.vhdr` in the same folder would share outputs.
- **`code/`**: `neurodags run` copies the pipeline YAML, `new_definitions` files and datasets YAML,
  plus `neurodags_env.json`, into `<derivatives_path>/code/` (see
  [Usage](usage.md)).

(naming-dataframe-columns)=

## Dataframe column names

`build_derivative_dataframe` / `neurodags dataframe` name columns after the **YAML name**, with
{func}`neurodags.naming.format_column`:

```text
<yaml name><artifact suffix>                                  # one column per artifact
<yaml name><artifact suffix>@<dim>-<value>_<dim>-<value>...   # one column per DataArray element
<yaml name>.error                                             # the derivative failed for this file
```

A `.nc` artifact holding an N-dimensional `xarray.DataArray` is flattened into one column per
element: one `dim-value` pair per dimension, joined by `_`, **sorted by encoded dimension name**
(0-d arrays get no `@` part). Dimension names keep only `A-Z`, `a-z`, `0-9` literal; values also
keep `.`, `+`, `~`, `-`. Every other character is percent-encoded as UTF-8 with uppercase hex:

| Coordinate | Column |
|------------|--------|
| `band="alpha"`, `channel="Fp1"` | `BandPower.nc@band-alpha_channel-Fp1` |
| `band="8-12"` | `BandPower.nc@band-8-12` |
| `freq=-1.5` | `Psd.nc@freq--1.5` |
| `channel="EEG 001"` | `Psd.nc@channel-EEG%20001` |
| `cond="a_b"` | `Erp.nc@cond-a%5Fb` |
| dimension `freq_band` | `Psd.nc@freq%5Fband-alpha` |

The encoding is reversible, so any dimension name and coordinate value is safe. Metadata columns
(`index`, `dataset`, `file_path`) never contain `.`, while generated columns always do. Two
coordinates, artifacts or derivatives that would produce the same column (a non-unique
coordinate index, or mixed types such as `1` and `"1"`) stop the build with
`neurodags.naming.ColumnCollisionError`, whatever `raise_on_error` is.

**Reading a column back.** {func}`neurodags.naming.parse_column` returns
`(name, suffix, coords, is_error)`, or `None` for a metadata column. By hand, for
`band_power.nc@freq%5Fband-alpha%5Flow_spaces-Fp1-F3`:

1. Split at the first `@`: `band_power.nc` and `freq%5Fband-alpha%5Flow_spaces-Fp1-F3`.
2. Split the head at its first `.`: name `band_power`, suffix `.nc` (suffix `.error` marks the
   error column).
3. Split the coordinates on `_` (never literal inside a dimension or value):
   `freq%5Fband-alpha%5Flow`, `spaces-Fp1-F3`.
4. Split each pair at its first `-` (never literal inside a dimension) and percent-decode:
   `freq_band = alpha_low`, `spaces = Fp1-F3`.

In long format (`output_format="long"`), the same string is in the `derivative` column. The
dataframe is *easy to split, not pre-split*: see
{ref}`Splitting column names <dataframe-splitting-columns>` for a pandas recipe that expands
coordinates and name parameters into columns.

## Worked example

`pipeline.yml` defines `clean_prep` (on disk `CleanPrep`, a `.fif` plus a `.report.html`) and
`band_power` (on disk `BandPower`, a `.nc` with dimensions `spaces` × `freqbands`); the dataset
matches `rawdata/**/*.vhdr` and sets `derivatives_path: derivatives`:

```text
rawdata/                                         # common root of the matched files
├── sub-01/eeg/sub-01_task-rest_eeg.vhdr
└── sub-02/eeg/sub-02_task-rest_eeg.vhdr

derivatives/
├── code/                                        # config snapshot (neurodags run)
│   ├── pipeline.yml
│   ├── datasets.yml
│   └── neurodags_env.json
├── sub-01/eeg/
│   ├── sub-01_task-rest_eeg.vhdr@CleanPrep.fif
│   ├── sub-01_task-rest_eeg.vhdr@CleanPrep.report.html
│   └── sub-01_task-rest_eeg.vhdr@BandPower.nc
└── sub-02/eeg/
    └── sub-02_task-rest_eeg.vhdr@CleanPrep.error # clean_prep failed; band_power not written
```

A second pipeline uses the cleaned files as its sources (`file_pattern:
derivatives/**/*@CleanPrep.fif`, `derivatives_path: derivatives2`) with a derivative `Entropy`.
Only sub-01 matches, so the common root is `derivatives/sub-01/eeg` and the `@` of the source
becomes `&`:

```text
derivatives2/
└── sub-01_task-rest_eeg.vhdr&CleanPrep.fif@Entropy.json
```

Dataframe columns from the first pipeline (wide format, `band_power` with `for_dataframe: true`):

```text
index  dataset  file_path                                 band_power.nc@freqbands-alpha_spaces-Fp1-F3  band_power.nc@freqbands-alpha_spaces-O1  ...
0      demo     rawdata/sub-01/eeg/sub-01_task-rest_eeg.vhdr  0.32                                         0.29                                     ...
```

To find all `band_power` outputs on disk: `find derivatives -name '*@BandPower*'`.
