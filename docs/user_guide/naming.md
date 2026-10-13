# File Naming and Output Layout

NeuroDAGs has no database: an output's **location and name are its identity**, and a small
`.done` manifest next to the outputs records that a unit finished. This page lists every naming
rule, where outputs go, and when a unit counts as done. The rules are those of
`naming_version: 1` (neurodags 0.4.0); `neurodags validate` reports problems before anything
runs.

NeuroDAGs is **agnostic to your file-naming convention**: it keeps your source file names (BIDS or
not) and only appends its own markers — `@` (source/derivative separator), the derivative name
and the artifact suffix. Source file names are kept verbatim, including any `@` or `&` they
contain.

```{important}
**Changed in 0.4.0.** Derivative names are used **verbatim** on disk (no CamelCase conversion),
output folders are computed from each dataset's `source_root` (never from the set of discovered
files), and a unit is done only when its `.done` manifest is valid. Outputs written by 0.3.x have
no manifest: `neurodags status` reports them as `incomplete` and the next run recomputes them,
unless you convert the tree with `neurodags migrate` (see {doc}`migrating_to_0_4`).
```

## Rules at a glance

| Do | Don't |
|----|-------|
| Use names matching `^[A-Za-z](?:[A-Za-z0-9_-]*[A-Za-z0-9])?$`: `BandPower`, `band_power`, `psd-welch`, `Fit_alpha-0p5_w-6`. | Use `.`, `@`, `&`, `/`, `\`, glob characters, spaces, `,`, `:`, non-ASCII, a leading digit or `-`, or the name `SourceFile` — rejected (see [Derivative names](#naming-derivative-names)). |
| Quote YAML keys that YAML would read as something else: `"Yes":`, `"1_000":`. | Write `Yes:`, `On:`, `1_000:`, `0x1F:`, `~:` as derivative names — YAML loads them as booleans, numbers or null; rejected with "quote the key". |
| Use the YAML name everywhere: files, `derivative: band_power.nc`, `DerivativeList`, `--derivative`, dataframe columns, `find -name '*@band_power*'`. | Define names that differ only in case (`Clean` / `clean`) or a name plus a split-like tail (`Clean` / `Clean-1`) — rejected. |
| Start custom node artifact keys with `.` (`.nc`, `.report.html`, `.EO_baseline.fif`). | Use keys without a leading `.`, with spaces, ending in `.done`/`.error`/`.skip`/`.lock`, two keys differing only in case, or a key plus the same key with `-<digits>` before its extension (`.EO.fif` / `.EO-1.fif`) — the unit fails with `.error` and nothing is written. |
| Make custom artifact writers return every file they write (see [Writer contract](#naming-writer-contract)). | Write extra files from a writer without reporting them — they are not part of the unit and the manifest ignores them. |
| Set `source_root` to the root folder of the dataset's sources; outputs mirror the tree below it. | Rely on the set of matched files to decide output folders — it never does since 0.4.0. |
| Split dataframe column names with `neurodags.naming.parse_column` (see [Dataframe column names](#naming-dataframe-columns)). | Split columns by hand on `_` or `-` — dimension names and values are percent-encoded. |
| Treat `<derivatives_path>/code/` and `<derivatives_path>/.neurodags/` as reserved. | Store your own files there — `neurodags run` rewrites its snapshot in `code/`, and discovery never returns files from either folder. |

(naming-output-files)=

## Output file names

Every saved artifact of a *unit* (one source file × one derivative) is written to

```text
<reference_base>@<name><artifact suffix>
```

- **`<reference_base>`** identifies the source file (see [Output layout](#naming-output-layout)).
  It keeps the source's own extension: `sub-01_task-rest_eeg.vhdr@...`.
- **`@`** separates the source from the derivative: it is the **last** `@` of the file name.
  The source part is the source basename verbatim and may itself contain `@` (a chained output
  such as `sub-1.fif@CleanedEEG.fif@PowerSpectrum.nc`), but names and suffixes never do, so
  `basename.rsplit("@", 1)` gives `(source part, name + suffix)`, and
  `name, _, s = rest.partition(".")` gives the name and suffix `"." + s`
  ({func}`neurodags.naming.parse_output_basename`). Do not split on the first `@`.
- **`<name>`** is the derivative name exactly as written in `DerivativeDefinitions`.
- **`<artifact suffix>`** is the key of each artifact the last node returns, appended verbatim:
  `.fif`, `.nc`, `.json`, `.report.html`, `.EO_baseline.fif`, `.fif.gz`. It must match
  `^(?:\.[A-Za-z0-9][A-Za-z0-9_-]*)+$`, its last segment must not be `done`, `error`, `skip`
  or `lock` (any case), and two suffixes of one result must not differ only in case. Nor may one
  suffix be another with `-<digits>` inserted before the final extension (`.EO.fif` and
  `.EO-1.fif`, compared ignoring case): MNE writes the split parts of a large
  `<prefix>.EO.fif` as `<prefix>.EO-1.fif`, `<prefix>.EO-2.fif`, ..., which the second artifact
  would overwrite (the artifact-level analogue of the `Clean` / `Clean-1` name rule). Suffixes
  are checked **before the first writer runs**: an invalid key fails the unit with an `.error`
  marker and writes nothing. As a second line of defence, a writer is not run when an earlier
  artifact of the unit already reported its target, and two artifacts reporting the same file
  fail the unit (no `.done`).

(naming-done-manifest)=

### Completion manifest (`.done`) and markers

Three marker files share the unit's prefix; a unit has at most one of them:

| File | Written when | Effect |
|------|--------------|--------|
| `<reference_base>@<name>.done` | every artifact was written | JSON manifest listing the unit's files. The unit is **done** (cached) while it is valid. |
| `<reference_base>@<name>.error` | anything inside the unit failed: a node, a derivative it depends on, preparing a node's arguments, selecting a referenced artifact, or writing an artifact | Contains the error. The unit is retried on the next run unless `--skip-errors` / `skip_errors=True`. |
| `<reference_base>@<name>.skip` | a node raised `SkipDerivative` (not applicable to this file) | The unit (and units depending on it) is skipped until the marker is deleted or `overwrite: true`. |

A fourth file, `<reference_base>@<name>.lock`, exists only while a process (re)computes the unit
(see [Unit lock](#naming-unit-lock)); it is not a completion marker.

The manifest:

```json
{
  "manifest_version": 1,
  "naming_version": 1,
  "derivative": "Fit_alpha-0p5_w-6",
  "source": {"dataset": "study", "rel": "sub-01/eeg/sub-01_task-rest_eeg.vhdr"},
  "artifacts": {".nc": ["sub-01_task-rest_eeg.vhdr@Fit_alpha-0p5_w-6.nc"]},
  "companions": [],
  "written": "2026-10-12T10:00:00Z",
  "origin": "run"
}
```

- `artifacts` maps each suffix to the files it consists of: usually one, every part for an MNE
  split recording (`X.fif`, `X-1.fif`, ...). File names are relative to the manifest's folder, so
  a tree moved or copied to another mount (local ↔ HPC) still verifies.
- `source.rel` is the source path relative to `source_root` with its original spelling
  (including any `@`).

**A unit is done** (a cache hit) when its `.done` parses, its `naming_version` and `derivative`
match, and every file it lists exists. Files that match the prefix but are not listed are
ignored (and reported by `neurodags status`). There is no other cache check: a file named like
an output but without a valid manifest is not trusted.

**Write order.** On (re)compute NeuroDAGs takes the unit's lock, deletes the old `.done` (before
computing anything, including the derivatives it depends on), validates the suffixes, runs
every writer, then publishes the new `.done` atomically (a temporary file in the same folder
renamed with `os.replace`), removes any `.error`/`.skip` and releases the lock. Writing an
`.error` or `.skip` removes the other markers, including `.done`. A crash, a failing writer or a
failing dependency therefore never leaves a `.done` for a half-written or failed recompute; the
next run recomputes it.

`save: false` derivatives write neither outputs nor markers and are never a cache hit, also
when another derivative references them: an old manifest left from when the derivative was
saved is not read.

(naming-unit-lock)=

### Unit lock

Before (re)computing a saved unit, a run creates `<reference_base>@<name>.lock` exclusively
(`O_CREAT | O_EXCL`, which is atomic on local filesystems and NFSv3 or later). The file holds the
pid, host and start time of the process. It is removed once the `.done`, `.error` or `.skip` is
published (or the run is interrupted with an exception, such as Ctrl-C).

A second process reaching a locked unit (a duplicate or retried array task, or two pipelines
sharing a dependency) does not touch it: the unit fails for that process with
`neurodags.outputs.UnitLockedError` ("locked by another process"), logged as an error like any
failed unit (with `raise_on_error`, the run stops with a non-zero exit). No `.error` is written,
because the unit belongs to the holder. A parent derivative whose dependency is locked fails the
same way, again without an `.error`. Reading a done unit (a cache hit) never needs the lock.

Locks are **never broken automatically**. A process killed without cleanup (`kill -9`, a
scheduler timeout or preemption, a node crash) leaves its `.lock` behind, and every later run
refuses the unit until the file is deleted. `neurodags status` reports such units as `locked`
and prints their lock files; delete them once no job is running for those units, for example:

```bash
find <derivatives_path> -name '*.lock' -delete   # only when no neurodags job is running
```

(naming-unit-states)=

### Unit states

`neurodags status`, the dry-run plan (`state` field of the `derivative_output` step) and the TUI
report each unit (one source file × one derivative) as:

| State | Meaning | Next `run` |
|-------|---------|------------|
| `done` | valid `.done` manifest (see above) | skipped (cached) |
| `done (migrated)` | `done`, but the manifest was written by `neurodags migrate` (`"origin": "migrate"`) from 0.3.x outputs; counted as `done` (the table adds a `(migrated)` column, JSON a `migrated` count and per-file `label`) | skipped (cached) |
| `skipped` | `.skip` marker (the derivative does not apply to this file) | skipped |
| `locked` | `.lock` file without a valid `.done`: a job is writing the unit, or one was killed while writing it (see [Unit lock](#naming-unit-lock)). Not a column of the `status` table: the table is followed by the list of lock files; JSON has a `locked` count, `locked_indices` and per-file `lock_path`. `status` exits 1 while any unit is locked | refused until the lock file is gone |
| `errored` | `.error` marker (state `error` in the dry-run plan) | retried, unless `--skip-errors` |
| `incomplete` | output files (or an invalid `.done`) without a valid manifest: an interrupted write, a deleted part, or 0.3.x outputs not yet migrated | recomputed |
| `missing` | nothing on disk for the unit | computed |

With a frozen input list (`--inputs`), a listed source file that no longer exists is reported
as `missing-source` at its index; it is a state of the *input*, not of a unit, and is never
recomputed. See {doc}`inspection` for the `status` output and exit codes.

(naming-writer-contract)=

### Writer contract

`Artifact.writer(path)` receives the artifact path (`<prefix><suffix>`) and returns the list of
files it wrote, or `None` meaning exactly `path`. Every reported file must exist afterwards
(otherwise the unit fails with `.error`), and must be in the same folder as `path` or below it.
The manifest is built from these reports, never from a directory listing.

MNE splits recordings larger than `split_size` (2 GB by default) into `X.fif`, `X-1.fif`,
`X-2.fif`, ...; the first part names the next, so the parts belong together. The built-in
preprocessing nodes save MNE objects with {func}`neurodags.writers.save_mne`, which returns the
list of parts MNE reports having written. Use it in your own nodes:

```python
from neurodags.definitions import Artifact
from neurodags.writers import save_mne

Artifact(item=epochs, writer=lambda path: save_mne(epochs, path))
```

(naming-derivative-names)=

## Derivative names

Grammar (`naming_version: 1`):

```text
name = LETTER ( [A-Za-z0-9_-]* [A-Za-z0-9] )?      ^[A-Za-z](?:[A-Za-z0-9_-]*[A-Za-z0-9])?$
```

Names are opaque identifiers used verbatim everywhere: file names, references, `DerivativeList`,
`--derivative`, dataframe columns, `status` and logs. No case style is required.

Every entry point (`neurodags validate`, `run`, `dry-run`, `status`, `count-inputs`,
`dataframe`, `slurm-script`, the TUI and the Python API) checks, when the pipeline is loaded and
before anything is written, every `DerivativeDefinitions` key (not only those selected to run),
every `DerivativeList` entry, every reference (`derivative:` value) and every `--derivative`
value. All problems are reported together (exit code `2` on the command line):

| Problem | Why |
|---------|-----|
| Not a YAML string (`Yes`, `On`, `No`, `1_000`, `0x1F`, `~` unquoted) | YAML loads them as booleans, numbers or null; quote the key. |
| Does not match the grammar | `.` starts the artifact suffix, `@` delimits the source, `&` and path and glob characters break shells and paths, `,`/`:`/spaces break YAML lists, CSV and the TUI filter. |
| `SourceFile` (any case) | `derivative: SourceFile` always means the input file. |
| Two names differ only in case (`Clean` / `clean`) | Same files on case-insensitive filesystems (macOS, Windows). |
| `X` and `X-<digits>` both defined (`Clean` / `Clean-1`) | MNE writes split recordings as `X-1.fif`, so `X-1`'s files would look like parts of `X`. |

`validate` also prints warnings (not errors) for names longer than 64 characters and for names
that follow the key-value convention below but repeat a key (`Fit_w-6_w-7`).

```{important}
These checks are **per pipeline configuration**. Two different pipeline files writing to the
same `derivatives_path` are not checked against each other; give them distinct names or
distinct `derivatives_path`/`derivatives_subdir` values.
```

### Optional key-value convention

A name may encode parameters as `_`-separated `key-value` pairs plus at most one bare word:
`Fit_alpha-0p5_w-6`, `alpha-0p5_w-6_Fit`. Numbers use `p` for the decimal point and `m` for
minus (`0p5`, `m1`). Any valid name is accepted whether or not it follows the convention, and
NeuroDAGs never parses names itself; {func}`neurodags.naming.parse_entities` and
{func}`neurodags.naming.decode_number` split them for your own analysis.

(naming-references)=

## Derivative references (`derivative: Name.suffix`)

A reuse step value is `SourceFile`, a derivative name, or a name followed by an artifact suffix,
split at the **first** `.`. The suffix (everything from that `.` on) must equal one of the
derivative's artifact suffixes **exactly** (string equality, case included):

| Reference | Derivative | Artifact selected |
|-----------|-----------|-------------------|
| `SourceFile` | the raw input file | — |
| `Clean.fif` | `Clean` | the `.fif` artifact, never `.report.html` |
| `Clean.report.html` | `Clean` | the `.report.html` artifact |
| `Splitter.EO_baseline.fif` | `Splitter` | the `.EO_baseline.fif` artifact |
| `Splitter.fif` | `Splitter` | error when `Splitter` has only `.EO_baseline.fif` and `.EC_baseline.fif`: write the suffix in full |
| `Fit` | `Fit` | its only artifact; error when `Fit` has several (or none) |

The same rule applies on every path: when the referenced unit is done (read from its `.done`
manifest), when it is computed in the same run, and when it is kept in memory (`save: false`).
There is no fallback to another artifact or to the whole result. A suffix the derivative did not
produce, or a bare name for a derivative without exactly one artifact, stops the consuming unit
with `neurodags.dag.ArtifactSelectionError` (its `.error` marker records the message). The
message is the same on every path and lists the available suffixes:

```text
Derivative 'Features': reference 'Splitter.fif' selects artifact '.fif', which derivative
'Splitter' did not produce (available suffixes: '.EC_baseline.fif', '.EO_baseline.fif').
References select the artifact whose suffix is exactly the text after the name. Write the
full suffix, e.g. 'Splitter.EC_baseline.fif'.
```

What the node receives:

- from a done unit, the path of the selected artifact's main file (the first file its manifest
  lists, e.g. the first part of a split recording);
- from a unit computed in the same run (saved or `save: false`), a `NodeResult` holding only the
  selected artifact, which the [`id.N` heuristic](pipeline_yaml.md) unwraps to the artifact's
  object.

A bare name `Fit` for a single-artifact derivative behaves exactly like `Fit` followed by its
suffix.

The reference grammar (a valid name; a suffix matching `^(?:\.[A-Za-z0-9][A-Za-z0-9_-]*)+$` whose
last segment is not `done`, `error` or `skip`; `SourceFile` with no suffix) is checked when the
configuration loads, before anything runs. Whether the suffix exists is known only once the
referenced derivative has produced its artifacts, so that check happens per file at run time.

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

For a source file `f` discovered in dataset `D` on the active mount:

```text
rel    = f relative to D's source_root            (lexical, POSIX separators, no "..")
base   = basename(rel)                         (verbatim, including any "@" or "&")
outdir = derivatives_path / [derivatives_subdir] / dirname(rel)    with derivatives_path
outdir = dirname(f)                                                 without (next to the source)
prefix = outdir / (base + "@" + name)
```

- **The location depends only on the configuration and the file's own path**: adding or
  removing files, narrowing `file_pattern`, `exclude_pattern`, `max_files_per_dataset`,
  `--only-index`, a frozen `--inputs` list or a previous run never moves outputs. A dataset with a
  single file keeps its folders below `source_root`.
- **Mount points**: the paths below `source_root` must be the same on every mount; outputs
  computed on one mount are then found (and verified by their manifests) on another after copying
  the tree.
- **`derivatives_subdir`**: datasets sharing a `derivatives_path` must use different
  `derivatives_subdir` values (see [datasets.yml](datasets_yaml.md)), so adding a dataset never
  moves another one's outputs.
- **Source basenames are verbatim**: a source whose name already contains `@` (typically a
  NeuroDAGs output used as the input of another pipeline) keeps it, so its outputs have several
  `@` (`x.vhdr@clean_prep.fif@Entropy.json`); the last one is the separator. `a@b.vhdr` and
  `a&b.vhdr` in one folder are two distinct units. Before 0.4.0 every `@` of the source basename
  was written as `&`; `run`, `dry-run`, `status` and `dataframe` stop with an error pointing to
  `neurodags migrate` when the output tree holds a file named like an output of one of the pipeline's derivatives whose
  source part contains `&` and that no `.done` claims (0.3.x never wrote `.done`) and no real
  source of that spelling owns.
- **Collisions are refused before running**: two sources mapping to the same reference base
  (datasets sharing an output tree, or names differing only in case on a case-insensitive
  filesystem) stop `run`, `dry-run`, `status` and `dataframe` with an error naming both.
- **Chained pipelines**: a dataset whose `source_root` is an upstream `derivatives_path` mirrors
  the upstream tree. Discovery skips files in `code/` or `.neurodags/` under any
  `derivatives_path` of the configuration, and the dataset's own output tree
  (`derivatives_path[/derivatives_subdir]`) when it lies strictly below its `source_root`. It
  does not skip outputs written next to the sources, an output tree equal to or above
  `source_root`, or other pipelines' trees. `validate` warns when a dataset's `source_root`
  contains its own outputs without an `exclude_pattern` covering them. Anchor chained patterns
  on the full artifact suffix (`*@clean_prep.fif`): a broad `*@clean_prep*` or `*@clean_prep.*`
  also matches downstream outputs such as `x.vhdr@clean_prep.fif@Entropy.json` in the same
  tree.
- **`code/`**: `neurodags run` copies the pipeline YAML, `new_definitions` files and datasets YAML,
  plus `neurodags_env.json`, into `<derivatives_path>/code/` (not under `derivatives_subdir`; see
  [Usage](usage.md)). Without `derivatives_path` there is no snapshot.

The layout is computed by one function, {func}`neurodags.outputs.output_reference_base`, used by
every reader and writer (run, dry-run, `status`, dataframe, TUI).

(naming-dataframe-columns)=

## Dataframe column names

`build_derivative_dataframe` / `neurodags dataframe` name columns after the derivative name, with
{func}`neurodags.naming.format_column`:

```text
<name><artifact suffix>                                  # one column per artifact
<name><artifact suffix>@<dim>-<value>_<dim>-<value>...   # one column per DataArray element
<name>.error                                             # the derivative failed for this file
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
`neurodags.naming.ColumnCollisionError`, whatever `raise_on_error` is. Derivatives with
`for_dataframe: true` are read only from done units (valid `.done`).

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

`pipeline.yml` defines `clean_prep` (a `.fif` plus a `.report.html`) and `band_power` (a `.nc`
with dimensions `spaces` × `freqbands`); the dataset sets `source_root: rawdata`,
`file_pattern: rawdata/**/*.vhdr` and `derivatives_path: derivatives`:

```text
rawdata/                                         # source_root
├── sub-01/eeg/sub-01_task-rest_eeg.vhdr
└── sub-02/eeg/sub-02_task-rest_eeg.vhdr

derivatives/
├── code/                                        # config snapshot (neurodags run)
│   ├── pipeline.yml
│   ├── datasets.yml
│   └── neurodags_env.json
├── sub-01/eeg/
│   ├── sub-01_task-rest_eeg.vhdr@clean_prep.fif
│   ├── sub-01_task-rest_eeg.vhdr@clean_prep.report.html
│   ├── sub-01_task-rest_eeg.vhdr@clean_prep.done
│   ├── sub-01_task-rest_eeg.vhdr@band_power.nc
│   └── sub-01_task-rest_eeg.vhdr@band_power.done
└── sub-02/eeg/
    └── sub-02_task-rest_eeg.vhdr@clean_prep.error  # clean_prep failed for this file
```

A second pipeline uses the cleaned files as its sources (`source_root: derivatives`,
`file_pattern: derivatives/**/*@clean_prep.fif`, `derivatives_path: derivatives2`) with a
derivative `Entropy`. The tree below `source_root` is mirrored (`code/` is never discovered)
and the source name is kept verbatim, so the last `@` separates it from `Entropy`:

```text
derivatives2/
└── sub-01/eeg/
    ├── sub-01_task-rest_eeg.vhdr@clean_prep.fif@Entropy.json
    └── sub-01_task-rest_eeg.vhdr@clean_prep.fif@Entropy.done
```

Dataframe columns from the first pipeline (wide format, `band_power` with `for_dataframe: true`;
`file_path` is the absolute path of the source, here with the project in `/study`):

```text
index  dataset  file_path                                           band_power.nc@freqbands-alpha_spaces-Fp1-F3  band_power.nc@freqbands-alpha_spaces-O1  ...
0      demo     /study/rawdata/sub-01/eeg/sub-01_task-rest_eeg.vhdr  0.32                                         0.29                                     ...
```

To find all `band_power` outputs on disk: `find derivatives -name '*@band_power.*'`.
