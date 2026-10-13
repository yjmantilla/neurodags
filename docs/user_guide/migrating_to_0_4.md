# Migrating to 0.4

```{admonition} Upgrade checklist
:class: important

1. **Install 0.4.0** and check which copy runs:
   `python -c "import neurodags; print(neurodags.__version__, neurodags.__file__)"`.
2. **Preview** from the directory you ran 0.3.x from:
   `neurodags migrate pipeline.yml --report plan.json` (changes nothing).
3. **Read the preview**: confirm `source_root` and the artifact sets, look at orphans, and
   resolve every `REFUSED` line. The most common one is a
   [pilot copy next to a full-run copy](#pilot-duplicates).
4. **Apply once per output tree / mount**: `neurodags migrate pipeline.yml --apply`, then
   `neurodags migrate pipeline.yml --mount hpc --apply` on the other machine.
5. **Check** with `neurodags status pipeline.yml` (migrated units show as `done` with a
   `(migrated)` count).
6. **Rebuild dataframes** with `neurodags dataframe` (column names changed).
7. **Update downstream code** with the report's `column_map` (`X__error` → `X.error`,
   coordinate encoding), and edit chained `file_pattern`s that name old `@Token` outputs.
8. **Regenerate SLURM scripts and input lists**: `neurodags slurm-script ...`, then
   `neurodags inputs pipeline.yml -o inputs.tsv` (file indices changed once: discovery is
   now sorted).
```

neurodags 0.4.0 changes how outputs are named, where they go and when they count as done
(see {doc}`naming`). A 0.3.x configuration is refused by every command until it declares
`naming_version: 1`, and 0.3.x outputs have no `.done` manifest, so they would all be
recomputed. `neurodags migrate` converts both in one pass:

- **configuration**: adds `naming_version: 1` and a `source_root` per dataset (and per mount),
  rewrites relative dataset paths so they still point to the same places (0.3.x resolved
  them against the working directory, 0.4.0 against the YAML that defines them), and
  optionally renames derivatives (`--map`);
- **outputs**: renames `@CamelCase` files to the verbatim YAML name
  (`x.vhdr@EpHiguchifd.nc` → `x.vhdr@Ep_HiguchiFD.nc`) and moves them to the 0.4.0 folders
  (computed from `source_root`), re-saving MNE split recordings with MNE;
- **manifests**: writes a `.done` (with `"origin": "migrate"`) for every migrated unit whose
  set of artifacts is known, so `neurodags status` shows it as `done (migrated)` and the
  next run does not recompute it.

It **previews by default** and changes nothing until you add `--apply`.

```text
neurodags migrate pipeline.yml [-d datasets.yml] [--mount NAME]
                  [--apply] [--report plan.json] [--map OLD=NEW ...]
                  [--legacy-cwd DIR] [--no-synthesize]
                  [--source-root DATASET[:MOUNT]=PATH ...] [--upstream-map OLDTOKEN=NEWNAME ...]
```

## Step by step

1. **Back up.** Commit your pipeline and datasets YAML files (migrate also keeps
   `pipeline.yml.bak-0.3` / `datasets.yml.bak-0.3`). The outputs are moved in place, never
   copied: if the tree matters, snapshot it first (or make sure you can recompute it).
2. **Run from where you ran 0.3.x.** In 0.3.x, relative paths inside the datasets
   (`file_pattern`, `derivatives_path`, `exclude_pattern`) were resolved against the
   directory you launched `neurodags` from. Run migrate from that same directory, or pass
   `--legacy-cwd DIR`. The preview prints the directory it used.
3. **Preview and keep the plan:**

   ```bash
   neurodags migrate pipeline.yml --report plan.json
   ```

4. **Read the preview** (details in [the next section](#reading-the-preview)). In
   particular, confirm the suggested `source_root` values and the artifact sets, check the
   orphans, and resolve anything listed under `REFUSED` (nothing is applied while a
   refusal remains; the command exits with code `2`).
5. **Apply, once per output tree (per mount):**

   ```bash
   neurodags migrate pipeline.yml --apply --report plan.json
   ```

   Moves and MNE rewrites happen first; the configuration files are written last, only
   after every move succeeded. Re-running is safe: an applied plan is a no-op
   (`Nothing to migrate`), and an interrupted run continues where it stopped.

   On a second machine or mount (say `hpc`), run it again there after the configuration
   was migrated: `neurodags migrate pipeline.yml --mount hpc --apply`. The configuration is
   already at `naming_version: 1`, so only that mount's tree is migrated. Keep the
   `pipeline.yml.bak-0.3` backup next to the pipeline (it shows the configuration was
   migrated by this tool) and pass the same `--map` options as the first time.
6. **Check:** `neurodags status pipeline.yml`. Migrated units are counted as `done`, with a
   `(migrated)` column; `--format json --include-files` labels them `done (migrated)`.
   Units without a `.done` (partial or errored ones, see below) show as `incomplete` or
   `errored` and are recomputed by the next `neurodags run`.
7. **Rebuild dataframes** with `neurodags dataframe`. Column names changed (see
   [Dataframe consumers](#dataframe-consumers)).
8. **Update downstream code** (analysis scripts, notebooks, other pipelines) using the
   column map of the report, and patterns such as `*@CleanPrep.fif` that refer to output
   file names (printed by the preview, never edited by migrate; see
   [Chained pipelines](#chained-pipelines)).
9. **Regenerate HPC scripts**: file indices changed once (sorted discovery) and the
   `slurm-script` templates now use a frozen `inputs.tsv`; regenerate them and the list (see
   {doc}`hpc`).

(reading-the-preview)=

## Reading the preview

| Section | What it means |
|---|---|
| `Configuration` | Each file that will be rewritten, with its changes (`+` added, `~` changed). If a file cannot be edited in place (for example a flow-style mapping `{name: ..., file_pattern: ...}`), it is re-serialized and the preview says its comments are lost (the backup keeps them). |
| `source_root` | The suggested value per dataset and mount: the fixed part of `file_pattern` before the first wildcard. Confirm it, or pass `--source-root DATASET=PATH` (all mounts) / `--source-root DATASET:MOUNT=PATH`. The value is written verbatim, so a relative path is relative to the YAML. |
| `on-disk names` | The 0.3.x CamelCase token and the 0.4.0 name of every derivative whose spelling changes. Names without `_` are unchanged. |
| `moves` | Every planned `old → new` file move, relative to the output tree. |
| `MNE split recordings` | Recordings saved in several parts (`X.fif`, `X-1.fif`, ...) whose name changes; they are re-saved with MNE (see below). |
| `orphans` | 0.3.x outputs whose source no longer exists (or is no longer matched by `file_pattern`). Left in place; delete them when you are sure. |
| `unresolved` | 0.3.x outputs of a chained pipeline (`x.vhdr&CleanPrep.fif@Feature.nc`) whose upstream source (`x.vhdr@CleanPrep.fif`) is not discovered. **Refused**, so they are not silently recomputed (see [Chained pipelines](#chained-pipelines)). |
| `ambiguous` | Outputs that several current sources could own. **Refused.** |
| `duplicates` | Old outputs identical to an existing new file. Left in place; delete them when happy. |
| `unrecognized` | Files with `@` that do not belong to this pipeline's derivatives (another pipeline's outputs, notes, ...). Left alone. |
| `Artifact sets` | Per derivative, the most common set of artifact suffixes among its units and how many units have each set. A unit gets a `.done` only when it has the whole set. |
| `Chained pipelines` | Dataset patterns and other YAML files in the same folder that refer to old output names. Edit them by hand. |
| `REFUSED` | Why nothing will be applied (see [Refusals](#refusals)). |

`--report plan.json` writes all of it as JSON: `moves`, `mne_rewrites`, `manifests`,
`without_manifest` (with the reason), `orphans`, `unresolved`, `ambiguous`, `duplicates`, `unrecognized`,
`artifact_sets`, `config` (files, changes, `source_root` values, renames), `column_map`,
`chained_patterns`, `refusals` and `notes`. After `--apply` it also has an `applied`
section.

## How outputs are matched

0.3.x placed outputs below a root computed from the files it discovered (with
`find_unique_root`), so the folder of an output depended on which files were matched at the
time (a single-subject pilot run wrote everything at the top of the tree). Migrate never
recomputes that root: it **scans** each `derivatives_path` (skipping `code/` and
`.neurodags/`) for file names containing `@` and parses them with the 0.3.x rules (the token
after `@` is the CamelCase spelling of a YAML name, longest first, followed by `.`, an MNE
split tail `-<n>.fif`, or `.error`/`.skip`). An old output is matched to the current source
whose 0.4.0 output folder **ends with** the old folder and whose base name, spelled the 0.3.x
way (every `@` written as `&`), equals the old source part. Every historical 0.3.x root lies at
or below `source_root`, so this works for full runs, pilot runs, `max_files_per_dataset` runs
and the pre-0.3.3 single-file form (`<derivatives_path>@Deriv.*` next to the tree).

- One candidate: the unit moves (`os.replace`) to the 0.4.0 name, with the source basename
  verbatim: `x.vhdr&CleanPrep.fif@BandPower.nc` becomes `x.vhdr@CleanPrep.fif@band_power.nc`.
  `.error` and `.skip` markers move with it. A source really named with `&` matches its own
  name and keeps the `&`; `&` is never decoded on its own.
- No candidate: orphan, left in place, unless the old source part looks like a 0.3.x chained
  output (`...&<Name><suffix>`): then it is *unresolved* and refused.
- Several candidates (for example `rec.txt` from a pilot run rooted at `sub-01/eeg/`, when
  `sub-02/eeg/rec.txt` also exists, or sources `x@A.fif` and `x&A.fif` that 0.3.x wrote to the
  same `x&A.fif@...` outputs): ambiguous, refused.

Datasets without `derivatives_path` keep their outputs next to the sources; only the names
change.

## MNE split recordings

MNE writes large recordings as `X.fif`, `X-1.fif`, ... and the first part stores the next
part's file name, so the parts cannot simply be renamed. For each such recording whose name
changes, migrate:

1. reads it with MNE (`read_raw_fif`, or `read_epochs` when it is not a raw file);
2. saves it under the new name with the same split size and the original data format (raw:
   `orig_format`; epochs: `single`, then `double` if `single` does not reproduce the data),
   through `neurodags.writers.save_mne`, which reports every part written;
3. reloads it and checks that the data are identical (`numpy.array_equal`) and that the
   channel names and types, sampling rate, bad channels, measurement date and, for raw,
   first sample, length and annotations (for epochs: events, event ids and times) match;
4. only then deletes the old parts. The new parts are listed in the unit's `.done`.

If the check fails, the new files are removed, the old recording stays where it is, and the
unit is listed after `--apply` as "left in place, to be recomputed". A recording whose file
name does not change (only its folder) is moved part by part without rewriting (the parts
refer to each other by base name only). The whole recording is loaded into memory.

## Manifests for migrated outputs

0.3.x records no list of artifacts per unit, so a migrated unit gets a `.done` only when its
**artifact set is known**: the derivative's most common suffix set in the tree (one suffix
for most derivatives) is complete for the unit. Not certified (no `.done`, status
`incomplete` or `errored`, recomputed by the next run):

- units missing an artifact of that set (a partial 0.3.x write);
- units with an `.error` marker;
- every unit of a derivative whose two most common sets tie.

Units with a `.skip` marker keep it (status `skipped`). `--no-synthesize` writes no
manifests at all, so every migrated unit is recomputed.

(pilot-duplicates)=

## Pilot copy and full-run copy of the same subject

This is the refusal you are most likely to meet on a real 0.3.x project:

```text
REFUSED (nothing will be applied):
  - 2 sets of 0.3.x outputs map to the same new outputs
    study/data/derivatives/sub-01/ses-1/eeg/sub-01_ses-1_eeg.vhdr@band_power* with different
    content: study/data/derivatives/sub-01_ses-1_eeg.vhdr@BandPower*,
    study/data/derivatives/sub-01/ses-1/eeg/sub-01_ses-1_eeg.vhdr@BandPower*. Keep one
    (delete or move the others away) and run migrate again.
```

(One line in the real output, with absolute paths; wrapped here.)

**Why it happens.** 0.3.x placed outputs below the common root of the files it discovered. A
pilot run on one subject (one matching file, or `max_files_per_dataset: 1`) used that file's
own folder as the root, so it wrote the outputs **flat**, directly under `derivatives_path`
(`<derivatives_path>/<file>@Token.*`). The later full run had a different common root and
looked for the outputs in the **nested** folder (`<derivatives_path>/sub-01/ses-1/eeg/...`),
found nothing there and computed the subject again. Both copies belong to the same source,
so both map to the same 0.4.0 outputs. When the two copies are byte-identical migrate keeps
the deeper one and lists the other as a duplicate; usually they are not, because the files
embed run-specific data (timestamps in MNE reports and JSON sidecars, file ids and
measurement metadata in FIF files, NetCDF history attributes). Migrate cannot know which copy
you want, so it refuses.

**Fix.** Keep the full-run (nested) copy and remove the pilot (flat) one, then preview again:

1. Read the refusal: the shallower prefix is the pilot copy, usually
   `<derivatives_path>/<file>@Token.*` directly under `derivatives_path`.
2. Check that the nested copy is complete (it has every artifact of the derivative, and no
   `.error`).
3. Move the flat copy out of the tree rather than deleting it outright, e.g.
   `mkdir -p ../pilot-0.3 && mv <derivatives_path>/<file>@Token.* ../pilot-0.3/`
   (`find <derivatives_path> -maxdepth 1 -name '*@*'` lists every flat output). Include its
   `.error`/`.skip` markers.
4. Run `neurodags migrate pipeline.yml --report plan.json` again; repeat until nothing is
   refused, then `--apply`.

Keep the flat copy instead only if it is the one you trust (for example the full run
failed for that subject); then move the nested copy away.

(dataframe-consumers)=

## Dataframe consumers

Dataframe columns use the YAML names (as in 0.3.x), so the CamelCase file tokens do not
matter for them. What changes:

- error columns: `<name>__error` → `<name>.error`;
- coordinate columns (`<name><suffix>@<dim>-<value>_...`): 0.4.0 percent-encodes dims and
  values instead of the lossy 0.3.x swaps (`-`→`~`, `_`→`|`, `@`→`$`, `,`→`.`, empty →
  `NA`), e.g. `psd.nc@band-8~12` → `psd.nc@band-8-12`;
- renamed derivatives (`--map`) rename their columns.

The report's `column_map` lists, per derivative, the exact `name.suffix` columns and the
error column, plus the coordinate rule. For coordinate columns the map is **best effort**
(the 0.3.x encoding cannot be inverted when a value contained `,` or a dimension name
contained `-` or `_`); `neurodags.migrate.legacy_column_to_new` applies it to a column
name:

```python
from neurodags.migrate import legacy_column_to_new

legacy_column_to_new("psd.nc@band-8~12_ch-C3", ["psd"])   # 'psd.nc@band-8-12_ch-C3'
legacy_column_to_new("psd__error", ["psd"])                # 'psd.error'
```

The reliable path is to rebuild the dataframe from the (unchanged) artifacts with 0.4.0.

(chained-pipelines)=

## Chained pipelines

A pipeline whose sources are another pipeline's outputs (`file_pattern: .../*@CleanPrep.fif`)
depends on the upstream file names. 0.3.x wrote every `@` of such a source as `&`
(`x.vhdr&CleanPrep.fif@Feature.nc`); 0.4.0 keeps the source name verbatim
(`x.vhdr@CleanPrep.fif@Feature.nc`), and migrate renames these outputs too. Until they are
migrated, `run`, `dry-run`, `status` and `dataframe` stop with an error pointing to
`neurodags migrate` instead of recomputing them. Scripts that split chained file names on the
first `@` must split on the last one (`rsplit("@", 1)`), and a broad pattern such as
`*@CleanPrep*` now also matches downstream outputs in the same tree: anchor it on the artifact
suffix (`*@clean_prep.fif`).

1. Migrate the upstream pipeline. If its names change on disk (`@CleanPrep` →
   `@clean_prep`), its preview lists the downstream patterns it can see.
2. Edit the downstream `file_pattern` by hand (`*@clean_prep.fif`). migrate never edits
   patterns.
3. Migrate the downstream pipeline with `--upstream-map CleanPrep=clean_prep`: its outputs,
   named after the old upstream files (`x.vhdr&CleanPrep.fif@Feature.nc`), then match the
   renamed sources (`x.vhdr@clean_prep.fif`) and move to `x.vhdr@clean_prep.fif@Feature.nc`.
   Without it they are *unresolved* and the migration is refused (they would otherwise be
   recomputed). To recompute them on purpose, move them out of the tree first.

(refusals)=

## Refusals

Nothing is applied while any of these remains:

| Refusal | How to resolve |
|---|---|
| Ambiguous mapping (several sources match an old output) | Move the listed outputs out of the tree (or delete them); they are recomputed. |
| Unresolved chained outputs (`x&A.fif@B.nc` with no discovered `x@A.fif`) | Fix the downstream `file_pattern`, pass `--upstream-map OLDTOKEN=NEWNAME` when the upstream name changed, or move the listed outputs out of the tree to recompute them. |
| A target exists with different content | Keep one of the two files (delete or move the other) and run again. Identical files are not refused: the old one is left as a duplicate. |
| Two old outputs map to one target with different content (e.g. a pilot-era copy and a full-run copy of the same unit) | Keep one: see [Pilot copy and full-run copy](#pilot-duplicates). If they are identical, the deeper one is used and the other is left as a duplicate. |
| The configuration already declares `naming_version: 1` but the tree still has 0.3.x outputs, and there is no `pipeline.yml.bak-0.3` | Restore the 0.3.x configuration and migrate it, or (if it was migrated on another mount) keep the backup next to the pipeline. |
| A folder to write is not writable | Fix the permissions. |
| A move would cross filesystems | Copy the tree onto one filesystem first; cross-filesystem migration is out of scope. |
| No `source_root` can be suggested (the pattern starts at a filesystem root, `/*/...`) | Pass `--source-root DATASET=PATH`. |
| A derivative name is invalid in 0.4.0 (`.`, `@`, spaces, `Clean` + `Clean-1`, ...) | Rename it with `--map OLD=NEW`. |
| Two derivatives share an on-disk token (`band_power` and `BandPower`) | Rename or remove one before migrating. |
| Two datasets share a `derivatives_path` without distinct `derivatives_subdir` | Add `derivatives_subdir` to the datasets in the 0.3.x file, then migrate (their outputs move into the subfolders). |
| A `$VAR/...` dataset path expands to a relative path | Write it as an absolute path. |
| A `.bak-0.3` backup already exists with other content | Move the old backup away. |
| Datasets partly migrated (`source_root` on some only) | Finish them by hand or restore the 0.3.x file. |

## Safety and resuming

- The preview reads only; `--apply` re-checks every precondition (old files still there,
  targets still free, configuration unchanged since the plan) before touching anything.
- Moves use `os.replace` (atomic, same filesystem). Every completed operation is appended to
  `<derivatives_path>/.neurodags/migrate-journal.json` (JSON Lines; for datasets without
  `derivatives_path`, `.neurodags/` next to the pipeline). If a run is interrupted, run the
  same command again: it scans the tree, continues the remaining moves, completes
  half-moved units and uses the journal to finish (or redo) an interrupted MNE rewrite. Files
  already moved to a chained 0.4.0 name (`x@A.fif@B.nc`, which the 0.3.x parser rejects) are
  recognized as the planned target of their discovered source and join their unit, which then
  gets its `.done`.
- Configuration files are written last with `.bak-0.3` backups, so an interrupted run keeps
  the 0.3.x configuration and can simply be re-run.
- A datasets YAML shared by several pipelines is migrated with the first one; later
  migrations see that every dataset has a `source_root`, take its paths as 0.4.0 paths and
  leave it unchanged.
- Skipped datasets (`skip: true`) get a `source_root` but their outputs are not moved; to
  migrate them later, un-skip them and run migrate again (it then works on the tree only,
  which requires the `pipeline.yml.bak-0.3` backup next to the pipeline).
