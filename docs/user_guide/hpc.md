# HPC / SLURM Array Jobs

NeuroDAGs processes files independently — each file is self-contained. This maps naturally onto SLURM array jobs: one array task per file, all tasks running in parallel across nodes.

The key parameter is `only_index`, which restricts a run to specific file indices from the full file list. Combined with `$SLURM_ARRAY_TASK_ID`, each array task processes exactly one file. An index that does not exist (or is negative) is an error, never silently ignored.

---

## File indices and frozen input lists

Every invocation enumerates its inputs **once** and numbers them from `0`:

1. per dataset: glob `file_pattern`, apply `exclude_pattern`, drop split-FIF continuations,
   remove duplicates, check that every file lies below the dataset's `source_root` (a file
   outside it is a configuration error naming the file), then **sort by the path relative to
   `source_root`** (`/` separators, plain code-point order);
2. keep the first `max_files_per_dataset` files of each dataset (after sorting, so a limited
   run is a prefix of the full run);
3. concatenate the datasets in YAML order, skipping `skip: true` ones; the index is the
   position in that list.

The order is code-point order, not "natural" order: `sub-10` sorts **before** `sub-2`, and
upper-case letters before lower-case ones. Use zero-padded labels (`sub-02`, `sub-10`) if you
want the indices to follow the subject numbers.

Discovery depends on what is on disk, so adding or removing a file shifts the indices of the
files after it. For array jobs, **freeze the list once** and give it to every task:

```bash
neurodags inputs pipeline.yml -o inputs.tsv                 # discover once, write the list
N=$(neurodags count-inputs pipeline.yml --inputs inputs.tsv) # number of entries
sbatch --array=0-$((N - 1)) run_array.sh                     # workers pass --inputs / inputs=
```

`inputs.tsv` is a small text file: the header `# neurodags-inputs naming_version=1`, then
one `index<TAB>dataset<TAB>rel` line per input, where `rel` is the path relative to the
dataset's `source_root` with `/` separators:

```text
# neurodags-inputs naming_version=1
0	study	sub-01/eeg/sub-01_task-rest_eeg.vhdr
1	study	sub-02/eeg/sub-02_task-rest_eeg.vhdr
```

`run`, `dry-run`, `status`, `count-inputs` and `dataframe` accept `--inputs inputs.tsv` (a
path relative to the current working directory); the Python API takes `inputs=`
(`run_pipeline`, `iterate_derivative_pipeline`, `build_derivative_dataframe`). With a list:

- the list **replaces discovery**: indices come from the list, and each path is the dataset's
  `source_root` on the active mount joined with `rel`. Because only `rel` is stored, the same
  list works on your laptop and on the cluster (`mount_point: local` vs `hpc`);
- files added on disk later are **not processed**; their number is logged, and `status`
  prints it ("N file(s) on disk are not in the input list");
- listed files that disappeared keep their index and are reported as **`missing-source`**:
  `status` shows them, `run` skips them with a per-file error record, and `--only-index` on
  such an index fails explicitly;
- the list is validated before anything runs: header and version, dataset names that exist
  (and are not `skip: true`) in the configuration, no duplicate indices, indices contiguous
  from `0`;
- `--max-files-per-dataset` cannot be combined with `--inputs` (the list already fixes the
  inputs); write a pilot list instead.

---

## Generating submission scripts

Use `neurodags slurm-script` to generate a ready-to-edit SLURM template populated with your pipeline's actual derivative names and dependency order:

```bash
# per-file (default) — all derivatives per file task
neurodags slurm-script pipeline.yml --output run_array.sh

# flat — file x derivative flat array (max parallelism, independent derivatives only)
neurodags slurm-script pipeline.yml --pattern flat --output run_array_per_deriv.sh

# chained — per-derivative sequential arrays linked via --dependency=afterok
neurodags slurm-script pipeline.yml --pattern chained --output submit_pipeline.sh
```

The `chained` pattern writes two files: `submit_pipeline.sh` (the submission script) and `run_one_derivative.sh` (the worker). Edit the `#SBATCH` resource lines before submitting.

All templates use the frozen-list flow: generate `inputs.tsv` once with `neurodags inputs`,
size the array from it, and the workers read it (`inputs=os.path.abspath("inputs.tsv")`, so
submit from the directory that holds `inputs.tsv`). Keep `inputs.tsv` until every array of
the run, including resubmissions, has finished.

The selection given to `slurm-script` is carried into every generated command and worker:
`-d datasets.yml` (resolved against the directory you run `slurm-script` from, and written
into the script as an absolute path) is passed to `neurodags inputs`/`count-inputs` and as
`datasets_configuration=` to `run_pipeline`, and `--derivative` limits what the workers run
(per-file workers pass `derivatives=[...]`; flat and chained arrays only list the selected
derivatives). Paths are shell-quoted, and the Python part of each worker is a quoted heredoc
(`python - "$PIPELINE" ... <<'EOF'`) that reads them from `sys.argv`, so unusual file names
(spaces, quotes, `$`) reach `run_pipeline` unchanged.

### Counting source files

`neurodags count-inputs` prints the number of input entries the pipeline will process, i.e. the size of the index range `--only-index` accepts (`0` to `N-1`). A file listed under two dataset entries counts twice, because it has two indices. With `--inputs`, it prints the number of entries in the list (`missing-source` ones included, so the array covers every index of the list). Useful in submission scripts:

```bash
neurodags inputs pipeline.yml -o inputs.tsv
N=$(neurodags count-inputs pipeline.yml --inputs inputs.tsv)
sbatch --array=0-$((N - 1)) run_array.sh
```

### Estimating output file count

Clusters often enforce per-job or total output file quotas. Before submitting, estimate how many output files the pipeline will produce:

```bash
neurodags status pipeline.yml --format json
```

The JSON output contains:

- `n_files` — number of source (input) files
- `grand_total.total` — total `(derivative × source-file)` pairs across all derivatives with `save: True`
- `grand_total.total / n_files` — estimated output files **per source file**

```bash
# estimated output files per source file and total
neurodags status pipeline.yml --format json \
  | python3 -c "
import json, sys
d = json.load(sys.stdin)
per_file = d['grand_total']['total'] // d['n_files']
print(f\"derivatives per file : {per_file}\")
print(f\"estimated total files: {d['grand_total']['total']}\")
"
```

**Caveat:** these counts are derivative instances, not actual output files. Most derivatives write exactly one file, so the estimate is reliable for typical pipelines. Derivatives that return multiple artifacts (multiple keys in `NodeResult.artifacts`) will produce more files than this estimate — inspect those nodes manually if precise quota compliance is required.

---

## per-file: One Array Task per File (All Derivatives)

Each task runs all derivatives in dependency order for a single file.

```bash
neurodags inputs pipeline.yml -o inputs.tsv
N=$(neurodags count-inputs pipeline.yml --inputs inputs.tsv)
sbatch --array=0-$((N - 1)) run_array.sh
```

```bash
#!/bin/bash
# run_array.sh
#SBATCH --job-name=neurodags
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/neurodags_%A_%a.out
#SBATCH --error=logs/neurodags_%A_%a.err

source activate myenv

python - <<'EOF'
import os
from neurodags.loaders import load_pipeline_configuration
from neurodags.orchestrators import run_pipeline

config_path = os.path.abspath("pipeline.yml")
config = load_pipeline_configuration(config_path)  # requires naming_version: 1
run_pipeline(
    config,
    base_dir=os.path.dirname(config_path),  # relative inline dataset paths (0.4.0)
    inputs=os.path.abspath("inputs.tsv"),   # frozen input list
    only_index=int(os.environ["SLURM_ARRAY_TASK_ID"]),
    raise_on_error=True,
    skip_errors=True,
)
EOF
```

`raise_on_error=True` makes the SLURM task exit with a non-zero code on failure, so `sacct` correctly reports failed tasks. `skip_errors=True` skips files whose previous run wrote a `.error` marker.

---

## flat: One Array Task per File × Derivative

Use this when derivatives are independent (no inter-derivative dependencies) and you want maximum parallelism.

```bash
neurodags inputs pipeline.yml -o inputs.tsv
N=$(neurodags count-inputs pipeline.yml --inputs inputs.tsv)
N_DERIVATIVES=3
TOTAL=$(( N * N_DERIVATIVES ))
sbatch --array=0-$((TOTAL - 1)) run_array_per_deriv.sh
```

```bash
#!/bin/bash
# run_array_per_deriv.sh
#SBATCH --job-name=neurodags
#SBATCH --time=02:00:00
#SBATCH --mem=8G
#SBATCH --output=logs/neurodags_%A_%a.out

# Task ID encodes both file index and derivative index
# e.g. 3 derivatives × N files → array size = 3 × N
# SLURM_ARRAY_TASK_ID = file_index * n_derivatives + derivative_index

source activate myenv

DERIVATIVES=("Preprocessed" "Spectrum" "BandPower")
N_DERIVATIVES=${#DERIVATIVES[@]}

FILE_INDEX=$(( SLURM_ARRAY_TASK_ID / N_DERIVATIVES ))
DERIV_INDEX=$(( SLURM_ARRAY_TASK_ID % N_DERIVATIVES ))
DERIVATIVE=${DERIVATIVES[$DERIV_INDEX]}

# Quoted heredoc: the shell passes values as arguments, never into the Python source.
python - "$DERIVATIVE" "$FILE_INDEX" pipeline.yml <<'EOF'
import os
import sys
from neurodags.loaders import load_pipeline_configuration
from neurodags.orchestrators import run_pipeline

config_path = os.path.abspath(sys.argv[3])
config = load_pipeline_configuration(config_path)  # requires naming_version: 1
run_pipeline(
    config,
    base_dir=os.path.dirname(config_path),  # relative inline dataset paths (0.4.0)
    inputs=os.path.abspath("inputs.tsv"),   # frozen input list
    derivatives=[sys.argv[1]],
    only_index=int(sys.argv[2]),
    raise_on_error=True,
)
EOF
```

> **Note**: When derivatives have inter-dependencies (e.g. `Spectrum` reads `Preprocessed` output), this pattern requires `Preprocessed` to finish before `Spectrum` starts. Use SLURM `--dependency=afterok` between derivative-level job arrays, or use Pattern 1 which respects dependency order automatically.

---

## chained: Per-Derivative Sequential Array

Run derivatives one at a time in dependency order, with all files parallelised within each derivative. Use SLURM job dependencies to chain them.

```bash
#!/bin/bash
# submit_pipeline.sh

neurodags inputs pipeline.yml -o inputs.tsv
N=$(neurodags count-inputs pipeline.yml --inputs inputs.tsv)
ARRAY="0-$((N - 1))"

# The derivative is the worker's argument. --export=ALL keeps PATH and the module/venv
# environment; an environment variable would not do, because with ALL an inherited value
# of the same name takes precedence over one set in --export.
JOB1=$(sbatch --parsable --array=$ARRAY --export=ALL \
    run_one_derivative.sh Preprocessed)

JOB2=$(sbatch --parsable --array=$ARRAY --export=ALL \
    --dependency=afterok:$JOB1 \
    run_one_derivative.sh Spectrum)

sbatch --array=$ARRAY --export=ALL \
    --dependency=afterok:$JOB2 \
    run_one_derivative.sh BandPower
```

```bash
#!/bin/bash
# run_one_derivative.sh
#SBATCH --job-name=neurodags
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --output=logs/%x_%A_%a.out

SELECTED_DERIVATIVE="$1"  # sbatch ... run_one_derivative.sh <derivative>

source activate myenv

python - "$SELECTED_DERIVATIVE" pipeline.yml <<'EOF'
import os
import sys
from neurodags.loaders import load_pipeline_configuration
from neurodags.orchestrators import run_pipeline

config_path = os.path.abspath(sys.argv[2])
config = load_pipeline_configuration(config_path)  # requires naming_version: 1
run_pipeline(
    config,
    base_dir=os.path.dirname(config_path),  # relative inline dataset paths (0.4.0)
    inputs=os.path.abspath("inputs.tsv"),   # frozen input list
    derivatives=[sys.argv[1]],
    only_index=int(os.environ["SLURM_ARRAY_TASK_ID"]),
    raise_on_error=True,
    skip_errors=True,
)
EOF
```

---

## Resubmitting missing, incomplete or errored files

`neurodags status` reports the global **file index** of every missing, incomplete or errored
file (see {ref}`Unit states <naming-unit-states>`) — the
same index the per-file and chained arrays use as `SLURM_ARRAY_TASK_ID` (see
{doc}`inspection`). `--format indices` prints just those indices, space-separated. Pass the
**same input list** the array used, so the indices refer to the same files:

```bash
# --exit-zero: exit 0 even when something is missing/incomplete/errored, but still fail on a
# broken config, so `set -e` keeps protecting you
IDX=$(neurodags status pipeline.yml --inputs inputs.tsv --format indices --exit-zero)
[ -z "$IDX" ] && { echo "nothing to resubmit"; exit 0; }

# run them locally / in one job ...
neurodags run pipeline.yml --inputs inputs.tsv --only-index $IDX

# ... or as a sparse per-file array (sbatch wants a comma-separated list)
sbatch --array=$(echo "$IDX" | tr ' ' ',') retry_array.sh
```

**Report / resubmission jobs chained with `--dependency=afterok`.** By default `status` exits
`1` whenever anything is missing, incomplete or errored, which is the expected state after a partial run
or a staged array — so a report job would fail and SLURM would cancel every job chained after
it. Use `--exit-zero` in such jobs: `status` then exits `0` whenever the report was produced,
in every `--format`, while real failures (missing or invalid config, unset environment
variable, unreadable `-d` file) still exit non-zero and correctly break the chain.

```bash
REPORT=$(sbatch --parsable --dependency=afterok:$COMPUTE_JOB \
    --wrap "neurodags status pipeline.yml --format json --exit-zero > status.json")
```

Do not use `|| true` for this (as older versions of this guide did): it also turns a broken
config or an unset `$VAR` into a "successful" job with an empty or misleading report.

Narrow the list with `--list-missing` or `--list-errors`, and per derivative with
`--derivative` (for the chained pattern, resubmit each derivative's array with its own list:
`neurodags status pipeline.yml --derivative Spectrum --format indices`).
`--format json` carries the same lists as `missing_indices` / `incomplete_indices` /
`errored_indices`, and
`--format json --include-files` adds one record per file × derivative with the index,
dataset and path, so a script can verify the index → path mapping before submitting.

Things to keep in mind:

- **Retry workers must not use `skip_errors=True`.** The per-file and chained templates above
  (and the ones `neurodags slurm-script` emits) pass `skip_errors=True`, which skips exactly
  the errored files you are trying to retry. Use a copy of the worker without it (the default
  `skip_errors=False` retries files with a `.error` marker), e.g. `retry_array.sh` = `run_array.sh`
  minus the `skip_errors=True` line. Partial outputs left by a failed or killed task have no
  `.done` manifest, so the retry recomputes them without any cleanup.
- **Killed tasks never block retries.** Artifacts are written in a hidden staging folder and
  moved into place only when complete (see
  {ref}`Atomic publication <naming-atomic-publication>`), so a task killed by a time limit,
  preemption or `scancel` (`SIGTERM` or `SIGKILL`) leaves no partial file at an output name and
  no `.done`; the retry recomputes the unit. It may leave its staging folder
  (`.<output prefix>.tmp~<token>/`), which nothing reads; `neurodags status` counts them.
  Delete them when no job is writing to the tree, e.g.
  `find <derivatives_path> -name '.*.tmp~*' -prune -mmin +1440 -exec rm -rf {} +`.
- **Never run two tasks on the same unit at once, and never overwrite a unit another task is
  reading.** This is not supported (see {ref}`Concurrency <naming-concurrency>`): interleaved
  publications can mix files of two runs (detected afterwards by the `.done` fingerprints, so
  the unit is recomputed), and a reader can mix old and new data without any error. Cancel a
  task before requeueing it, run dependencies before their consumers (per-file pattern or
  chained arrays), and do not let two derivatives of one file compute the same uncached
  intermediate in parallel.
- **Flat-array task IDs are not file indices.** In the flat pattern
  `SLURM_ARRAY_TASK_ID = file_index * N_DERIVATIVES + derivative_index`, so never pass status
  indices straight to `run_array_per_deriv.sh`. Convert them per derivative, using the
  position of the derivative in that worker's `DERIVATIVES` array:

  ```bash
  DERIVATIVES=("Preprocessed" "Spectrum" "BandPower")
  N_DERIVATIVES=${#DERIVATIVES[@]}
  TASKS=()
  for D in "${!DERIVATIVES[@]}"; do
    for F in $(neurodags status pipeline.yml --inputs inputs.tsv \
                 --derivative "${DERIVATIVES[$D]}" \
                 --format indices --exit-zero); do
      TASKS+=($(( F * N_DERIVATIVES + D )))
    done
  done
  [ ${#TASKS[@]} -gt 0 ] && sbatch --array=$(IFS=,; echo "${TASKS[*]}") run_array_per_deriv.sh
  ```

- **Indices are positions, not IDs.** Without `--inputs` they are stable only while the
  dataset configuration and the set of input files stay the same. With a frozen list they
  stay fixed for as long as you keep the list, on every mount point. `--format indices` never
  lists `missing-source` entries (listed files that no longer exist cannot be recomputed);
  `--format json` reports them as `missing_source_indices`.

---

## Checking Failed Tasks

After a run, find failed files via `.error` markers:

```bash
find derivatives/ -name "*.error" | sort
```

Each `.error` file contains the derivative name, step, and exception. To retry only failed files, delete their `.error` markers and resubmit — or use `skip_errors=False` (the default) which retries them automatically.

To skip known-bad files and process only new/uncached ones:

```python
run_pipeline(config, inputs="inputs.tsv", only_index=index, skip_errors=True)
```

---

## Tips

| Concern | Recommendation |
|---------|---------------|
| Cluster paths differ from local | Set `mount_point: hpc` in `pipeline.yml`; give `source_root`, `file_pattern` and `derivatives_path` an `hpc:` entry each (`neurodags validate` prints the resolved values for the active mount) |
| Relative dataset paths | Since 0.4.0 they are resolved against the datasets YAML's directory, not the job's working directory; a config passed as a dict needs `base_dir=` (the generated `slurm-script` templates pass it) |
| Node has multiple cores | Set `n_jobs: -1` for intra-job file parallelism (combine with array jobs for two-level parallelism) |
| Partial reruns | Caching is automatic — resubmit the full array; units with a valid `.done` manifest are skipped, and tasks killed mid-write (no `.done`) are recomputed; nothing needs to be unlocked or cleaned first |
| Outputs computed on another mount | Copy the derivatives tree as is with `rsync -a` or `cp -a`: manifests list files relative to their folder and record their size and modification time, so the copy verifies on the new mount as long as the paths below `source_root` are the same (a plain `cp` changes mtimes and the units are recomputed) |
| Debugging | Run `dry_run=True` first; check `has_error_marker` column for prior failures |
| `--only-index` errors | An unknown or negative index is an error (exit `2`), never ignored: size the array from `count-inputs --inputs inputs.tsv`, and pass the same `--inputs` list to `status` and the workers |
| Upgrading from 0.3.x | File indices changed once (sorted discovery) and the templates changed (frozen input list, `load_pipeline_configuration`, `base_dir=`): regenerate the scripts with `neurodags slurm-script` and run `neurodags migrate` once per output tree (see {doc}`migrating_to_0_4`) |
| Array size limit | Most clusters cap at 1000–10000 tasks; split `--array` into index-range batches (e.g. `0-999`, `1000-1999`). With a frozen `inputs.tsv`, each task ID maps to the same file in every batch, so range batching is safe and idempotent — cached files are skipped automatically on resubmit |
| Index order | Files are sorted by their path below `source_root` in code-point order (`sub-10` before `sub-2`); zero-pad labels (`sub-02`) for a natural order |
