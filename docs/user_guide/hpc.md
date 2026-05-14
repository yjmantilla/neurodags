# HPC / SLURM Array Jobs

NeuroDAGs processes files independently — each file is self-contained. This maps naturally onto SLURM array jobs: one array task per file, all tasks running in parallel across nodes.

The key parameter is `only_index`, which restricts a run to specific file indices from the full file list. Combined with `$SLURM_ARRAY_TASK_ID`, each array task processes exactly one file.

---

## Pattern 1: One Array Task per File (All Derivatives)

Each task runs all derivatives in dependency order for a single file.

### Step 1 — Count files

```python
# count_files.py
from neurodags.loaders import load_configuration
from neurodags.orchestrators import run_pipeline

config = load_configuration("pipeline.yml")
plan = run_pipeline(config, dry_run=True)
print(plan["file_path"].nunique())
```

```bash
N=$(python count_files.py)
echo "Submitting array for $N files"
sbatch --array=0-$((N - 1)) run_array.sh
```

### Step 2 — Array job script

```bash
#!/bin/bash
#SBATCH --job-name=neurodags
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/neurodags_%A_%a.out
#SBATCH --error=logs/neurodags_%A_%a.err

source activate myenv

python - <<'EOF'
import os
from neurodags.loaders import load_configuration
from neurodags.orchestrators import run_pipeline

config = load_configuration("pipeline.yml")
index = int(os.environ["SLURM_ARRAY_TASK_ID"])
run_pipeline(config, only_index=index, raise_on_error=True)
EOF
```

`raise_on_error=True` makes the SLURM task exit with a non-zero code on failure, so `sacct` correctly reports failed tasks.

On resubmission, already-cached files are skipped automatically. Add `--skip-errors` to also skip files whose previous run wrote a `.error` marker:

```bash
run_pipeline(config, only_index=index, raise_on_error=True, skip_errors=True)
```

---

## Pattern 2: One Array Task per File × Derivative

Use this when derivatives are independent (no inter-derivative dependencies) and you want maximum parallelism.

```bash
#!/bin/bash
#SBATCH --job-name=neurodags
#SBATCH --time=02:00:00
#SBATCH --mem=8G
#SBATCH --output=logs/neurodags_%A_%a.out

# Task ID encodes both file index and derivative index
# e.g. 3 derivatives × N files → array size = 3 × N
# SLURM_ARRAY_TASK_ID = file_index * n_derivatives + derivative_index

DERIVATIVES=("Preprocessed" "Spectrum" "BandPower")
N_DERIVATIVES=${#DERIVATIVES[@]}

FILE_INDEX=$(( SLURM_ARRAY_TASK_ID / N_DERIVATIVES ))
DERIV_INDEX=$(( SLURM_ARRAY_TASK_ID % N_DERIVATIVES ))
DERIVATIVE=${DERIVATIVES[$DERIV_INDEX]}

python - <<EOF
from neurodags.loaders import load_configuration
from neurodags.orchestrators import run_pipeline

config = load_configuration("pipeline.yml")
run_pipeline(
    config,
    derivatives=["$DERIVATIVE"],
    only_index=$FILE_INDEX,
    raise_on_error=True,
)
EOF
```

Submit with:

```bash
N_FILES=$(python count_files.py)
N_DERIVATIVES=3
TOTAL=$(( N_FILES * N_DERIVATIVES ))
sbatch --array=0-$((TOTAL - 1)) run_array_per_deriv.sh
```

> **Note**: When derivatives have inter-dependencies (e.g. `Spectrum` reads `Preprocessed` output), this pattern requires `Preprocessed` to finish before `Spectrum` starts. Use SLURM `--dependency=afterok` between derivative-level job arrays, or use Pattern 1 which respects dependency order automatically.

---

## Pattern 3: Per-Derivative Sequential Array

Run derivatives one at a time in dependency order, with all files parallelised within each derivative. Use SLURM job dependencies to chain them.

```bash
#!/bin/bash
# submit_pipeline.sh

N=$(python count_files.py)
ARRAY="0-$((N - 1))"

JOB1=$(sbatch --parsable --array=$ARRAY \
    --export=DERIVATIVE=Preprocessed \
    run_one_derivative.sh)

JOB2=$(sbatch --parsable --array=$ARRAY \
    --dependency=afterok:$JOB1 \
    --export=DERIVATIVE=Spectrum \
    run_one_derivative.sh)

sbatch --array=$ARRAY \
    --dependency=afterok:$JOB2 \
    --export=DERIVATIVE=BandPower \
    run_one_derivative.sh
```

```bash
#!/bin/bash
# run_one_derivative.sh
#SBATCH --job-name=neurodags
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --output=logs/%x_%A_%a.out

python - <<EOF
import os
from neurodags.loaders import load_configuration
from neurodags.orchestrators import run_pipeline

config = load_configuration("pipeline.yml")
run_pipeline(
    config,
    derivatives=["$DERIVATIVE"],
    only_index=int(os.environ["SLURM_ARRAY_TASK_ID"]),
    raise_on_error=True,
    skip_errors=True,
)
EOF
```

---

## Checking Failed Tasks

After a run, find failed files via `.error` markers:

```bash
find derivatives/ -name "*.error" | sort
```

Each `.error` file contains the derivative name, step, and exception. To retry only failed files, delete their `.error` markers and resubmit — or use `skip_errors=False` (the default) which retries them automatically.

To skip known-bad files and process only new/uncached ones:

```bash
run_pipeline(config, only_index=index, skip_errors=True)
```

---

## Tips

| Concern | Recommendation |
|---------|---------------|
| Cluster paths differ from local | Set `mount_point: hpc` in `pipeline.yml` |
| Node has multiple cores | Set `n_jobs: -1` for intra-job file parallelism (combine with array jobs for two-level parallelism) |
| Partial reruns | Caching is automatic — resubmit the full array, cached files are skipped |
| Debugging | Run `dry_run=True` first; check `has_error_marker` column for prior failures |
| Array size limit | Most clusters cap at 1000–10000 tasks; use `max_files_per_dataset` to batch if needed |
