# Parallelism and Execution Control

## Local Parallel Execution

NeuroDAGs uses joblib for file-level parallelism. Each file is an independent job.

### Via pipeline.yml

```yaml
n_jobs: 4           # -1 = all available cores, 1 or null = serial
joblib_backend: loky
joblib_prefer: processes
```

### Via Python

```python
from neurodags.orchestrators import run_pipeline

run_pipeline(config, n_jobs=4, joblib_backend="loky", joblib_prefer="processes")
```

### Via CLI

```bash
neurodags run pipeline.yml --derivative MyDerivative --n-jobs 4 --joblib-backend loky
```

## Subset Execution

Process only a subset of files by index:

```python
from neurodags.orchestrators import run_pipeline

# Single file
run_pipeline(config, derivatives=["MyDerivative"], only_index=0)

# Multiple files
run_pipeline(config, derivatives=["MyDerivative"], only_index=[0, 2, 5])

# Limit files per dataset
run_pipeline(config, derivatives=["MyDerivative"], max_files_per_dataset=10)
```

## Error Handling

By default (`raise_on_error=False`) errors are caught per-file, logged with full traceback, and a `.error` marker written — then execution continues with the next file. To stop immediately on first failure:

```python
from neurodags.orchestrators import run_pipeline

run_pipeline(config, derivatives=["MyDerivative"], raise_on_error=True)
```

`raise_on_error=True` raises `RuntimeError` with the file path, derivative name, and traceback. Useful for CI or single-file debugging where you want a hard stop rather than a partial run.

## HPC / SLURM

For SLURM array job templates and submission patterns see {doc}`hpc`.
