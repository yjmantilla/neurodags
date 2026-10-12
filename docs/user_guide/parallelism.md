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

Process only a subset of files by index. The index is the position in the sorted input list
(datasets in YAML order, files sorted by their path below `source_root`; see
{ref}`Discovery and file order <datasets-discovery-order>`); an unknown or negative index raises
`neurodags.iterators.IndexSelectionError` before anything is written. Pass `inputs=` (a list
written by `neurodags inputs`) to keep the indices fixed while files are added:

```python
from neurodags.orchestrators import run_pipeline

# Single file
run_pipeline(config, derivatives=["MyDerivative"], only_index=0)

# Multiple files
run_pipeline(config, derivatives=["MyDerivative"], only_index=[0, 2, 5])

# Limit files per dataset (the first 10 in sorted order)
run_pipeline(config, derivatives=["MyDerivative"], max_files_per_dataset=10)

# Indices of a frozen input list
run_pipeline(config, inputs="inputs.tsv", only_index=[0, 2, 5])
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
