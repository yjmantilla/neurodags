# cocofeats

A slurm friendly MEEG feature extraction package leveraging bids-like data organization and DAG processing.

- Agnostic to data organization
- Slurm friendly
- Reusage of existing derivatives through DAG processing.
- Yaml-based configuration
- Extensible without needing to fork the repository  

## Basic Example

Let's say you are exploring 


## Quickstart


## For developers

```bash
# Clone the repository

pip install -e .[dev,test,docs]

# Run quality checks
ruff check src/ (fix with `ruff check src/ --fix` if needed)
black --check . (fix with `black .` if needed)
pytest -q

# To debug pytest, use:
pytest -q --pdb
pytest -s -q --no-cov --pdb

# Build docs
sphinx-build -b html docs docs/_build/html -W --keep-going

# Clean docs
sphinx-build -M clean docs docs/_build/html

or

rm -rf docs/_build

```


# HDF5 and NetCDF4

If you get an error like:

```bash
  File "src/netCDF4/_netCDF4.pyx", line 5645, in netCDF4._netCDF4.Variable.__setitem__
  File "src/netCDF4/_netCDF4.pyx", line 5961, in netCDF4._netCDF4.Variable._put
  File "src/netCDF4/_netCDF4.pyx", line 2160, in netCDF4._netCDF4._ensure_nc_success
RuntimeError: NetCDF: HDF error
```

You may need to install hdf5 in your system and built from source:

```bash
pip install --no-binary=h5py h5py
```

## Documentation

- [Docs](https://yjmantilla.github.io/cocofeats/)

## Custom node definitions

Pipelines can import additional node definitions before registering features by pointing `new_definitions` to one or more Python files:

```yaml
datasets: example_pipelines/datasets_epilepsy.yml
mount_point: local
new_definitions:
  - custom_nodes/artifacts.py
  - /abs/path/to/local_nodes.py
FeatureDefinitions:
  MyCustomFeature:
    nodes:
      - id: 0
        node: my_custom_node  # registered inside the imported modules
```

Relative paths are resolved from the pipeline YAML location. Each module is executed once and may call `@register_node` as part of its import.

## Parallel execution

`iterate_feature_pipeline` can fan out across files using joblib. You can enable it either by passing `n_jobs` (and optionally `joblib_backend` / `joblib_prefer`) when calling the orchestrator or by adding the keys to your pipeline YAML:

```yaml
n_jobs: 4           # -1 to use all cores, 1 or null keeps it serial
joblib_backend: loky
joblib_prefer: processes
```

The CLI mirrors these options via `--n-jobs`, `--joblib-backend`, and `--joblib-prefer`.

## Visualization

You can visualize `.fif` or `.nc` files using the built-in visualization tool:

```bash
python -m cocofeats.visualization path/to/your_file.fif
```

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md).

## License

MIT. See [`LICENSE`](LICENSE).
