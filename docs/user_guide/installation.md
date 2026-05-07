# Installation

## Stable Release

```bash
pip install neurodags
```

## Development Install

```bash
git clone https://github.com/yjmantilla/neurodags
cd neurodags
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -e .[dev,test,docs]
pre-commit install
```

## HDF5 / NetCDF Note

If you encounter this error when writing `.nc` files:

```
RuntimeError: NetCDF: HDF error
```

Rebuild h5py against your system HDF5:

```bash
pip install --no-binary=h5py h5py
```

## Requirements

- Python >= 3.10
- Core dependencies: `mne`, `xarray`, `netCDF4`, `pydantic`, `networkx`, `joblib`, `plotly`, `dash`, `structlog`, `pyyaml`, `fooof`, `antropy`, `neurokit2`
