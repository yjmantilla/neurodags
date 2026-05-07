"""Shared pytest fixtures for neurodags tests."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from neurodags.datasets import generate_dummy_dataset, get_dummy_epochs, get_dummy_raw


# ---------------------------------------------------------------------------
# MNE object fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def dummy_raw_obj():
    """Minimal MNE Raw (4 ch, 100 Hz, 10 s, pink noise)."""
    raw, events = get_dummy_raw(NCHANNELS=4, SFREQ=100.0, STOP=10.0, NUMEVENTS=5, random_state=0)
    return raw, events


@pytest.fixture(scope="session")
def dummy_epochs_obj():
    """Minimal MNE Epochs derived from the same dummy raw."""
    return get_dummy_epochs(
        NCHANNELS=4, SFREQ=100.0, STOP=10.0, NUMEVENTS=5,
        tmin=0.0, tmax=0.9, random_state=0, baseline=None,
    )


# ---------------------------------------------------------------------------
# On-disk vhdr fixture (one file, scoped to function via tmp_path)
# ---------------------------------------------------------------------------

@pytest.fixture()
def dummy_vhdr_file(tmp_path: Path) -> Path:
    """Write a single dummy BrainVision trio and return the .vhdr path."""
    from neurodags.datasets import save_dummy_vhdr

    vhdr = tmp_path / "sub-01" / "sub-01_task-rest.vhdr"
    trio = save_dummy_vhdr(vhdr, dummy_args={"NCHANNELS": 4, "SFREQ": 100.0, "STOP": 10.0, "NUMEVENTS": 5, "random_state": 0})
    assert trio is not None, "dummy vhdr creation failed"
    return vhdr


# ---------------------------------------------------------------------------
# Full pipeline fixture — generates a dataset on disk + pipeline config dict
# ---------------------------------------------------------------------------

_GENERATION_ARGS: dict[str, Any] = {
    "NCHANNELS": 4,
    "SFREQ": 100.0,
    "STOP": 10.0,
    "NUMEVENTS": 5,
    "random_state": 42,
}


@pytest.fixture()
def dummy_pipeline(tmp_path: Path) -> dict[str, Any]:
    """
    Generate a tiny on-disk dataset (2 subjects × 1 session) and return
    a ready-to-use pipeline configuration dict.

    Returns
    -------
    dict with keys:
        config   - pipeline config dict accepted by iterate_derivative_pipeline
        data_dir - Path where source vhdr files were generated
        out_dir  - Path where derivatives will be written
    """
    data_dir = tmp_path / "rawdata"
    out_dir = tmp_path / "derivatives"
    out_dir.mkdir(parents=True, exist_ok=True)

    generate_dummy_dataset(
        data_params={
            "DATASET": "test",
            "PATTERN": "sub-%subject%/ses-%session%/sub-%subject%_ses-%session%_task-rest",
            "NSUBS": 2,
            "NSESSIONS": 1,
            "NTASKS": 1,
            "NACQS": 1,
            "NRUNS": 1,
            "PREFIXES": {"subject": "S", "session": "SE", "task": "T", "acquisition": "A", "run": "R"},
            "ROOT": str(data_dir),
        },
        generation_args=_GENERATION_ARGS,
    )

    datasets = {
        "test_dataset": {
            "name": "TestDataset",
            "file_pattern": str(data_dir / "**" / "*.vhdr"),
            "derivatives_path": str(out_dir),
        }
    }

    config = {
        "datasets": datasets,
        "mount_point": None,
        "DerivativeDefinitions": {
            "BasicPrep": {
                "overwrite": False,
                "nodes": [
                    {"id": 0, "derivative": "SourceFile"},
                    {
                        "id": 1,
                        "node": "basic_preprocessing",
                        "args": {
                            "mne_object": "id.0",
                            "filter_args": {"l_freq": 1.0, "h_freq": 40.0},
                            "epoch_config": {"duration": 2.0, "overlap": 0.0},
                        },
                    },
                ],
            },
            "Spectrum": {
                "overwrite": False,
                "nodes": [
                    {"id": 0, "derivative": "BasicPrep.fif"},
                    {
                        "id": 1,
                        "node": "mne_spectrum_array",
                        "args": {
                            "meeg": "id.0",
                            "method": "welch",
                            "method_kwargs": {"n_per_seg": 100},
                        },
                    },
                ],
            },
            "BandPowerMean": {
                "save": False,
                "for_dataframe": True,
                "nodes": [
                    {"id": 0, "derivative": "Spectrum.nc"},
                    {
                        "id": 1,
                        "node": "extract_data_var",
                        "args": {"dataset_like": "id.0", "data_var": "spectrum"},
                    },
                    {
                        "id": 2,
                        "node": "bandpower",
                        "args": {
                            "psd_like": "id.1",
                            "bands": {"alpha": [8.0, 13.0], "beta": [13.0, 30.0]},
                            "relative": True,
                        },
                    },
                    {
                        "id": 3,
                        "node": "aggregate_across_dimension",
                        "args": {"xarray_data": "id.2", "dim": "epochs", "operation": "mean"},
                    },
                ],
            },
        },
        "DerivativeList": ["BasicPrep", "Spectrum", "BandPowerMean"],
    }

    return {"config": config, "data_dir": data_dir, "out_dir": out_dir}
