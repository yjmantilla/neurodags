"""Run the shipped example pipelines end-to-end so they cannot silently rot."""

from __future__ import annotations

from pathlib import Path

import mne
import pytest

from neurodags.datasets import save_dummy_vhdr
from neurodags.orchestrators import run_pipeline

EXAMPLES = Path(__file__).resolve().parents[1] / "example_pipelines"


def test_quickstart_preprocessing_template(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "out"
    save_dummy_vhdr(raw_dir / "sub-01" / "sub-01_task-rest_eeg.vhdr")
    monkeypatch.setenv("RAW_DATA", str(raw_dir))
    monkeypatch.setenv("OUTPUT_DIR", str(out_dir))

    run_pipeline(
        str(EXAMPLES / "pipeline_quickstart_preprocessing.yml"),
        raise_on_error=True,
    )

    stem = out_dir / "derivatives" / "sub-01" / "sub-01_task-rest_eeg.vhdr@Preprocessed"
    assert Path(f"{stem}.report.html").is_file()
    epochs = mne.read_epochs(f"{stem}.fif", verbose="error")
    assert epochs.info["sfreq"] == 256.0
    assert epochs.tmax - epochs.tmin < 2.0  # 2 s windows
    assert epochs.info["highpass"] == pytest.approx(0.1)
    assert epochs.info["lowpass"] == pytest.approx(75.0)
