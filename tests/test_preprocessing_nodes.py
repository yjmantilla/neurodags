"""Tests for preprocessing nodes using real MNE objects."""
from __future__ import annotations

import mne
import pytest

from neurodags.definitions import Artifact, NodeResult
from neurodags.nodes.preprocessing import basic_preprocessing, keep_channels


# ---------------------------------------------------------------------------
# basic_preprocessing
# ---------------------------------------------------------------------------

def test_basic_preprocessing_returns_noderesulet(dummy_raw_obj):
    raw, _ = dummy_raw_obj
    result = basic_preprocessing(raw.copy())
    assert isinstance(result, NodeResult)
    assert ".fif" in result.artifacts


def test_basic_preprocessing_filter_changes_spectrum(dummy_raw_obj):
    raw, _ = dummy_raw_obj
    result = basic_preprocessing(raw.copy(), filter_args={"l_freq": 1.0, "h_freq": 40.0})
    out = result.artifacts[".fif"].item
    assert isinstance(out, mne.io.BaseRaw)
    assert out.info["sfreq"] == raw.info["sfreq"]


def test_basic_preprocessing_resample_changes_sfreq(dummy_raw_obj):
    raw, _ = dummy_raw_obj
    result = basic_preprocessing(raw.copy(), resample=50)
    out = result.artifacts[".fif"].item
    assert out.info["sfreq"] == 50.0


def test_basic_preprocessing_epoch_returns_epochs(dummy_raw_obj):
    raw, _ = dummy_raw_obj
    result = basic_preprocessing(raw.copy(), epoch_config={"duration": 2.0, "overlap": 0.0})
    out = result.artifacts[".fif"].item
    assert isinstance(out, mne.BaseEpochs)


def test_basic_preprocessing_epoch_single_epoch(dummy_raw_obj):
    raw, _ = dummy_raw_obj
    result = basic_preprocessing(raw.copy(), epoch_config="SingleEpoch")
    out = result.artifacts[".fif"].item
    assert isinstance(out, mne.BaseEpochs)
    assert len(out) == 1


def test_basic_preprocessing_accepts_noderesulet_input(dummy_raw_obj):
    raw, _ = dummy_raw_obj
    nr = NodeResult(
        artifacts={".fif": Artifact(item=raw.copy(), writer=lambda path: None)}
    )
    result = basic_preprocessing(nr)
    assert isinstance(result, NodeResult)
    assert ".fif" in result.artifacts


def test_basic_preprocessing_noderesulet_missing_fif_raises(dummy_raw_obj):
    with pytest.raises(ValueError, match=".fif"):
        basic_preprocessing(NodeResult(artifacts={".nc": Artifact(item=None, writer=lambda p: None)}))


def test_basic_preprocessing_from_path(dummy_vhdr_file):
    result = basic_preprocessing(dummy_vhdr_file)
    assert isinstance(result, NodeResult)
    assert ".fif" in result.artifacts


def test_basic_preprocessing_all_options_combined(dummy_raw_obj):
    raw, _ = dummy_raw_obj
    result = basic_preprocessing(
        raw.copy(),
        filter_args={"l_freq": 1.0, "h_freq": 40.0},
        epoch_config={"duration": 2.0, "overlap": 0.0},
        resample=50,
    )
    out = result.artifacts[".fif"].item
    assert isinstance(out, mne.BaseEpochs)
    assert out.info["sfreq"] == 50.0


# ---------------------------------------------------------------------------
# keep_channels
# ---------------------------------------------------------------------------

def test_keep_channels_reduces_count(dummy_raw_obj):
    raw, _ = dummy_raw_obj
    ch_names = raw.ch_names[:2]
    result = keep_channels(raw.copy(), channel_names=ch_names)
    out = result.artifacts[".fif"].item
    assert list(out.ch_names) == ch_names


def test_keep_channels_from_path(dummy_vhdr_file):
    import mne as _mne
    raw_info = _mne.io.read_raw(str(dummy_vhdr_file), preload=False, verbose="error")
    ch_names = raw_info.ch_names[:2]
    result = keep_channels(dummy_vhdr_file, channel_names=ch_names)
    out = result.artifacts[".fif"].item
    assert list(out.ch_names) == ch_names
