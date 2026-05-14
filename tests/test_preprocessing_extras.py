"""Extra tests for preprocessing nodes — branches not covered by test_preprocessing_nodes.py."""
from __future__ import annotations

import mne
import pytest

from neurodags.definitions import NodeResult
from neurodags.nodes.preprocessing import basic_preprocessing, keep_channels


# ---------------------------------------------------------------------------
# keep_channels — save=True
# ---------------------------------------------------------------------------

def test_keep_channels_save_true_sets_writer(dummy_raw_obj):
    raw, _ = dummy_raw_obj
    ch_names = raw.ch_names[:2]
    result = keep_channels(raw.copy(), channel_names=ch_names, save=True)
    assert result.artifacts[".fif"].writer is not None


def test_keep_channels_save_false_writer_is_none(dummy_raw_obj):
    raw, _ = dummy_raw_obj
    ch_names = raw.ch_names[:2]
    result = keep_channels(raw.copy(), channel_names=ch_names, save=False)
    assert result.artifacts[".fif"].writer is None


# ---------------------------------------------------------------------------
# basic_preprocessing — notch_filter
# ---------------------------------------------------------------------------

def test_basic_preprocessing_notch_filter(dummy_raw_obj):
    raw, _ = dummy_raw_obj
    result = basic_preprocessing(raw.copy(), notch_filter={"freqs": [20.0]})
    assert isinstance(result, NodeResult)
    assert ".fif" in result.artifacts


# ---------------------------------------------------------------------------
# basic_preprocessing — epoch_config="Events"
# ---------------------------------------------------------------------------

def test_basic_preprocessing_epoch_events(dummy_raw_obj):
    raw, events = dummy_raw_obj
    raw_copy = raw.copy()
    annotations = mne.annotations_from_events(
        events, sfreq=raw_copy.info["sfreq"], event_desc={1: "stim"}
    )
    raw_copy.set_annotations(annotations)
    result = basic_preprocessing(raw_copy, epoch_config="Events")
    assert isinstance(result, NodeResult)
    out = result.artifacts[".fif"].item
    assert isinstance(out, mne.BaseEpochs)


def test_basic_preprocessing_epoch_events_no_annotations_raises(dummy_raw_obj):
    raw, _ = dummy_raw_obj
    raw_copy = raw.copy()
    raw_copy.set_annotations(mne.Annotations([], [], []))
    with pytest.raises(ValueError, match="No annotations"):
        basic_preprocessing(raw_copy, epoch_config="Events")


def test_basic_preprocessing_unknown_epoch_config_raises(dummy_raw_obj):
    raw, _ = dummy_raw_obj
    with pytest.raises(ValueError, match="Unknown epoch_config"):
        basic_preprocessing(raw.copy(), epoch_config="UnknownConfig")


# ---------------------------------------------------------------------------
# basic_preprocessing — extra_artifacts
# ---------------------------------------------------------------------------

def test_basic_preprocessing_extra_artifacts_produces_report(dummy_raw_obj):
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    raw, _ = dummy_raw_obj
    result = basic_preprocessing(raw.copy(), extra_artifacts=True)
    assert ".report.html" in result.artifacts


def test_basic_preprocessing_extra_artifacts_with_filter(dummy_raw_obj):
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    raw, _ = dummy_raw_obj
    result = basic_preprocessing(
        raw.copy(),
        filter_args={"l_freq": 1.0, "h_freq": 40.0},
        extra_artifacts=True,
    )
    assert ".report.html" in result.artifacts


def test_basic_preprocessing_extra_artifacts_with_notch(dummy_raw_obj):
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    raw, _ = dummy_raw_obj
    result = basic_preprocessing(
        raw.copy(),
        notch_filter={"freqs": [20.0]},
        extra_artifacts=True,
    )
    assert ".report.html" in result.artifacts


def test_basic_preprocessing_extra_artifacts_with_epoch(dummy_raw_obj):
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    raw, _ = dummy_raw_obj
    result = basic_preprocessing(
        raw.copy(),
        epoch_config={"duration": 2.0, "overlap": 0.0},
        extra_artifacts=True,
    )
    assert ".report.html" in result.artifacts


def test_basic_preprocessing_extra_artifacts_with_resample(dummy_raw_obj):
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    raw, _ = dummy_raw_obj
    result = basic_preprocessing(
        raw.copy(),
        resample=50,
        extra_artifacts=True,
    )
    assert ".report.html" in result.artifacts
