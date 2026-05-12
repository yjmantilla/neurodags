import json
import os

import mne

from neurodags.definitions import Artifact, NodeResult
from neurodags.loaders import load_meeg
from neurodags.loggers import get_logger

from . import register_node

log = get_logger(__name__)

@register_node
def keep_channels(mne_object, channel_names, save=False) -> NodeResult:
    if isinstance(mne_object, (str, os.PathLike)):
        mne_object = load_meeg(mne_object)
        log.debug("keep_channels: loaded MNE object from file", input=mne_object)

    mne_object = mne_object.copy().pick(channel_names)

    if save:
        def writer(path):
            return mne_object.save(path, overwrite=True)
    else:
        writer = None

    artifacts = {
        ".fif": Artifact(item=mne_object, writer=writer)
    }

    out = NodeResult(artifacts=artifacts)

    return out

@register_node
def basic_preprocessing(
    mne_object,
    resample=None,
    filter_args=None,
    epoch_config=None,
    notch_filter=None,
    extra_artifacts: bool = False,
) -> NodeResult:

    if isinstance(mne_object, NodeResult):
        if ".fif" in mne_object.artifacts:
            mne_object = mne_object.artifacts[".fif"].item
        else:
            raise ValueError("NodeResult does not contain a .fif artifact to process.")

    if isinstance(mne_object, str | os.PathLike):
        mne_object = load_meeg(mne_object)
        log.debug("MNEReport: loaded MNE object from file", input=mne_object)


    mne_object = mne_object.copy()

    report = mne.Report(title="Basic Preprocessing", verbose="error")

    # EEG before preprocessing:
    if extra_artifacts:
        report.add_figure(mne_object.plot(show=False), title="Raw Data Before Preprocessing")
        log.debug("MNEReport: added raw data figure")
        # psd
        report.add_figure(
            mne_object.compute_psd().plot(show=False),
            title="Power Spectral Density Before Preprocessing",
        )
        log.debug("MNEReport: added raw PSD figure")
    # Filter the data
    if notch_filter is not None:
        mne_object = mne_object.notch_filter(**notch_filter, verbose=False)
        log.info("Notch Filter Applied", notch_filter=notch_filter)
        if extra_artifacts:
            report.add_figure(
                mne_object.compute_psd().plot(show=False),
                title="Power Spectral Density After Notch Filtering",
            )
            log.debug("MNEReport: added PSD after notch filtering figure")

    if filter_args is not None:
        mne_object = mne_object.filter(**filter_args, verbose=False)
        log.info("Filter Applied", filter_args=filter_args)
        if extra_artifacts:
            report.add_figure(
                mne_object.compute_psd().plot(show=False),
                title="Power Spectral Density After Filtering",
            )
            log.debug("MNEReport: added PSD after filtering figure")

    # Extract epochs
    if epoch_config is not None:
        if isinstance(epoch_config, dict):
            mne_object = mne.make_fixed_length_epochs(mne_object, preload=True, **epoch_config)
            log.info("EPOCH SEGMENTATION with make_fixed_length_epochs", epoch_config=epoch_config)
        elif isinstance(epoch_config, str):
            if epoch_config == "SingleEpoch":
                mne_object = mne.make_fixed_length_epochs(
                    mne_object, preload=True, duration=mne_object.times[-1], overlap=0
                )
                log.info("EPOCH SEGMENTATION with SingleEpoch")
            elif epoch_config == "Events":
                if not hasattr(mne_object, "annotations") or len(mne_object.annotations) == 0:
                    raise ValueError("No annotations found in the MNE object for epoching.")
                events, event_id = mne.events_from_annotations(mne_object)
                mne_object = mne.Epochs(mne_object, events=events, event_id=event_id, preload=True)
                log.info("EPOCH SEGMENTATION with Events", event_id=event_id)
            else:
                raise ValueError(f"Unknown epoch_config: {epoch_config}")

        if extra_artifacts:
            report.add_figure(mne_object.plot(show=False), title="Epochs After Segmentation")
            log.debug("MNEReport: added epochs after segmentation figure")

    if resample:
        mne_object = mne_object.resample(resample, verbose=False)
        log.info("Resample Applied", resample=resample)
        if extra_artifacts:
            report.add_figure(
                mne_object.compute_psd().plot(show=False),
                title="Power Spectral Density After Resampling",
            )
            log.debug("MNEReport: added PSD after resampling figure")

    this_metadata = {
        "resample": resample,
        "filter_args": filter_args,
        "epoch_config": epoch_config,
        "notch_filter": notch_filter,
    }

    # Also add metadata to the report
    if extra_artifacts:
        report.add_html(
            f"<pre>{json.dumps(this_metadata, indent=2)}</pre>",
            title="Metadata",
            section="Metadata",
        )

    artifacts = {
        ".fif": Artifact(item=mne_object, writer=lambda path: mne_object.save(path, overwrite=True))
    }

    if extra_artifacts:
        artifacts[".report.html"] = Artifact(
            item=report, writer=lambda path: report.save(path, overwrite=True, open_browser=False)
        )

    out = NodeResult(artifacts=artifacts)

    return out
