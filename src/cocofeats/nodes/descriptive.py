import json
import os
import mne
import xarray as xr

from cocofeats.definitions import Artifact, NodeResult
from cocofeats.loaders import load_meeg
from cocofeats.loggers import get_logger
from . import register_node

from cocofeats.writers import _json_safe

log = get_logger(__name__)


def _format_meas_date(meas_date):
    """Return a human-readable measurement date."""
    if meas_date is None:
        return None

    try:
        if hasattr(meas_date, "isoformat"):
            return meas_date.isoformat()
        if isinstance(meas_date, tuple) and len(meas_date) == 2:
            seconds, microseconds = meas_date
            return f"{seconds + microseconds * 1e-6:.6f}"
    except Exception:  # pragma: no cover - best effort formatting
        pass

    try:
        return str(meas_date)
    except Exception:  # pragma: no cover - best effort formatting
        return None


def _build_metadata(mne_object, *, kind: str, extra: dict | None = None) -> dict:
    """Collect lightweight metadata for xarray attributes."""

    info = mne_object.info
    metadata: dict[str, object] = {"kind": kind}

    sfreq = info.get("sfreq")
    if sfreq is not None:
        metadata["sfreq"] = float(sfreq)

    ch_names = list(info.get("ch_names", []))
    metadata["n_channels"] = len(ch_names)
    metadata["channel_names"] = ch_names

    channel_types = mne_object.get_channel_types()
    if channel_types:
        metadata["channel_types"] = channel_types

    bads = list(info.get("bads", []))
    if bads:
        metadata["bad_channels"] = bads

    for key in ("line_freq", "lowpass", "highpass"):
        value = info.get(key)
        if value is not None:
            metadata[key] = float(value)

    meas_date_str = _format_meas_date(info.get("meas_date"))
    if meas_date_str:
        metadata["meas_date"] = meas_date_str

    description = info.get("description")
    if description:
        metadata["description"] = description

    if extra:
        metadata.update(extra)

    return metadata


@register_node
def extract_meeg_metadata(mne_object) -> NodeResult:
    """
    Extract metadata from an MNE object (Raw or Epochs) and save as JSON.

    Parameters
    ----------
    mne_object : str | os.PathLike | mne.io.Raw | mne.BaseEpochs
        Path to a MEEG file or an already loaded MNE object.

    Returns
    -------
    NodeResult
        A feature result containing a JSON artifact with metadata.
    """

    if isinstance(mne_object, NodeResult):
        if ".fif" in mne_object.artifacts:
            mne_object = mne_object.artifacts[".fif"].item
        else:
            raise ValueError("NodeResult does not contain a .fif artifact to process.")

    if isinstance(mne_object, (str, os.PathLike)):
        mne_object = load_meeg(mne_object)
        log.debug("Loaded MNE object from file", input=mne_object)

    info_dict = {}

    # Shared info
    info_dict["sfreq"] = float(mne_object.info.get("sfreq", None))
    info_dict["n_channels"] = mne_object.info.get("nchan", None)
    info_dict["ch_names"] = list(mne_object.info.get("ch_names", []))

    # Channel types
    try:
        info_dict["ch_types"] = mne_object.get_channel_types()
    except Exception:
        info_dict["ch_types"] = None

    # Dimensions like xarray
    dims = []
    coords = {}
    shape = ()

    if isinstance(mne_object, mne.io.BaseRaw):
        dims = ["times", "spaces"]
        n_times = mne_object.n_times
        shape = (n_times, len(info_dict["ch_names"]))
        coords["times"] = {'start': float(mne_object.times[0]), 'stop': float(mne_object.times[-1]), 'n_times': n_times, 'delta': float(mne_object.times[1] - mne_object.times[0])}
        coords["spaces"] = info_dict["ch_names"]

    elif isinstance(mne_object, mne.BaseEpochs):
        dims = ["epochs", "times", "spaces"]
        n_epochs, n_channels, n_times = mne_object.get_data().shape
        shape = (n_epochs, n_times, n_channels)
        coords["epochs"] = list(range(n_epochs))
        coords["times"] = {'start': float(mne_object.times[0]), 'stop': float(mne_object.times[-1]), 'n_times': n_times, 'delta': float(mne_object.times[1] - mne_object.times[0])}
        coords["spaces"] = info_dict["ch_names"]

        # Add event info if available
        if hasattr(mne_object, "events"):
            info_dict["events_shape"] = (
                None if mne_object.events is None else mne_object.events.shape
            )

    info_dict["shape"] = shape
    info_dict["dims"] = dims
    info_dict["coords"] = coords

    # Optional: add annotations if present
    if hasattr(mne_object, "annotations") and len(mne_object.annotations) > 0:
        info_dict["annotations"] = [
            {"onset": float(ann["onset"]), "duration": float(ann["duration"]), "description": ann["description"]}
            for ann in mne_object.annotations
        ]

    # Prepare artifact: human-readable JSON
    def _writer(path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(_json_safe(info_dict), f, indent=2, ensure_ascii=False)

    artifacts = {".json": Artifact(item=info_dict, writer=_writer)}

    # also present a mne.report with the info?

    return NodeResult(artifacts=artifacts)


@register_node
def meeg_to_xarray(mne_object) -> NodeResult:
    """Convert an MNE Raw/Epochs object to an xarray DataArray artifact."""

    if isinstance(mne_object, NodeResult):
        if ".fif" in mne_object.artifacts:
            mne_object = mne_object.artifacts[".fif"].item
        else:
            raise ValueError("NodeResult does not contain a .fif artifact to process.")

    if isinstance(mne_object, (str, os.PathLike)):
        mne_object = load_meeg(mne_object)
        log.debug("Loaded MNE object from file", input=mne_object)

    if isinstance(mne_object, mne.io.BaseRaw):
        data = mne_object.get_data()
        ch_names = list(mne_object.info.get("ch_names", []))
        coords = {
            "spaces": ("spaces", ch_names),
            "times": ("times", mne_object.times.copy()),
        }
        channel_types = mne_object.get_channel_types()
        if channel_types:
            coords["channel_type"] = ("spaces", channel_types)

        dims = ("spaces", "times")
        metadata_payload = _build_metadata(
            mne_object,
            kind="raw",
            extra={
                "n_times": data.shape[1],
                "time_start": float(mne_object.times[0]) if data.shape[1] else None,
                "time_stop": float(mne_object.times[-1]) if data.shape[1] else None,
            },
        )

    elif isinstance(mne_object, mne.BaseEpochs):
        data = mne_object.get_data()
        ch_names = list(mne_object.ch_names)
        coords = {
            "epochs": ("epochs", list(range(data.shape[0]))),
            "spaces": ("spaces", ch_names),
            "times": ("times", mne_object.times.copy()),
        }
        channel_types = mne_object.get_channel_types()
        if channel_types:
            coords["channel_type"] = ("spaces", channel_types)

        if getattr(mne_object, "events", None) is not None:
            events = mne_object.events
            coords["event_id"] = ("epochs", events[:, 2].tolist())
            coords["event_sample"] = ("epochs", events[:, 0].tolist())
            id_to_name = {val: key for key, val in mne_object.event_id.items()}
            coords["event_name"] = (
                "epochs",
                [id_to_name.get(event, str(event)) for event in events[:, 2]],
            )

        dims = ("epochs", "spaces", "times")
        metadata_payload = _build_metadata(
            mne_object,
            kind="epochs",
            extra={
                "n_epochs": data.shape[0],
                "n_times": data.shape[-1],
                "time_start": float(mne_object.times[0]) if data.shape[-1] else None,
                "time_stop": float(mne_object.times[-1]) if data.shape[-1] else None,
            },
        )

        if getattr(mne_object, "metadata", None) is not None:
            metadata_payload["epoch_metadata"] = mne_object.metadata.to_dict(orient="list")

    else:
        raise TypeError(
            "Input must be a path to a MEEG file, an MNE Raw, or an MNE Epochs object."
        )

    xarray_data = xr.DataArray(data, dims=dims, coords=coords)
    xarray_data.attrs["kind"] = metadata_payload.get("kind")
    xarray_data.attrs["time_unit"] = "s"

    for attr_name in ("sfreq", "n_channels", "n_times", "n_epochs", "line_freq", "lowpass", "highpass"):
        value = metadata_payload.get(attr_name)
        if value is not None:
            xarray_data.attrs[attr_name] = value

    xarray_data.attrs["metadata"] = json.dumps(
        _json_safe(metadata_payload),
        indent=2,
        ensure_ascii=False,
    )

    artifacts = {
        ".nc": Artifact(item=xarray_data, writer=lambda path: xarray_data.to_netcdf(path)),
    }

    return NodeResult(artifacts=artifacts)
