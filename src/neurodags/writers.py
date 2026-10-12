import json
import os
import time
from typing import Any

import numpy as np

try:
    import xarray as xr
except ImportError:
    xr = None

from neurodags.loggers import get_logger

log = get_logger(__name__)


def save_dict_to_json(jsonfile, data):
    with open(jsonfile, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    log.debug("Saved JSON file", file=jsonfile)


def _json_safe(value: Any) -> Any:
    """Return a JSON-serialisable representation of `value`.

    Handles:
    - Python primitives (str, int, float, bool, None)
    - dict (keys coerced to str)
    - list, tuple, set (converted to lists)
    - numpy arrays and scalars
    - xarray DataArray (if available)
    - Fallback to repr(value)
    """
    # JSON-native types
    if isinstance(value, str | int | float | bool) or value is None:
        return value

    # Dictionaries (ensure str keys for JSON)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}

    # Sequences
    if isinstance(value, list | tuple | set):
        return [_json_safe(v) for v in value]

    # NumPy arrays
    if isinstance(value, np.ndarray):
        return value.tolist()

    # NumPy scalars
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.generic):  # catch-all for other numpy scalars
        return value.item()

    # xarray DataArray
    if xr is not None and isinstance(value, xr.DataArray):
        return value.to_dict()

    # Fallback
    return repr(value)


def save_mne(mne_object: Any, path: str, **kwargs: Any) -> list[str]:
    """Save an MNE object (``Raw``, ``Epochs``, ...) and return every file written.

    Artifact writer for MNE objects following the writer contract: an ``Artifact.writer``
    returns the list of files it produced, so the ``.done`` manifest lists them all.

    MNE splits recordings larger than ``split_size`` (default 2 GB) into several files.
    With the default ``split_naming="neuromag"`` the parts are ``<name>.fif``,
    ``<name>-1.fif``, ``<name>-2.fif``, ... (``mne._fiff.utils._make_split_fnames``), and
    the first part stores the next part's file name, so they must stay together.

    Detection of the parts:

    - MNE's ``Raw.save``/``Epochs.save`` return the list of files they wrote (the
      ``_write_raw``/``_save_split`` bookkeeping): that list is used as is. This is the
      normal path with current MNE.
    - With an MNE whose ``save`` returns ``None``, the parts are reconstructed from the
      deterministic neuromag names: ``<name>-1.fif``, ``<name>-2.fif``, ... are added
      while they exist **and** were modified by this call (mtime not older than the save
      start), so stale parts left by an earlier, larger save are never picked up. No
      directory glob is used.

    Parameters
    ----------
    mne_object : mne object with a ``save`` method
        The object to save.
    path : str
        Target path (the artifact path ``<prefix><suffix>``).
    **kwargs
        Passed to ``mne_object.save`` (``overwrite=True`` is the default).

    Returns
    -------
    list of str
        The files written, the target (first part) first.
    """
    kwargs.setdefault("overwrite", True)
    started = time.time() - 1.0  # filesystem mtime resolution margin
    returned = mne_object.save(path, **kwargs)
    if isinstance(returned, list | tuple) and returned:
        return [os.fspath(p) for p in returned]
    files = [os.fspath(path)]
    if kwargs.get("split_naming", "neuromag") == "neuromag":
        base, ext = os.path.splitext(os.fspath(path))
        part = 1
        while True:
            candidate = f"{base}-{part}{ext}"
            if not os.path.isfile(candidate) or os.path.getmtime(candidate) < started:
                break
            files.append(candidate)
            part += 1
    return files
