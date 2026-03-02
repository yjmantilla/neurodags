import json

from neurodags.loggers import get_logger

log = get_logger(__name__)


def save_dict_to_json(jsonfile, data):
    with open(jsonfile, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    log.debug("Saved JSON file", file=jsonfile)


import json
import numpy as np
import xarray as xr

from typing import Any

from typing import Any
import numpy as np

try:
    import xarray as xr
except ImportError:
    xr = None


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
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    # Dictionaries (ensure str keys for JSON)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}

    # Sequences
    if isinstance(value, (list, tuple, set)):
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
