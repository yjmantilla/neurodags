from __future__ import annotations

import os
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any, Union

import mne
import yaml

from cocofeats.definitions import RulesLike
from cocofeats.loggers import get_logger

log = get_logger(__name__)

# Prefer C-accelerated safe loader if available.
try:
    _BaseSafeLoader = yaml.CSafeLoader  # type: ignore[attr-defined]
except AttributeError:
    _BaseSafeLoader = yaml.SafeLoader


class UniqueKeySafeLoader(_BaseSafeLoader):
    """
    Safe YAML loader that raises on duplicate keys within the same mapping.
    Logs the exact location (line/column).
    """

    def construct_mapping(self, node, deep: bool = False):  # type: ignore[override]
        if not isinstance(node, yaml.MappingNode):
            raise yaml.constructor.ConstructorError(
                None, None, f"Expected a mapping node, got {node.id}", node.start_mark
            )

        seen: set[Any] = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in seen:
                mark = key_node.start_mark
                log.error(
                    "Duplicate YAML key encountered",
                    key=key,
                    line=mark.line + 1,
                    column=mark.column + 1,
                )
                raise ValueError(
                    f"Duplicate key '{key}' at line {mark.line + 1}, column {mark.column + 1}"
                )
            seen.add(key)

        return super().construct_mapping(node, deep=deep)


def load_configuration(rules: RulesLike) -> dict[str, Any]:
    """
    Load a rules dictionary from:
      - a mapping (returned deep-copied),
      - a path (str/PathLike) to a YAML file, or
      - a text file-like object with YAML content.

    Features:
      - Safe loader (no object construction).
      - Raises on duplicate keys (with line/column) and logs them.
      - Ensures the document root is a mapping/dict.
      - Handles empty files (returns {}).

    Raises:
      IOError   : File cannot be read.
      ValueError: Invalid YAML or duplicate keys.
      TypeError : YAML root is not a mapping/dict.
    """
    log.debug("Loading YAML rules", arg_type=type(rules).__name__)

    # If a mapping was provided, return a defensive copy.
    if isinstance(rules, Mapping):
        log.debug("Loading YAML from in-memory mapping", action="deepcopy", keys=len(rules))
        return deepcopy(rules)

    # Open file if given a path; otherwise assume text file-like.
    close_after = False
    if isinstance(rules, str | os.PathLike):
        path = Path(rules)
        local_log = get_logger(__name__, path=str(path))
        local_log.debug("Opening YAML file")
        try:
            f = path.open("r", encoding="utf-8")
            close_after = True
        except OSError as e:
            local_log.exception("Failed to open YAML file")
            raise OSError(f"Couldn't read rules file: {path}") from e
        active_log = local_log
    else:
        f = rules  # type: ignore[assignment]
        active_log = log
        active_log.debug("Using provided file-like object")

    try:
        try:
            data: Any | None = yaml.load(f, Loader=UniqueKeySafeLoader)
            active_log.debug("YAML parsed successfully")
        except yaml.YAMLError as e:
            active_log.exception("Invalid YAML encountered during parsing")
            raise ValueError(f"Invalid YAML: {e}") from e

        if data is None:
            active_log.warning("YAML document is empty; returning an empty dictionary")
            return {}

        if not isinstance(data, dict):
            active_log.error("YAML root is not a mapping/dict", root_type=type(data).__name__)
            raise TypeError(f"YAML root must be a mapping/dict, got {type(data).__name__}")

        active_log.debug("YAML root is a mapping", top_level_keys=len(data))
        return data

    finally:
        if close_after:
            f.close()
            active_log.debug("Closed YAML file handle")


log = get_logger(__name__)

PathLike = Union[str, "os.PathLike[str]"]


def load_meeg(meeg_file: PathLike, kwargs: dict | None = None):
    """
    Load a MEEG file (Raw or Epochs) using MNE-Python.

    Parameters
    ----------
    meeg_file : PathLike
        Path to the MEEG file to be loaded.
    kwargs : dict, optional
        Additional keyword arguments passed to the MNE loader.
        Defaults to ``dict(preload=True, verbose='error')``.

    Returns
    -------
    mne.io.BaseRaw | mne.BaseEpochs
        Loaded MEEG object (either Raw or Epochs).

    Raises
    ------
    ValueError
        If the file cannot be loaded as Raw or Epochs.

    Notes
    -----
    - This function first attempts to load the file with
      :func:`mne.io.read_raw` (which automatically detects file types).
    - If that fails, it falls back to trying :func:`mne.read_epochs`.
    - If both attempts fail, a ValueError is raised.
    """
    if kwargs is None:
        kwargs = {"preload": True, "verbose": "error"}

    meeg_file = Path(meeg_file)
    log.debug("Attempting to load MEEG file", file=str(meeg_file), kwargs=kwargs)

    try:
        # Try to load as Raw
        meeg = mne.io.read_raw(meeg_file, **kwargs)
        log.info("Loaded MEEG file as Raw", file=str(meeg_file))
    except Exception as e_raw:
        #log.debug("Failed to load as Raw, trying Epochs", file=str(meeg_file), error=str(e_raw))
        try:
            # try to load as epochs
            meeg = mne.read_epochs(meeg_file, **kwargs)
            log.info("Loaded MEEG file as Epochs", file=str(meeg_file))
        except Exception as e_epochs:
            log.debug(
                "Failed to load as Epochs and as Raw",
                file=str(meeg_file),
                error=str(e_epochs),
            )
            raise ValueError(
                f"Could not load MEEG file {meeg_file}. Error: {e_epochs}"
            ) from e_epochs

    return meeg
