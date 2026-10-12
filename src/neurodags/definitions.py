import os
from collections.abc import Callable, Mapping
from typing import IO, Any, NamedTuple

from pydantic import BaseModel, PrivateAttr

PathLike = str | os.PathLike[str]
RulesLike = Mapping[str, Any] | PathLike | IO[str]


class DatasetConfig(BaseModel):
    """Configuration for a dataset.

    Relative path fields (``source_root``, ``file_pattern``, ``exclude_pattern``,
    ``derivatives_path``) are resolved, after ``~``/``$VAR`` expansion, against the
    directory of the YAML file that defined the dataset, or the ``base_dir=`` passed to
    the Python API for an in-memory mapping (see
    :func:`neurodags.utils.resolve_dataset_field`). That directory is kept in the private
    ``_base_dir`` attribute, set by
    :func:`neurodags.datasets.get_datasets_and_mount_point_from_pipeline_configuration`.
    """

    name: str
    # Root of the source tree (string or mount map). Required since 0.4.0; it is
    # declared optional here so that ``neurodags validate`` can report a missing value
    # with a suggestion instead of a pydantic error (checked by resolve_dataset_paths).
    source_root: str | dict[str, str] | None = None
    file_pattern: str | dict[str, str]  # path or mountpoint mapping
    exclude_pattern: str | None = None
    drop_split_continuations: bool = True  # drop split-FIF continuation files (keep entry only)
    skip: bool = False
    derivatives_path: str | dict[str, str] | None = None
    # Optional subfolder of derivatives_path for this dataset; required to differ
    # between datasets sharing one derivatives_path (spec §4.4).
    derivatives_subdir: str | None = None
    vars: dict[str, Any] | None = (
        None  # dataset-level variables; referenced as $var_name in pipeline node args
    )

    # Directory that relative path fields resolve against (None: absolute paths only).
    _base_dir: str | None = PrivateAttr(default=None)

    class Config:
        extra = "allow"  # allow arbitrary extra fields for user flexibility


class Artifact(NamedTuple):
    """An artifact produced by a node, with its associated writer."""

    item: Any
    writer: Callable[[str], None]  # how to save it


class NodeResult(NamedTuple):
    """The result of a node execution."""

    artifacts: dict[str, Artifact]  # Objects with writers


class SkipDerivative(Exception):
    """Raise from a node to signal that this source file is not applicable for this derivative.

    neurodags catches this exception, writes a ``.skip`` marker file next to where the
    artifact would have been saved, and reports the derivative as **skipped** in
    ``neurodags status`` output — distinct from *missing* (never ran) or *errored*
    (failed unexpectedly).

    Skipped derivatives are not retried on subsequent runs unless the ``.skip`` marker
    is deleted manually or ``overwrite: true`` is set on the derivative.

    Typical use-case: a condition that does not exist in a particular subject's recording.
    Instead of raising a generic ``ValueError`` (which neurodags treats as an error),
    raise ``SkipDerivative`` to record that the absence is intentional::

        from neurodags.definitions import SkipDerivative

        if condition_name not in found_conditions:
            raise SkipDerivative(
                f"Condition '{condition_name}' not present in this recording."
            )

    The message is written verbatim into the ``.skip`` marker file for later inspection.
    """


# In general we want at least one artifact, which should be
# an xarray DataArray with dimensions and coordinates fully populated
# and optional metadata in attrs (json-serializable)
