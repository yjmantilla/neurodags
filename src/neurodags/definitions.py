import os
from collections.abc import Callable, Mapping
from typing import IO, Any, NamedTuple

from pydantic import BaseModel

PathLike = str | os.PathLike[str]
RulesLike = Mapping[str, Any] | PathLike | IO[str]


class DatasetConfig(BaseModel):
    """Configuration for a dataset."""
    name: str
    file_pattern: str | dict[str, str]  # path or mountpoint mapping
    exclude_pattern: str | None = None
    skip: bool = False
    derivatives_path: str | dict[str, str] | None = None

    class Config:
        extra = "allow"  # allow arbitrary extra fields for user flexibility


class Artifact(NamedTuple):
    """An artifact produced by a node, with its associated writer."""
    item: Any
    writer: Callable[[str], None]  # how to save it


class NodeResult(NamedTuple):
    """The result of a node execution."""
    artifacts: dict[str, Artifact]  # Objects with writers


# In general we want at least one artifact, which should be
# an xarray DataArray with dimensions and coordinates fully populated
# and optional metadata in attrs (json-serializable)
