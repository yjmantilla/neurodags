"""Parser nodes: extract structured properties from a file path.

A *parser node* takes a file path and returns a **flat** dict of properties
(e.g. BIDS entities). Downstream nodes consume individual fields via
``id.N.field`` references in the pipeline YAML.

Contract for a parser node: return a flat, JSON-serializable dict whose keys are
**plain identifiers** (letters/digits/underscore, not starting with a digit).
The ``.`` in ``id.N.field`` is the reference separator, so dotted/nested keys
cannot be addressed — keep the dict flat. Values are unconstrained.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from neurodags.definitions import Artifact, NodeResult
from neurodags.loggers import get_logger

from . import register_node

log = get_logger(__name__)


@register_node(name="bids_parse", override=True)
def bids_parse(path):
    """Parse BIDS entities from a file path into a flat dict.

    Parameters
    ----------
    path : str | os.PathLike
        Source file path. Only the basename is parsed; the directory is ignored.

    Returns
    -------
    NodeResult
        A single ``.json`` artifact whose item is a flat dict of the BIDS
        entities present in the filename (``subject``, ``session``, ``task``,
        ``acquisition``, ``run``, ...). Entities absent from the name are
        omitted. All keys are plain identifiers, so downstream args can reference
        them as ``id.N.subject``, ``id.N.task``, etc. Non-BIDS names simply yield
        few/no entities rather than raising.
    """
    from mne_bids import get_entities_from_fname

    name = Path(os.fspath(path)).name
    entities = get_entities_from_fname(name, on_error="ignore")
    props = {key: value for key, value in entities.items() if value is not None}
    log.debug("bids_parse: parsed entities", path=name, fields=sorted(props))

    def _writer(out_path: str) -> None:
        Path(out_path).write_text(json.dumps(props, indent=2), encoding="utf-8")

    return NodeResult(artifacts={".json": Artifact(item=props, writer=_writer)})
