"""Upgrade a neurodags 0.3.x project to ``naming_version: 1``: ``neurodags migrate`` (spec 10).

One pass converts both the configuration and the existing output tree:

- **Configuration** (spec 10.2). The 0.3.x pipeline is loaded with 0.3.x semantics (dataset
  paths relative to the directory the old commands were run from, ``legacy_cwd``). The new
  configuration adds ``naming_version: 1`` and a ``source_root`` per dataset and mount,
  rewrites relative dataset paths so they resolve to the same places under the 0.4.0 rule
  (relative to the YAML that defines them) and applies the ``--map`` derivative renames.
  Files are edited in place (comments and ordering kept) through YAML node positions; when
  that is not possible the file is re-serialized and the preview says comments are lost.
  Backups ``<file>.bak-0.3`` are kept. Configuration files are written last, after every file
  move succeeded: a separate datasets YAML first, then the pipeline that declares
  ``naming_version: 1`` (a re-run after a crash in between accepts the migrated datasets).
- **Outputs** (spec 10.3). The new layout is enumerated (discovery with the new config,
  :func:`neurodags.outputs.output_reference_base`); the existing trees are *scanned* (never
  recomputed from the 0.3.x roots, which depended on which files were discovered). An old
  output ``<dir>/<source>@<Token><tail>`` (``Token`` = the 0.3.x
  :func:`neurodags._legacy.snake_to_camel` spelling of a YAML name) maps to the source whose
  *legacy key* (its basename with every ``@`` written as ``&``, the 0.3.x spelling) equals
  ``<source>`` and whose new folder ends with the old folder (every historical 0.3.x root
  lies at or below ``source_root``). One candidate: the unit moves (``os.replace``) to
  ``<new prefix><tail>``, with the source basename verbatim (a chained ``x&A.fif@B.nc``
  becomes ``x@A.fif@B.nc``) and the name verbatim. None: *orphan*, left in place, unless
  the source part looks like a 0.3.x chained output (``...&<Name><suffix>``): then
  *unresolved*, refused (it would otherwise be recomputed silently). Several (e.g. sources
  ``x@A.fif`` and ``x&A.fif`` both exist): *ambiguous*, refused. ``&`` is never decoded
  globally: a source really named with ``&`` matches its own key and keeps it.
- **Resume.** Files already at their 0.4.0 name (moved by an interrupted ``--apply``) are
  recognized by the planned targets: a file ``<b>@<name><suffix>`` that the 0.3.x parser
  rejects counts only when ``<dir>/<b>`` is the new reference base of a discovered source.
  It is an ``identity`` unit and joins the remaining old files of the same unit, so the
  unit gets its ``.done``.
- **MNE split recordings** (spec 10.3.5; ``-<n>.fif`` and compressed ``.fif-<n>.gz`` parts,
  the rule of :func:`neurodags.iterators.neuromag_split_part`) are re-saved with MNE into a
  private staging folder (``.neurodags/staging/<txn>``), verified sample for sample,
  fingerprinted in the journal and published with ``os.link`` (never replacing a file);
  only then are the old parts deleted. A split whose names do not change (folder only) is
  moved part by part, its part list and data shape journaled once so any number of
  interrupted runs rejoin it, and loaded in full before its ``.done`` is written.
- **Manifests** (spec 10.4). Moved units get a ``.done`` with ``"origin": "migrate"`` when the
  derivative's set of artifact suffixes is known (the set decides completeness; the manifest
  lists every artifact of the unit).
- **No overwrites, one run per tree.** Moves are ``os.link`` + ``os.unlink`` (an exclusive
  claim of the new name without hard links). ``--apply`` holds ``migrate.lock`` next to each
  tree's journal for its whole run; a lock is never taken over automatically.

Use :func:`plan_migration` (read-only), :func:`format_preview` and :func:`apply_migration`.
The CLI wrapper is ``neurodags migrate``.
"""

from __future__ import annotations

import errno
import filecmp
import hashlib
import json
import ntpath
import os
import posixpath
import re
import shutil
import socket
import tempfile
import uuid
import warnings
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import yaml

from neurodags._legacy import snake_to_camel
from neurodags.definitions import DatasetConfig
from neurodags.iterators import discover_inputs, neuromag_split_name, neuromag_split_part
from neurodags.loaders import UniqueKeySafeLoader, check_pipeline_names, load_configuration
from neurodags.loggers import get_logger
from neurodags.naming import NamingError, error_column, format_column, parse_output_basename
from neurodags.outputs import (
    build_manifest,
    dataset_derivatives_base,
    output_prefix,
    output_reference_base,
    unit_state,
    write_manifest,
)
from neurodags.utils import (
    DatasetConfigError,
    expand_path,
    inspect_dataset_paths,
    resolve_dataset_field,
    suggest_source_root,
)

log = get_logger(__name__)

__all__ = [
    "BACKUP_SUFFIX",
    "JOURNAL_NAME",
    "LOCK_NAME",
    "ApplyResult",
    "MigrationError",
    "MigrationPlan",
    "MigrationRefused",
    "apply_migration",
    "format_apply_summary",
    "format_preview",
    "legacy_column_to_new",
    "plan_migration",
    "write_report",
]

#: Suffix of the backups of rewritten configuration files.
BACKUP_SUFFIX = ".bak-0.3"
#: Journal of completed operations, under ``<derivatives_path>/.neurodags/`` (JSON Lines).
JOURNAL_NAME = "migrate-journal.json"
#: Lock held by ``--apply`` for its whole run, next to the journal (one migrate per tree).
LOCK_NAME = "migrate.lock"
#: Version of the ``--report`` JSON document.
REPORT_VERSION = 1

_PATH_FIELDS = ("file_pattern", "exclude_pattern", "derivatives_path")
_MARKER_TAILS = {".error": "error", ".skip": "skip", ".done": "done"}
_SPLIT_START = re.compile(r"^-\d+\.")
_SPLIT_NAME = re.compile(r"(.+)-(\d+)")
_WINDOWS_ABS = re.compile(r"^[A-Za-z]:[\\/]")
_MAX_LISTED = 20


class MigrationError(ValueError):
    """The configuration cannot be read or migrated (a user error; nothing was changed)."""


class MigrationRefused(MigrationError):
    """The plan has refusals (spec 10.5); nothing was applied."""

    def __init__(self, refusals: Sequence[str]):
        self.refusals = list(refusals)
        lines = "\n".join(f"  - {r}" for r in self.refusals)
        super().__init__(f"migration refused, nothing was changed:\n{lines}")


# ---------------------------------------------------------------------------
# Plan data structures
# ---------------------------------------------------------------------------


@dataclass
class ConfigChange:
    """A configuration file the migration rewrites (written only by ``--apply``)."""

    path: str
    role: str  # "pipeline", "datasets" or "pipeline+datasets"
    original: str
    new_text: str
    changes: list[str]
    comments_preserved: bool

    @property
    def backup(self) -> str:
        return self.path + BACKUP_SUFFIX


@dataclass
class _Candidate:
    dataset: str
    rel: str
    path: str
    reference_base: str  # absolute, new layout


@dataclass
class MneRewrite:
    """One MNE split recording re-saved under its new name."""

    suffix: str
    old_parts: list[str]
    new_path: str
    #: Transaction id (journal and staging folder name).
    txn: str = ""
    #: Private folder the new recording is saved and verified in before publication
    #: (``<journal folder>/staging/<txn>``).
    staging: str = ""
    #: Set when a previous, interrupted ``--apply`` already saved and verified the new
    #: recording in its staging folder (journal ``mne-staged``): ``[name, size, sha256]``
    #: per part; it is published (never overwriting) and the old parts are deleted.
    staged: list[list[Any]] | None = None


def _is_fif_tail(tail: str) -> bool:
    low = tail.lower()
    return low.endswith((".fif", ".fif.gz"))


def _split_tail(tail: str) -> tuple[str, int] | None:
    """``(main_tail, index)`` of a split continuation tail (``-2.fif``, ``.fif-1.gz``).

    Same rule as discovery (:func:`neurodags.iterators.neuromag_split_part`), applied to
    the tail after the derivative token.
    """
    part = neuromag_split_part("x" + tail)
    if part is None:
        return None
    return part[0][1:], part[1]


def _split_names(target: str, count: int) -> list[str]:
    """*target* and the names of its first *count* continuation parts (MNE naming)."""
    return [target] + [neuromag_split_name(target, i) for i in range(1, count + 1)]


@dataclass
class Unit:
    """The 0.3.x files of one (source, derivative) unit found in a tree."""

    tree: str
    directory: str
    old_dir: str | None  # relative to the tree root; None for the pre-0.3.3 single-file form
    source_part: str
    token: str
    derivative: str  # new (verbatim) name
    artifacts: dict[str, list[str]] = field(default_factory=dict)  # tail -> [main, parts...]
    markers: dict[str, str] = field(default_factory=dict)  # error/skip/done -> path
    #: move | identity | orphan | unresolved | ambiguous | duplicate | conflict
    status: str = "pending"
    #: Recognized at its 0.4.0 name (a planned target), not by the 0.3.x parser.
    canonical: bool = False
    candidate: _Candidate | None = None
    candidates: list[_Candidate] = field(default_factory=list)
    new_prefix: str | None = None
    moves: list[tuple[str, str]] = field(default_factory=list)
    #: Where each old file ends up (moved, already there, or identical duplicate).
    targets: dict[str, str] = field(default_factory=dict)
    mne: list[MneRewrite] = field(default_factory=list)
    #: Split recordings moved part by part (only the folder changes): ``txn``, every
    #: part's original ``(old, new)`` and the recording's ``shape``; journaled before the
    #: first move so any number of interrupted runs rejoin the parts.
    split_moves: list[dict[str, Any]] = field(default_factory=list)
    #: Scanned path -> original 0.3.x path, for parts an interrupted run already moved.
    logical: dict[str, str] = field(default_factory=dict)
    duplicate_files: list[str] = field(default_factory=list)
    manifest_suffixes: list[str] | None = None
    manifest_reason: str | None = None
    duplicate_of: str | None = None
    #: Units sharing this unit's target with disjoint files (e.g. a run interrupted halfway
    #: through a unit), this unit first; set on the group's first unit only.
    members: list[Unit] = field(default_factory=list)
    #: True for the other units of such a group (their manifest is the first unit's).
    secondary: bool = False

    @property
    def tails(self) -> set[str]:
        return set(self.artifacts) | {"." + m for m in self.markers}

    @property
    def old_prefix(self) -> str:
        return os.path.join(self.directory, f"{self.source_part}@{self.token}")

    @property
    def files(self) -> list[str]:
        out = [p for paths in self.artifacts.values() for p in paths]
        out += [p for marker, p in self.markers.items() if marker != "done"]
        return out

    @property
    def label(self) -> str:
        return f"{self.old_prefix}*"


@dataclass
class TreePlan:
    """One scanned output tree: a ``derivatives_path`` or a folder of sources."""

    root: str
    kind: str  # "derivatives" or "sources"
    datasets: list[str]
    journal: str
    units: list[Unit] = field(default_factory=list)
    unrecognized: list[str] = field(default_factory=list)
    #: Scan below ``root`` recursively (default: only for ``derivatives`` trees).
    recursive: bool | None = None
    #: Folders another tree of the plan already scans (skipped here).
    skip_dirs: set[str] = field(default_factory=set)


@dataclass
class MigrationPlan:
    """Everything :func:`apply_migration` will do; built by :func:`plan_migration`."""

    pipeline: str
    datasets_file: str | None
    mount: str | None
    legacy_cwd: str | None
    mode: str  # "legacy" (0.3.x config) or "tree-only" (config already naming_version 1)
    synthesize: bool
    renames: dict[str, str]
    tokens: dict[str, str]
    config_changes: list[ConfigChange] = field(default_factory=list)
    source_roots: dict[str, dict[str, str]] = field(default_factory=dict)
    trees: list[TreePlan] = field(default_factory=list)
    artifact_sets: dict[str, dict[str, Any]] = field(default_factory=dict)
    refusals: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    chained_patterns: list[str] = field(default_factory=list)
    column_map: dict[str, Any] = field(default_factory=dict)
    #: Journal state of an interrupted ``--apply``: the last ``mne-*`` entry per rewrite
    #: transaction, open split moves, and recorded duplicates (by target prefix).
    _mne_txns: dict[str, dict[str, Any]] = field(default_factory=dict, repr=False)
    _split_txns: dict[str, dict[str, Any]] = field(default_factory=dict, repr=False)
    _duplicates: dict[str, dict[str, list[Any]]] = field(default_factory=dict, repr=False)

    def units(self, *statuses: str) -> list[Unit]:
        return [u for t in self.trees for u in t.units if not statuses or u.status in statuses]

    @property
    def moves(self) -> list[tuple[str, str]]:
        return [m for u in self.units("move") for m in u.moves]

    @property
    def mne_rewrites(self) -> list[MneRewrite]:
        return [r for u in self.units("move") for r in u.mne]

    @property
    def manifests(self) -> list[Unit]:
        return [u for u in self.units("move", "identity") if u.manifest_suffixes is not None]

    @property
    def has_work(self) -> bool:
        return bool(self.moves or self.mne_rewrites or self.manifests or self.config_changes)

    def to_dict(self) -> dict[str, Any]:
        """The ``--report`` document (spec 10.6)."""

        def unit_info(u: Unit) -> dict[str, Any]:
            return {
                "dataset": u.candidate.dataset if u.candidate else None,
                "rel": u.candidate.rel if u.candidate else None,
                "derivative": u.derivative,
                "old_prefix": u.old_prefix,
                "new_prefix": u.new_prefix,
            }

        return {
            "report_version": REPORT_VERSION,
            "pipeline": self.pipeline,
            "datasets_file": self.datasets_file,
            "mount": self.mount,
            "legacy_cwd": self.legacy_cwd,
            "mode": self.mode,
            "config": {
                "renames": self.renames,
                "source_roots": self.source_roots,
                "files": [
                    {
                        "path": c.path,
                        "backup": c.backup,
                        "role": c.role,
                        "changes": c.changes,
                        "comments_preserved": c.comments_preserved,
                    }
                    for c in self.config_changes
                ],
            },
            "trees": [
                {"root": t.root, "kind": t.kind, "datasets": t.datasets, "journal": t.journal}
                for t in self.trees
            ],
            "moves": [
                {**unit_info(u), "old": old, "new": new}
                for u in self.units("move")
                for old, new in u.moves
            ],
            "mne_rewrites": [
                {
                    **unit_info(u),
                    "old_parts": r.old_parts,
                    "new": r.new_path,
                    "resume": r.staged is not None,
                }
                for u in self.units("move")
                for r in u.mne
            ],
            "manifests": [
                {**unit_info(u), "done": u.new_prefix + ".done", "artifacts": u.manifest_suffixes}
                for u in self.manifests
            ],
            "without_manifest": [
                {**unit_info(u), "reason": u.manifest_reason}
                for u in self.units("move", "identity")
                if u.manifest_suffixes is None and u.manifest_reason
            ],
            "orphans": [f for u in self.units("orphan") for f in u.files],
            "unresolved": [f for u in self.units("unresolved") for f in u.files],
            "ambiguous": [
                {
                    "files": u.files,
                    "candidates": [f"{c.dataset}: {c.rel}" for c in u.candidates],
                }
                for u in self.units("ambiguous")
            ],
            "duplicates": [
                {"files": u.duplicate_files or u.files, "same_as": u.duplicate_of}
                for u in self.units()
                if u.status == "duplicate" or u.duplicate_files
            ],
            "unrecognized": [f for t in self.trees for f in t.unrecognized],
            "artifact_sets": self.artifact_sets,
            "chained_patterns": self.chained_patterns,
            "column_map": self.column_map,
            "refusals": self.refusals,
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# YAML text editing (keeps comments and ordering)
# ---------------------------------------------------------------------------


class _NeedsDump(Exception):
    """The YAML text cannot be edited in place; fall back to re-serializing it."""


def _yaml_scalar(value: str, style: str | None = None) -> str:
    """Render *value* as a one-line YAML scalar, keeping the original quoting style."""
    if style == '"':
        return json.dumps(value, ensure_ascii=False)
    if style != "'" and value and "\n" not in value and " #" not in value:
        try:
            if yaml.safe_load(f"k: {value}\n") == {"k": value}:
                return value
        except yaml.YAMLError:
            pass
    return "'" + value.replace("'", "''") + "'"


def _map_items(node: Any) -> list[tuple[Any, Any]]:
    if not isinstance(node, yaml.MappingNode):
        raise _NeedsDump("not a mapping")
    return list(node.value)


def _map_get(node: Any, key: str) -> tuple[Any, Any] | None:
    for k, v in _map_items(node):
        if isinstance(k, yaml.ScalarNode) and k.value == key:
            return k, v
    return None


def _line_start(text: str, node: Any) -> tuple[int, int]:
    """Index of the start of *node*'s line and its column; refuses non-block layouts."""
    index, column = node.start_mark.index, node.start_mark.column
    start = index - column
    if start < 0 or text[start:index].strip():
        raise _NeedsDump("node is not first on its line")
    return start, column


class _Edits:
    def __init__(self, text: str):
        self.text = text
        self.edits: list[tuple[int, int, str]] = []

    def replace_scalar(self, node: Any, value: str) -> None:
        if not isinstance(node, yaml.ScalarNode) or node.style in ("|", ">"):
            raise _NeedsDump("not a plain scalar")
        self.edits.append(
            (node.start_mark.index, node.end_mark.index, _yaml_scalar(value, node.style))
        )

    def insert(self, index: int, text: str) -> None:
        self.edits.append((index, index, text))

    def apply(self) -> str:
        out = self.text
        last = None
        for start, end, new in sorted(self.edits, key=lambda e: (e[0], e[1]), reverse=True):
            if last is not None and end > last:
                raise _NeedsDump("overlapping edits")
            out = out[:start] + new + out[end:]
            last = start
        return out


def _show(value: Any) -> str:
    if isinstance(value, Mapping):
        return "{" + ", ".join(f"{k}: {v}" for k, v in value.items()) + "}"
    return str(value)


def _block_lines(indent: int, key: str, value: Any, child_delta: int) -> str:
    pad = " " * indent
    if isinstance(value, Mapping):
        lines = [f"{pad}{key}:"]
        lines += [
            f"{' ' * (indent + child_delta)}{k}: {_yaml_scalar(str(v))}" for k, v in value.items()
        ]
        return "\n".join(lines) + "\n"
    return f"{pad}{key}: {_yaml_scalar(str(value))}\n"


# ---------------------------------------------------------------------------
# Path rewriting (0.3.x cwd-relative -> 0.4.0 YAML-relative)
# ---------------------------------------------------------------------------


def _looks_absolute(value: str) -> bool:
    return value.startswith(("/", "~", "$", "%", "\\")) or bool(_WINDOWS_ABS.match(value))


def _rewrite_path(value: Any, shift: str) -> Any:
    """Rewrite a relative 0.3.x path so it resolves to the same place from the YAML's folder.

    *shift* is the old working directory relative to the YAML's directory (POSIX). Values
    that are absolute (or start with ``~``/``$VAR``) are kept; mount maps are rewritten per
    entry.
    """
    if isinstance(value, Mapping):
        return {k: _rewrite_path(v, shift) for k, v in value.items()}
    if not isinstance(value, str) or not value or shift == "." or _looks_absolute(value):
        return value
    if "\\" in value:
        return ntpath.normpath(ntpath.join(shift.replace("/", "\\"), value))
    return posixpath.normpath(posixpath.join(shift, value))


def _suggest_root(pattern: Any) -> str | None:
    """``suggest_source_root`` on a raw (unexpanded) pattern; ``"."`` for a bare relative glob."""
    if not isinstance(pattern, str) or not pattern:
        return None
    suggestion = suggest_source_root(pattern)
    if suggestion is None and not _looks_absolute(pattern):
        return "."
    return suggestion


def _is_filesystem_root(path: str) -> bool:
    norm = os.path.normpath(path)
    return os.path.dirname(norm) == norm


# ---------------------------------------------------------------------------
# Loading the configuration
# ---------------------------------------------------------------------------


@dataclass
class _Source:
    """Where the datasets live: a separate YAML or the pipeline's inline ``datasets:``."""

    path: str
    inline: bool
    data: dict[str, Any]


def _read_text(path: str) -> str:
    try:
        with open(path, encoding="utf-8") as handle:
            return handle.read()
    except OSError as exc:
        raise MigrationError(f"cannot read {path}: {exc}") from exc


def _load_yaml(path: str) -> dict[str, Any]:
    try:
        return load_configuration(path)
    except (OSError, ValueError, TypeError) as exc:
        raise MigrationError(f"cannot load {path}: {exc}") from exc


def _datasets_source(
    pipeline_path: str, config: Mapping[str, Any], datasets: str | None
) -> _Source:
    if datasets is not None:
        path = os.path.abspath(datasets)
        return _Source(path, False, _load_yaml(path))
    value = config.get("datasets")
    if value is None:
        raise MigrationError(f"{pipeline_path} has no 'datasets' and no -d/--datasets was given.")
    if isinstance(value, str):
        try:
            expanded = expand_path(value, field="datasets")
        except ValueError as exc:
            raise MigrationError(str(exc)) from exc
        if not os.path.isabs(expanded):
            expanded = os.path.join(os.path.dirname(pipeline_path), expanded)
        path = os.path.normpath(expanded)
        return _Source(path, False, _load_yaml(path))
    if isinstance(value, dict):
        return _Source(pipeline_path, True, value)
    raise MigrationError("'datasets' must be a mapping or a path to a datasets YAML.")


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------


def _parse_pairs(pairs: Mapping[str, str] | Iterable[str] | None, what: str) -> dict[str, str]:
    if pairs is None:
        return {}
    if isinstance(pairs, Mapping):
        return {str(k): str(v) for k, v in pairs.items()}
    out: dict[str, str] = {}
    for item in pairs:
        old, sep, new = str(item).partition("=")
        if not sep or not old or not new:
            raise MigrationError(f"{what} expects OLD=NEW, got {item!r}.")
        if old in out and out[old] != new:
            raise MigrationError(f"{what} {old!r} given twice with different values.")
        out[old] = new
    return out


def _parse_source_roots(
    values: Mapping[str, str] | Iterable[str] | None,
) -> dict[str, dict[str | None, str]]:
    """``DATASET=PATH`` (every mount) or ``DATASET:MOUNT=PATH``."""
    out: dict[str, dict[str | None, str]] = {}
    for key, path in _parse_pairs(values, "--source-root").items():
        dataset, sep, mount = key.partition(":")
        out.setdefault(dataset, {})[mount if sep else None] = path
    return out


def _rename_reference(value: Any, renames: Mapping[str, str]) -> Any:
    if not isinstance(value, str):
        return value
    name, dot, rest = value.partition(".")
    if name in renames:
        return renames[name] + dot + rest
    return value


def _renamed_pipeline(config: Mapping[str, Any], renames: Mapping[str, str]) -> dict[str, Any]:
    new = dict(config)
    defs = config.get("DerivativeDefinitions")
    if isinstance(defs, Mapping):
        new_defs: dict[Any, Any] = {}
        for key, definition in defs.items():
            new_key = renames.get(str(key), key)
            if isinstance(definition, Mapping) and isinstance(definition.get("nodes"), list):
                definition = dict(definition)
                definition["nodes"] = [
                    (
                        {**step, "derivative": _rename_reference(step["derivative"], renames)}
                        if isinstance(step, Mapping) and "derivative" in step
                        else step
                    )
                    for step in definition["nodes"]
                ]
            new_defs[new_key] = definition
        new["DerivativeDefinitions"] = new_defs
    if isinstance(config.get("DerivativeList"), list):
        new["DerivativeList"] = [renames.get(str(n), n) for n in config["DerivativeList"]]
    return new


def _pipeline_edits(
    edits: _Edits, root: Any, renames: Mapping[str, str], add_version: bool
) -> list[str]:
    changes: list[str] = []
    items = _map_items(root)
    if not items:
        raise _NeedsDump("empty pipeline")
    if add_version:
        start, column = _line_start(edits.text, items[0][0])
        edits.insert(start, " " * column + "naming_version: 1\n")
        changes.append("+ naming_version: 1")
    if not renames:
        return changes
    defs = _map_get(root, "DerivativeDefinitions")
    if defs is not None:
        for key_node, value_node in _map_items(defs[1]):
            if key_node.value in renames:
                edits.replace_scalar(key_node, renames[key_node.value])
                changes.append(
                    f"~ DerivativeDefinitions: {key_node.value} -> {renames[key_node.value]}"
                )
            nodes = (
                _map_get(value_node, "nodes") if isinstance(value_node, yaml.MappingNode) else None
            )
            if nodes is None or not isinstance(nodes[1], yaml.SequenceNode):
                continue
            for step in nodes[1].value:
                if not isinstance(step, yaml.MappingNode):
                    continue
                ref = _map_get(step, "derivative")
                if ref is None or not isinstance(ref[1], yaml.ScalarNode):
                    continue
                new_ref = _rename_reference(ref[1].value, renames)
                if new_ref != ref[1].value:
                    edits.replace_scalar(ref[1], new_ref)
                    changes.append(
                        f"~ DerivativeDefinitions.{key_node.value}: derivative "
                        f"{ref[1].value} -> {new_ref}"
                    )
    dlist = _map_get(root, "DerivativeList")
    if dlist is not None and isinstance(dlist[1], yaml.SequenceNode):
        for item in dlist[1].value:
            if isinstance(item, yaml.ScalarNode) and item.value in renames:
                edits.replace_scalar(item, renames[item.value])
                changes.append(f"~ DerivativeList: {item.value} -> {renames[item.value]}")
    return changes


def _datasets_edits(
    edits: _Edits,
    node: Any,
    old: Mapping[str, Any],
    new: Mapping[str, Any],
    prefix: str,
) -> list[str]:
    """Edits turning the dataset mapping *node* (with values *old*) into *new*."""
    changes: list[str] = []
    for key_node, ds_node in _map_items(node):
        name = key_node.value
        if name not in new or not isinstance(new[name], Mapping):
            continue
        old_ds, new_ds = old[name], new[name]
        for attr in _PATH_FIELDS:
            if old_ds.get(attr) == new_ds.get(attr):
                continue
            found = _map_get(ds_node, attr)
            if found is None:
                raise _NeedsDump(f"{attr} not written explicitly")
            value_node = found[1]
            if isinstance(value_node, yaml.ScalarNode):
                edits.replace_scalar(value_node, new_ds[attr])
            else:
                for mk, mv in _map_items(value_node):
                    if old_ds[attr].get(mk.value) != new_ds[attr].get(mk.value):
                        edits.replace_scalar(mv, new_ds[attr][mk.value])
            changes.append(
                f"~ {prefix}{name}.{attr}: {_show(old_ds[attr])} -> {_show(new_ds[attr])}"
            )
        if "source_root" in new_ds and "source_root" not in old_ds:
            anchor = _map_get(ds_node, "file_pattern")
            anchor_key = anchor[0] if anchor is not None else _map_items(ds_node)[0][0]
            start, column = _line_start(edits.text, anchor_key)
            delta = 2
            if anchor is not None and isinstance(anchor[1], yaml.MappingNode) and anchor[1].value:
                delta = max(1, anchor[1].value[0][0].start_mark.column - column)
            edits.insert(start, _block_lines(column, "source_root", new_ds["source_root"], delta))
            changes.append(f"+ {prefix}{name}.source_root: {_show(new_ds['source_root'])}")
    return changes


def _with_source_root(ds: Mapping[str, Any], source_root: Any, shift: str) -> dict[str, Any]:
    """New dataset mapping: rewritten paths and ``source_root`` placed before ``file_pattern``."""
    out: dict[str, Any] = {}
    for key, value in ds.items():
        if key == "file_pattern" and source_root is not None and "source_root" not in ds:
            out["source_root"] = source_root
        out[key] = _rewrite_path(value, shift) if key in _PATH_FIELDS else value
    if source_root is not None and "source_root" not in out:
        out["source_root"] = source_root
    return out


def _render_change(
    path: str,
    role: str,
    text: str,
    expected: dict[str, Any],
    build: Any,
) -> ConfigChange:
    """Edit *text* in place with *build(edits, root)*; fall back to a dump when needed."""
    changes: list[str] = []
    try:
        root = yaml.compose(text, Loader=yaml.SafeLoader)
        edits = _Edits(text)
        changes = build(edits, root)
        new_text = edits.apply()
        if yaml.load(new_text, Loader=UniqueKeySafeLoader) != expected:
            raise _NeedsDump("edited text does not load as expected")
        preserved = True
    except (_NeedsDump, yaml.YAMLError, ValueError) as exc:
        log.debug("Falling back to YAML dump", path=path, reason=str(exc))
        new_text = yaml.safe_dump(expected, sort_keys=False, allow_unicode=True)
        preserved = False
        if not changes:
            changes = ["(file re-serialized)"]
    return ConfigChange(path, role, text, new_text, changes, preserved)


def _scan_tree(root: str, *, recursive: bool, exclude: Iterable[str] = ()) -> Iterable[str]:
    """Files whose basename contains ``@`` (excluding ``code/``, ``.neurodags/``, *exclude*)."""
    excluded = {os.path.normcase(os.path.normpath(d)) for d in exclude}
    if not os.path.isdir(root):
        return
    if not recursive:
        for entry in sorted(os.listdir(root)):
            path = os.path.join(root, entry)
            if "@" in entry and not entry.startswith(".") and os.path.isfile(path):
                yield path
        return
    for dirpath, dirnames, filenames in os.walk(root):
        if dirpath == root:
            dirnames[:] = [d for d in dirnames if d != "code"]
        # .neurodags/ (journal, lock, migration staging) is never output, at any level.
        dirnames[:] = sorted(
            d
            for d in dirnames
            if d != ".neurodags"
            and os.path.normcase(os.path.normpath(os.path.join(dirpath, d))) not in excluded
        )
        for name in sorted(filenames):
            if "@" in name and not name.startswith("."):
                yield os.path.join(dirpath, name)


def _parse_legacy(basename: str, tokens: Sequence[str]) -> tuple[str, str, str] | None:
    """``(source_part, token, tail)`` with the 0.3.x rules (longest token first)."""
    source_part, sep, rest = basename.partition("@")
    if not sep or not source_part or "@" in rest:
        return None
    for token in tokens:
        if not rest.startswith(token):
            continue
        tail = rest[len(token) :]
        if (tail.startswith(".") and len(tail) > 1) or _SPLIT_START.match(tail):
            return source_part, token, tail
    return None


def _parse_canonical(basename: str, names: set[str]) -> tuple[str, str, str] | None:
    """``(source_part, name, tail)`` of a 0.4.0 output name (last ``@``), else ``None``.

    Only used for files the 0.3.x parser rejects (more than one ``@``); whether the file
    really is a planned target is decided by the caller against the discovered sources.
    """
    parsed = parse_output_basename(basename)
    if parsed is None:
        return None
    if parsed.name in names:
        return parsed.source_part, parsed.name, parsed.suffix
    match = _SPLIT_NAME.fullmatch(parsed.name)
    if match and match.group(1) in names and parsed.suffix == ".fif":
        return parsed.source_part, match.group(1), f"-{match.group(2)}.fif"
    return None


def _legacy_key(basename: str) -> str:
    """How 0.3.x spelled a source basename in output names (every ``@`` written ``&``)."""
    return basename.replace("@", "&")


def _looks_chained(source_part: str) -> bool:
    """A 0.3.x source part ``<head>&<Name><suffix>``: the output of an upstream stage."""
    head, sep, tail = source_part.rpartition("&")
    return bool(sep and head) and parse_output_basename(f"{head}@{tail}") is not None


def _is_path_suffix(old_dir: str, new_dir: str) -> bool:
    if old_dir in ("", "."):
        return True
    return new_dir == old_dir or new_dir.endswith("/" + old_dir)


def _same_files(a: Unit, b: Unit) -> bool:
    """Same artifacts and same 0.3.x ``.error``/``.skip`` markers (byte for byte).

    A ``.done`` is not old output content: the keeper of an earlier migration has one (it
    was written by that migration), its identical duplicate does not.
    """
    markers_a = {m: p for m, p in a.markers.items() if m != "done"}
    markers_b = {m: p for m, p in b.markers.items() if m != "done"}
    if set(a.artifacts) != set(b.artifacts) or set(markers_a) != set(markers_b):
        return False
    pairs = [
        (x, y) for t in a.artifacts for x, y in zip(a.artifacts[t], b.artifacts[t], strict=False)
    ]
    if any(len(a.artifacts[t]) != len(b.artifacts[t]) for t in a.artifacts):
        return False
    pairs += [(markers_a[m], markers_b[m]) for m in markers_a]
    try:
        return all(filecmp.cmp(x, y, shallow=False) for x, y in pairs)
    except OSError:
        return False


def _existing_ancestor(path: str) -> str:
    path = os.path.abspath(path)
    while not os.path.exists(path):
        parent = os.path.dirname(path)
        if parent == path:
            break
        path = parent
    return path


def read_journal(path: str) -> list[dict[str, Any]]:
    """Entries of a migration journal (JSON Lines; a truncated last line is ignored)."""
    entries: list[dict[str, Any]] = []
    try:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                try:
                    entry = json.loads(line)
                except ValueError:
                    continue
                if isinstance(entry, dict):
                    entries.append(entry)
    except OSError:
        pass
    return entries


def _txn_id(kind: str, *paths: str) -> str:
    """Stable id of one migration transaction (journal key and staging folder name)."""
    key = "\0".join([kind, *(os.path.normcase(os.path.normpath(p)) for p in paths)])
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def _unfinished_split_moves(journal: str) -> dict[str, dict[str, Any]]:
    """Split recordings an interrupted ``--apply`` was moving, by transaction id.

    The ``split-move-start`` entry (written once per transaction, with the original part
    list) of every transaction without a later ``split-move-done``.
    """
    open_moves: dict[str, dict[str, Any]] = {}
    for entry in read_journal(journal):
        op, txn, parts = entry.get("op"), entry.get("txn"), entry.get("parts")
        if op not in ("split-move-start", "split-move-done") or not isinstance(txn, str):
            continue
        if op == "split-move-done":
            open_moves.pop(txn, None)
            continue
        if not isinstance(parts, list) or not all(
            isinstance(p, list) and len(p) == 2 for p in parts
        ):
            continue
        if txn not in open_moves:  # the first record is the original part list
            entry = dict(entry)
            entry["parts"] = [
                [os.path.normpath(str(o)), os.path.normpath(str(n))] for o, n in parts
            ]
            open_moves[txn] = entry
    return open_moves


def _same_entry(a: str, b: str) -> bool:
    try:
        return os.path.samefile(a, b)
    except OSError:
        return False


def _two_names(a: str, b: str) -> bool:
    """*a* and *b* are two directory entries (hard links), not one name seen twice.

    On a case-insensitive filesystem ``x@Clean.nc`` and ``x@clean.nc`` are the same entry.
    """
    try:
        return os.path.basename(a) in os.listdir(os.path.dirname(a) or ".") and (
            os.path.basename(b) in os.listdir(os.path.dirname(b) or ".")
        )
    except OSError:
        return False


def plan_migration(
    pipeline: str | os.PathLike,
    datasets: str | os.PathLike | None = None,
    *,
    mount: str | None = None,
    renames: Mapping[str, str] | Iterable[str] | None = None,
    legacy_cwd: str | os.PathLike | None = None,
    synthesize: bool = True,
    source_roots: Mapping[str, str] | Iterable[str] | None = None,
    upstream_renames: Mapping[str, str] | Iterable[str] | None = None,
) -> MigrationPlan:
    """Plan the migration of one pipeline's configuration and output tree (read-only).

    Parameters
    ----------
    pipeline : path-like
        The 0.3.x ``pipeline.yml`` (or an already migrated one, see Notes).
    datasets : path-like, optional
        A datasets YAML used instead of the pipeline's ``datasets`` (the CLI ``-d``).
    mount : str, optional
        Mount point to migrate (default: the pipeline's ``mount_point``). Run once per
        output tree, i.e. once per mount.
    renames : mapping or iterable of ``"OLD=NEW"``, optional
        Derivative renames (``--map``). Default: keep the YAML spelling verbatim.
    legacy_cwd : path-like, optional
        Directory the 0.3.x commands were run from; relative dataset paths were resolved
        against it. Default: the current working directory.
    synthesize : bool
        Write ``.done`` manifests for moved units whose artifact set is known (spec 10.4).
    source_roots : mapping or iterable of ``"DATASET=PATH"`` / ``"DATASET:MOUNT=PATH"``
        Explicit ``source_root`` values (required when the suggestion would be a
        filesystem root). Written verbatim (0.4.0 rule: relative to the YAML).
    upstream_renames : mapping or iterable of ``"OLDTOKEN=NEWNAME"``, optional
        For chained pipelines whose sources are another pipeline's outputs: the upstream
        rename (``@OLDTOKEN`` -> ``@NEWNAME``). Old source parts ``...&OLDTOKEN.<ext>`` then
        match the renamed sources ``...@NEWNAME.<ext>`` (compared in the 0.3.x ``&``
        spelling, see the module docstring).

    Returns
    -------
    MigrationPlan
        With ``refusals`` listing every reason not to apply (empty when it can be applied).

    Notes
    -----
    A pipeline that already declares ``naming_version: 1`` is planned in *tree-only* mode
    (another mount's tree, or an interrupted ``--apply``): the configuration is not
    changed. Old tokens then come from the current names plus ``renames`` (``OLD`` is the
    0.3.x name, ``NEW`` the current one). Outputs still needing moves are refused unless
    the ``.bak-0.3`` backup written by a previous migration exists.
    """
    pipeline_path = os.path.abspath(os.fspath(pipeline))
    cwd = os.path.abspath(os.fspath(legacy_cwd)) if legacy_cwd is not None else os.getcwd()
    rename_map = _parse_pairs(renames, "--map")
    upstream = _parse_pairs(upstream_renames, "--upstream-map")
    explicit_roots = _parse_source_roots(source_roots)
    pipeline_text = _read_text(pipeline_path)
    config = _load_yaml(pipeline_path)

    version = config.get("naming_version")
    if version is None:
        mode = "legacy"
    elif type(version) is int and version == 1:
        mode = "tree-only"
    else:
        raise MigrationError(
            f"{pipeline_path} declares naming_version: {version!r}; migrate converts 0.3.x "
            "configurations (no naming_version) to naming_version 1."
        )
    source = _datasets_source(
        pipeline_path, config, os.fspath(datasets) if datasets is not None else None
    )
    active_mount = mount if mount is not None else config.get("mount_point")
    plan = MigrationPlan(
        pipeline=pipeline_path,
        datasets_file=None if source.inline else source.path,
        mount=active_mount,
        legacy_cwd=cwd if mode == "legacy" else None,
        mode=mode,
        synthesize=synthesize,
        renames=rename_map,
        tokens={},
    )

    # -- names and on-disk tokens --------------------------------------------------
    defs = config.get("DerivativeDefinitions") or {}
    if not isinstance(defs, Mapping):
        raise MigrationError("DerivativeDefinitions must be a mapping.")
    names = [str(k) for k in defs]
    if mode == "legacy":
        unknown = sorted(set(rename_map) - set(names))
        if unknown:
            plan.refusals.append(
                f"--map names unknown derivative(s) {', '.join(unknown)} (not in "
                "DerivativeDefinitions)."
            )
        old_to_new = {n: rename_map.get(n, n) for n in names}
    else:
        bad = sorted(o for o, n in rename_map.items() if n not in names)
        if bad:
            plan.refusals.append(
                f"--map OLD=NEW with an already migrated configuration: NEW must be a current "
                f"derivative name ({', '.join(f'{o}={rename_map[o]}' for o in bad)})."
            )
        old_to_new = {n: n for n in names}
        old_to_new.update({o: n for o, n in rename_map.items() if n in names})
    if _refuse_collapsing_renames(plan, mode, old_to_new, rename_map):
        return plan  # the renamed configuration would lose a definition: nothing to plan
    token_map: dict[str, set[str]] = {}
    for old, new in old_to_new.items():
        token_map.setdefault(snake_to_camel(old), set()).add(new)
        token_map.setdefault(new, set()).add(new)  # already renamed (resumed) outputs
    clashes = {t: sorted(n) for t, n in token_map.items() if len(n) > 1}
    for token, clash in sorted(clashes.items()):
        plan.refusals.append(
            f"derivatives {', '.join(clash)} share the on-disk token '@{token}'; their 0.3.x "
            "outputs cannot be told apart. Rename or remove one of them first."
        )
    plan.tokens = {t: next(iter(n)) for t, n in token_map.items() if len(n) == 1}

    # -- configuration rewrite ------------------------------------------------------
    old_datasets = source.data
    if not isinstance(old_datasets, dict) or not all(
        isinstance(v, Mapping) for v in old_datasets.values()
    ):
        raise MigrationError("every dataset entry must be a mapping.")
    ds_dir = os.path.dirname(source.path)
    with_root = [n for n, d in old_datasets.items() if d.get("source_root")]
    datasets_migrated = bool(old_datasets) and len(with_root) == len(old_datasets)
    if mode == "legacy" and with_root and not datasets_migrated:
        plan.refusals.append(
            f"datasets in {source.path} are partly migrated (source_root set for "
            f"{', '.join(map(str, with_root))} only); finish them by hand or restore the "
            "0.3.x file."
        )
    if mode == "tree-only" or datasets_migrated:
        new_datasets = {k: dict(v) for k, v in old_datasets.items()}
        if mode == "legacy":
            plan.notes.append(
                f"every dataset in {source.path} already has source_root: its paths are "
                "taken as 0.4.0 paths (relative to the YAML) and the file is not changed."
            )
    else:
        shift = os.path.relpath(cwd, ds_dir).replace(os.sep, "/")
        new_datasets = {}
        for name, ds in old_datasets.items():
            new_datasets[name] = _with_source_root(
                ds,
                _plan_source_root(plan, str(name), ds, shift, explicit_roots, active_mount),
                shift,
            )
            _check_var_paths(plan, str(name), ds, shift, active_mount)
    unknown_roots = sorted(set(explicit_roots) - {str(n) for n in old_datasets})
    if unknown_roots:
        plan.refusals.append(f"--source-root names unknown dataset(s) {', '.join(unknown_roots)}.")

    new_pipeline = _renamed_pipeline(config, rename_map) if mode == "legacy" else dict(config)
    if mode == "legacy":
        new_pipeline = {"naming_version": 1, **new_pipeline}
        if source.inline:
            new_pipeline["datasets"] = new_datasets
        elif datasets is not None and config.get("datasets") is not None:
            plan.notes.append(
                "the pipeline's own 'datasets' entry was not migrated (-d was given); run "
                "migrate without -d as well if it is still used."
            )
        try:
            for warning in check_pipeline_names(new_pipeline):
                plan.notes.append(f"name warning: {warning}")
        except NamingError as exc:
            plan.refusals.append(
                f"{exc}\n    Rename them with --map OLD=NEW (the new name must follow the "
                "0.4.0 grammar)."
            )
        _plan_config_files(plan, pipeline_text, config, new_pipeline, source, new_datasets)
        if not datasets_migrated:
            plan.notes.append(
                f"0.3.x dataset paths are read relative to the legacy working directory {cwd} "
                "(set with --legacy-cwd); in 0.4.0 they are relative to the YAML that "
                "defines them."
            )

    # -- enumerate the new layout ------------------------------------------------------
    configs: dict[str, DatasetConfig] = {}
    for name, ds in new_datasets.items():
        try:
            cfg = DatasetConfig(**ds)
        except Exception as exc:  # pydantic validation error
            plan.refusals.append(f"datasets.{name}: invalid dataset definition: {exc}")
            continue
        cfg._base_dir = ds_dir
        configs[str(name)] = cfg
    if plan.refusals:
        return plan
    _, problems = inspect_dataset_paths(configs, active_mount)
    if problems:
        plan.refusals.extend(problems)
        return plan
    try:
        enumeration = discover_inputs(configs, active_mount)
    except (DatasetConfigError, ValueError, KeyError) as exc:
        plan.refusals.append(str(exc))
        return plan

    _plan_trees(plan, configs, enumeration.entries, active_mount, upstream)
    ambiguous = plan.units("ambiguous")
    if ambiguous:
        shown = "; ".join(
            f"{u.label} matches "
            + ", ".join(f"{c.dataset}: {c.rel}" for c in u.candidates[:3])
            + (" ..." if len(u.candidates) > 3 else "")
            for u in ambiguous[:3]
        )
        plan.refusals.append(
            f"{len(ambiguous)} set(s) of 0.3.x outputs are ambiguous (several sources match "
            f"their name and folder): {shown}. Move them out of the tree (or delete them) "
            "and run migrate again; they will be recomputed."
        )
    unresolved = plan.units("unresolved")
    if unresolved:
        shown = ", ".join(u.label for u in unresolved[:3]) + (" ..." if len(unresolved) > 3 else "")
        plan.refusals.append(
            f"{len(unresolved)} set(s) of 0.3.x outputs of a chained pipeline match no "
            f"discovered source: {shown}. Their source part ('x&A.fif' in 'x&A.fif@B.nc') "
            "names an upstream output 'x@A.fif' that file_pattern does not find; left "
            "alone they would be recomputed. Fix the dataset (file_pattern, or "
            "--upstream-map OLDTOKEN=NEWNAME when the upstream name changed), or move "
            "them out of the tree to recompute them, and run migrate again."
        )
    _plan_targets(plan)
    _plan_artifact_sets(plan)
    _plan_manifests(plan)
    _plan_safety(plan)
    if mode == "tree-only" and plan.units("move"):
        if not os.path.exists(pipeline_path + BACKUP_SUFFIX):
            plan.refusals.append(
                f"{pipeline_path} already declares naming_version: 1 but the output tree "
                f"still has {len(plan.units('move'))} 0.3.x unit(s) to move, and no "
                f"{os.path.basename(pipeline_path)}{BACKUP_SUFFIX} shows this configuration "
                "was migrated by neurodags migrate. Restore the 0.3.x configuration and "
                "migrate it, or keep the backup next to the pipeline when migrating another "
                "mount's tree."
            )
    _plan_chained(plan, config, old_datasets, source)
    plan.column_map = _column_map(plan, old_to_new, defs)
    return plan


def _refuse_collapsing_renames(
    plan: MigrationPlan,
    mode: str,
    old_to_new: Mapping[str, str],
    rename_map: Mapping[str, str],
) -> bool:
    """Refuse ``--map`` renames that are not injective (also ignoring case).

    Legacy mode: every definition's new name (renamed or kept) must stay distinct, or two
    definitions collapse into one and one's outputs become the other's cache hits.
    Tree-only mode: two 0.3.x names cannot both be the same current name. Names that
    already collide without any rename are left to :func:`check_pipeline_names`.
    """
    if mode == "legacy":
        pairs = old_to_new.items()
    else:
        pairs = rename_map.items()
    by_new: dict[str, list[str]] = {}
    for old, new in pairs:
        by_new.setdefault(new.casefold(), []).append(old)
    found = False
    for olds in by_new.values():
        if len(olds) < 2 or all(old_to_new.get(o, o) == o for o in olds):
            continue
        found = True
        news = sorted({old_to_new.get(o, rename_map.get(o, o)) for o in olds})
        plan.refusals.append(
            f"--map would give the derivatives {', '.join(repr(o) for o in sorted(olds))} the "
            f"same name {' / '.join(repr(n) for n in news)} (names must stay distinct, also "
            "ignoring case): their definitions and outputs would collapse into one. Choose "
            "another NEW name."
        )
    return found


def _plan_source_root(
    plan: MigrationPlan,
    name: str,
    ds: Mapping[str, Any],
    shift: str,
    explicit: Mapping[str, Mapping[str | None, str]],
    active_mount: str | None,
) -> Any:
    """``source_root`` value for one dataset (explicit, else suggested per mount)."""
    pattern = _rewrite_path(ds.get("file_pattern"), shift)
    given = explicit.get(name, {})
    if None in given and len(given) == 1:
        value: Any = given[None]
        plan.source_roots[name] = {"*": f"{value} (given)"}
        return value
    if isinstance(pattern, Mapping):
        value = {}
        shown = {}
        for mount_name, mount_pattern in pattern.items():
            chosen = given.get(str(mount_name), given.get(None))
            how = "given"
            if chosen is None:
                chosen = _suggest_root(mount_pattern)
                how = "suggested"
            if chosen is None:
                plan.refusals.append(
                    f"datasets.{name}: no source_root can be suggested for mount "
                    f"'{mount_name}' (file_pattern {mount_pattern!r} starts at a filesystem "
                    f"root); pass --source-root {name}:{mount_name}=PATH."
                )
                continue
            value[mount_name] = chosen
            shown[str(mount_name)] = f"{chosen} ({how})"
        plan.source_roots[name] = shown
        _check_active_root(plan, name, value, active_mount)
        return value or None
    chosen = given.get(None) or (given.get(str(active_mount)) if active_mount else None)
    how = "given"
    if chosen is None:
        chosen = _suggest_root(pattern)
        how = "suggested"
    if chosen is None:
        plan.refusals.append(
            f"datasets.{name}: no source_root can be suggested (file_pattern {pattern!r} "
            f"starts at a filesystem root); pass --source-root {name}=PATH or add "
            "source_root to the dataset by hand."
        )
        return None
    plan.source_roots[name] = {"*": f"{chosen} ({how})"}
    _check_active_root(plan, name, chosen, active_mount)
    return chosen


def _check_active_root(
    plan: MigrationPlan, name: str, value: Any, active_mount: str | None
) -> None:
    if isinstance(value, Mapping) and (active_mount is None or active_mount not in value):
        return
    selected = value[active_mount] if isinstance(value, Mapping) else value
    try:
        expanded = expand_path(selected, field=f"datasets.{name}.source_root")
    except ValueError:
        return  # reported when the new configuration is checked
    if os.path.isabs(expanded) and _is_filesystem_root(expanded):
        plan.refusals.append(
            f"datasets.{name}: the suggested source_root {selected!r} is a filesystem root; "
            f"pass --source-root {name}=PATH."
        )


def _check_var_paths(
    plan: MigrationPlan, name: str, ds: Mapping[str, Any], shift: str, active_mount: str | None
) -> None:
    """``$VAR/...`` paths that expand to relative paths cannot be rewritten automatically."""
    if shift == ".":
        return
    for attr in _PATH_FIELDS:
        value = ds.get(attr)
        if isinstance(value, Mapping):
            value = value.get(active_mount) if active_mount is not None else None
        if not isinstance(value, str) or not value.startswith(("$", "%")):
            continue
        try:
            expanded = expand_path(value, field=f"datasets.{name}.{attr}")
        except ValueError:
            continue
        if not os.path.isabs(expanded):
            plan.refusals.append(
                f"datasets.{name}.{attr} {value!r} expands to the relative path {expanded!r}, "
                "which 0.3.x resolved against the working directory and 0.4.0 resolves "
                "against the YAML; rewrite it by hand (an absolute path is safest)."
            )


def _plan_config_files(
    plan: MigrationPlan,
    pipeline_text: str,
    config: Mapping[str, Any],
    new_pipeline: dict[str, Any],
    source: _Source,
    new_datasets: dict[str, Any],
) -> None:
    renames = {k: v for k, v in plan.renames.items() if k != v}
    datasets_changed = new_datasets != source.data

    def build_pipeline(edits: _Edits, root: Any) -> list[str]:
        changes = _pipeline_edits(edits, root, renames, add_version=True)
        if source.inline and datasets_changed:
            found = _map_get(root, "datasets")
            if found is None:
                raise _NeedsDump("datasets key not found")
            changes += _datasets_edits(edits, found[1], source.data, new_datasets, "datasets.")
        return changes

    role = "pipeline+datasets" if source.inline else "pipeline"
    plan.config_changes.append(
        _render_change(plan.pipeline, role, pipeline_text, new_pipeline, build_pipeline)
    )
    if not source.inline and datasets_changed:
        text = _read_text(source.path)

        def build_datasets(edits: _Edits, root: Any) -> list[str]:
            return _datasets_edits(edits, root, source.data, new_datasets, "")

        plan.config_changes.append(
            _render_change(source.path, "datasets", text, new_datasets, build_datasets)
        )
    for change in plan.config_changes:
        if os.path.exists(change.backup) and _read_text(change.backup) != change.original:
            plan.refusals.append(
                f"{change.backup} already exists with different content; move it away so the "
                f"backup of {change.path} cannot overwrite it."
            )
        if not os.access(change.path, os.W_OK) or not os.access(
            os.path.dirname(change.path), os.W_OK
        ):
            plan.refusals.append(f"{change.path} (or its folder) is not writable.")


def _plan_trees(
    plan: MigrationPlan,
    configs: Mapping[str, DatasetConfig],
    entries: Sequence[Any],
    mount: str | None,
    upstream: Mapping[str, str],
) -> None:
    """Scan every output tree and attach each 0.3.x unit to its candidate sources."""
    trees: dict[str, TreePlan] = {}
    # Per tree: candidates by legacy key (0.3.x spelling of the basename), and by their
    # new reference base (to recognize already moved, 0.4.0-named files).
    by_tree: dict[str, dict[str, list[_Candidate]]] = {}
    by_base: dict[str, dict[str, list[_Candidate]]] = {}
    pipeline_journal = os.path.join(os.path.dirname(plan.pipeline), ".neurodags", JOURNAL_NAME)
    derivatives_paths: dict[str, str | None] = {}
    for name, cfg in configs.items():
        if not cfg.skip:
            raw = resolve_dataset_field(cfg, "derivatives_path", mount)
            derivatives_paths[name] = os.path.normpath(raw) if raw else None
    # Every active dataset's output tree is scanned, even with zero discovered sources
    # (e.g. a chained pattern that no longer matches): its 0.3.x outputs must still be
    # reported (orphan / unresolved), never silently left behind.
    for name, root in derivatives_paths.items():
        if root:
            key = os.path.abspath(root)
            tree = trees.setdefault(
                key,
                TreePlan(key, "derivatives", [], os.path.join(key, ".neurodags", JOURNAL_NAME)),
            )
            if name not in tree.datasets:
                tree.datasets.append(name)
            by_tree.setdefault(key, {})
            by_base.setdefault(key, {})
    for entry in entries:
        cfg = configs[entry.dataset]
        base = os.path.abspath(
            output_reference_base(entry.rel, entry.path, dataset_derivatives_base(cfg, mount))
        )
        candidate = _Candidate(entry.dataset, entry.rel, entry.path, base)
        root = derivatives_paths.get(entry.dataset)
        if root:
            key = os.path.abspath(root)
            tree = trees.setdefault(
                key,
                TreePlan(key, "derivatives", [], os.path.join(key, ".neurodags", JOURNAL_NAME)),
            )
        else:
            key = os.path.dirname(os.path.abspath(entry.path))
            tree = trees.setdefault(key, TreePlan(key, "sources", [], pipeline_journal))
        if entry.dataset not in tree.datasets:
            tree.datasets.append(entry.dataset)
        by_tree.setdefault(key, {}).setdefault(_legacy_key(os.path.basename(base)), []).append(
            candidate
        )
        by_base.setdefault(key, {}).setdefault(os.path.normcase(base), []).append(candidate)
    # Datasets writing next to their sources: also scan the rest of source_root, so
    # outputs in folders without a discovered source are reported too.
    covered = {os.path.normcase(k) for k, t in trees.items() if t.kind == "sources"}
    for name, root in derivatives_paths.items():
        if root:
            continue
        source_root = resolve_dataset_field(configs[name], "source_root", mount)
        if not source_root:
            continue
        rest_root = os.path.abspath(source_root)
        key = "\0rest\0" + rest_root
        tree = trees.setdefault(
            key, TreePlan(rest_root, "sources", [], pipeline_journal, recursive=True)
        )
        tree.skip_dirs |= covered
        if name not in tree.datasets:
            tree.datasets.append(name)
        by_tree.setdefault(key, {})
        by_base.setdefault(key, {})

    tokens = sorted(plan.tokens, key=len, reverse=True)
    names = set(plan.tokens.values())
    derivative_roots = [k for k, t in trees.items() if t.kind == "derivatives"]
    for key, tree in trees.items():
        index = by_tree[key]
        bases = by_base[key]
        units: dict[tuple[str, str, str, bool], Unit] = {}
        recursive_scan = tree.kind == "derivatives" if tree.recursive is None else tree.recursive
        scans: list[tuple[str, bool]] = [(tree.root, recursive_scan)]
        if tree.kind == "derivatives":
            scans.append((os.path.dirname(tree.root), False))
        nested = [
            r for r in derivative_roots if r != tree.root and r.startswith(tree.root + os.sep)
        ]
        # Parts of a split recording an interrupted run was moving (folder-only): each
        # counts at its original location, so the recording is rejoined across folders.
        split_txns = _unfinished_split_moves(tree.journal)
        plan._split_txns.update(split_txns)
        relocated = {
            os.path.normcase(new): old for t in split_txns.values() for old, new in t["parts"]
        }
        for scan_root, recursive in scans:
            single_form = scan_root != tree.root
            for path in _scan_tree(scan_root, recursive=recursive, exclude=nested):
                if tree.skip_dirs and os.path.normcase(os.path.dirname(path)) in tree.skip_dirs:
                    continue
                logical = relocated.get(os.path.normcase(os.path.normpath(path)), path)
                if logical != path and os.path.lexists(logical):
                    if _same_entry(logical, path) and _two_names(logical, path):
                        continue  # interrupted between link and unlink: the old name counts
                    logical = path  # a different file at the old name: not ours to rejoin
                basename = os.path.basename(logical)
                if single_form and not basename.startswith(os.path.basename(tree.root) + "@"):
                    continue
                directory = os.path.dirname(logical)
                parsed = _parse_legacy(basename, tokens)
                canonical = False
                if parsed is None and not single_form:
                    # Kept apart from the 0.3.x parser: a file at its 0.4.0 name (last
                    # '@') counts only when it is the target of a discovered source.
                    parsed = _parse_canonical(basename, names)
                    canonical = parsed is not None and (
                        os.path.normcase(os.path.join(directory, parsed[0])) in bases
                    )
                    if not canonical:
                        parsed = None
                if parsed is None:
                    if not single_form and tree.kind == "derivatives":
                        tree.unrecognized.append(path)
                    continue
                source_part, token, tail = parsed
                ukey = (directory, source_part, token, canonical)
                unit = units.get(ukey)
                if unit is None:
                    old_dir = (
                        None
                        if single_form
                        else os.path.relpath(directory, tree.root).replace(os.sep, "/")
                    )
                    derivative = token if canonical else plan.tokens[token]
                    unit = Unit(key, directory, old_dir, source_part, token, derivative)
                    unit.canonical = canonical
                    units[ukey] = unit
                if logical != path:
                    unit.logical[path] = logical
                if tail in _MARKER_TAILS:
                    unit.markers[_MARKER_TAILS[tail]] = path
                else:
                    unit.artifacts[tail] = [path]
        for unit in units.values():
            _group_split_parts(unit, tree)
            _match_unit(unit, tree, index, upstream, bases)
        tree.units = [u for u in units.values() if u.artifacts or u.markers]
        if tree.recursive and not tree.units:
            continue  # nothing left behind next to undiscovered sources
        plan.trees.append(tree)
    _resume_from_journals(plan)


def _group_split_parts(unit: Unit, tree: TreePlan) -> None:
    """Attach split continuations to their first part's artifact.

    ``<main>-<n>.fif`` joins ``<main>.fif`` and ``.fif-<n>.gz`` joins ``.fif.gz`` (MNE's
    neuromag naming, the rule discovery uses).
    """
    parts: dict[str, list[tuple[int, str]]] = {}
    for tail in list(unit.artifacts):
        split = _split_tail(tail)
        if split is None:
            continue
        main, index = split
        if main in unit.artifacts or main.lower() in (".fif", ".fif.gz"):
            parts.setdefault(main, []).append((index, unit.artifacts.pop(tail)[0]))
    for main, found in parts.items():
        if main not in unit.artifacts:
            # Continuation parts without their first part: never a usable recording.
            tree.unrecognized.extend(p for _, p in sorted(found))
            continue
        unit.artifacts[main].extend(p for _, p in sorted(found))


def _match_unit(
    unit: Unit,
    tree: TreePlan,
    index: Mapping[str, list[_Candidate]],
    upstream: Mapping[str, str],
    bases: Mapping[str, list[_Candidate]] | None = None,
) -> None:
    """Attach *unit* to its candidate sources and set its status.

    0.3.x units are matched by legacy key (*index*: basename with ``@`` written ``&``);
    units recognized at their 0.4.0 name by their exact reference base (*bases*).
    """
    if unit.canonical:
        found = (bases or {}).get(os.path.normcase(os.path.join(unit.directory, unit.source_part)))
        unit.candidates = list(found or [])
        if len(unit.candidates) == 1:
            unit.candidate = unit.candidates[0]
            unit.new_prefix = output_prefix(unit.candidate.reference_base, unit.derivative)
            unit.status = "identity"
        else:
            unit.status = "ambiguous" if unit.candidates else "orphan"
        return
    parts = [unit.source_part]
    if "&" in unit.source_part and upstream:
        head, _, tail = unit.source_part.rpartition("&")
        token, dot, rest = tail.partition(".")
        if token in upstream:
            parts.append(f"{head}&{upstream[token]}{dot}{rest}")
    if unit.old_dir is None:  # pre-0.3.3 single file: "<derivatives_path>@Deriv.*"
        candidates = [c for group in index.values() for c in group]
    else:
        candidates = []
        for part in parts:
            for cand in index.get(part, []):
                same_dir = os.path.normcase(os.path.dirname(cand.reference_base)) == (
                    os.path.normcase(unit.directory)
                )
                suffix = tree.kind == "derivatives" and _is_path_suffix(
                    unit.old_dir, posixpath.dirname(cand.rel)
                )
                if same_dir or suffix:
                    candidates.append(cand)
    unit.candidates = candidates
    if len(candidates) == 1:
        unit.candidate = candidates[0]
        unit.new_prefix = output_prefix(candidates[0].reference_base, unit.derivative)
        unit.status = "identity" if unit.new_prefix == unit.old_prefix else "move"
    elif not candidates:
        unit.status = "unresolved" if _looks_chained(unit.source_part) else "orphan"
    else:
        unit.status = "ambiguous"


def _resume_from_journals(plan: MigrationPlan) -> None:
    """State an interrupted ``--apply`` left in the journals.

    The last ``mne-*`` entry of each MNE rewrite transaction (``mne-start``,
    ``mne-staged``, ``mne-done``, ``mne-failed``) and the duplicates recorded with their
    content fingerprints (``duplicate``), by target prefix.
    """
    for journal in {t.journal for t in plan.trees}:
        for entry in read_journal(journal):
            op = entry.get("op")
            if op == "duplicate":
                prefix, files = entry.get("new_prefix"), entry.get("files")
                if isinstance(prefix, str) and isinstance(files, dict):
                    key = os.path.normcase(os.path.normpath(prefix))
                    known = plan._duplicates.setdefault(key, {})
                    for path, fingerprint in files.items():
                        known[os.path.normcase(os.path.normpath(path))] = fingerprint
                continue
            if op in ("mne-start", "mne-staged", "mne-done", "mne-failed") and isinstance(
                entry.get("txn"), str
            ):
                plan._mne_txns[entry["txn"]] = entry


def _plan_artifact_sets(plan: MigrationPlan) -> None:
    """Most common suffix set per derivative among its mapped units (spec 10.4)."""
    sets: dict[str, Counter] = {}
    for unit in _primaries(plan):
        artifacts, markers = _group_view(unit)
        if "error" in markers or "skip" in markers or not artifacts:
            continue
        sets.setdefault(unit.derivative, Counter())[tuple(sorted(artifacts))] += 1
    for name, counter in sorted(sets.items()):
        ranked = counter.most_common()
        known = list(ranked[0][0])
        tie = len(ranked) > 1 and ranked[1][1] == ranked[0][1]
        plan.artifact_sets[name] = {
            "known": None if tie else known,
            "counts": [{"suffixes": list(s), "units": n} for s, n in ranked],
        }
        if tie:
            plan.notes.append(
                f"{name}: no single most common artifact set ({', '.join(' '.join(s) for s, _ in ranked[:2])} "
                "tie); its units get no .done and are recomputed."
            )


def _primaries(plan: MigrationPlan) -> list[Unit]:
    """One unit per target: the first unit of each group of units sharing a target."""
    return [u for u in plan.units("move", "identity") if not u.secondary]


def _group_view(unit: Unit) -> tuple[dict[str, list[str]], dict[str, str]]:
    """Artifacts and markers of a unit's whole group (old paths)."""
    artifacts: dict[str, list[str]] = {}
    markers: dict[str, str] = {}
    for member in unit.members or [unit]:
        artifacts.update(member.artifacts)
        markers.update(member.markers)
    return artifacts, markers


def _staged_txn(plan: MigrationPlan, old_first: str, target: str) -> dict[str, Any] | None:
    """The ``mne-staged`` journal entry of the rewrite ``old_first -> target``, if open."""
    entry = plan._mne_txns.get(_txn_id("mne", old_first, target))
    return entry if entry is not None and entry.get("op") == "mne-staged" else None


def _strip_resumed_mne(plan: MigrationPlan, group: list[Unit]) -> list[Unit]:
    """Drop the parts an interrupted rewrite already published from in-place units.

    Only a rewrite whose verified copy was staged (journal ``mne-staged``) publishes
    anything; the published files are checked against its fingerprints when it resumes.
    """
    in_progress: set[str] = set()
    for unit in group:
        if unit.status != "move":
            continue
        for tail, paths in unit.artifacts.items():
            if _staged_txn(plan, paths[0], (unit.new_prefix or "") + tail) is not None:
                in_progress.add(tail)
    if not in_progress:
        return group
    kept = []
    for unit in group:
        if unit.status == "identity":
            for tail in in_progress:
                unit.artifacts.pop(tail, None)
            if not unit.artifacts and not unit.markers:
                unit.status = "resumed"
                continue
        kept.append(unit)
    return kept


def _plan_targets(plan: MigrationPlan) -> None:
    """Group units by target, resolve duplicates, compute per-file moves and MNE rewrites."""
    groups: dict[str, list[Unit]] = {}
    for unit in plan.units("move", "identity"):
        groups.setdefault(os.path.normcase(unit.new_prefix), []).append(unit)
    for group in groups.values():
        if len(group) > 1:
            group = _strip_resumed_mne(plan, _drop_link_leftovers(group))
        if len(group) == 1:
            continue
        tails = [u.tails for u in group]
        if sum(len(t) for t in tails) == len(set().union(*tails)):
            # Disjoint files (a unit half moved by an interrupted run): one unit, one manifest.
            group.sort(key=lambda u: u.status != "identity")
            group[0].members = list(group)
            for other in group[1:]:
                other.secondary = True
            continue
        identity = [u for u in group if u.status == "identity"]
        if identity:
            keeper = identity[0]
        else:
            ranked = sorted(group, key=lambda u: len(u.old_dir or ""), reverse=True)
            keeper = ranked[0]
            if len(ranked[0].old_dir or "") == len(ranked[1].old_dir or "") and not all(
                _same_files(keeper, u) for u in group[1:]
            ):
                keeper = None  # type: ignore[assignment]
        others = [u for u in group if u is not keeper]
        if keeper is not None and all(
            _same_files(keeper, u) or _recorded_duplicate(plan, keeper, u) for u in others
        ):
            for u in others:
                u.status = "duplicate"
                u.duplicate_of = keeper.old_prefix
            continue
        for u in group:
            u.status = "conflict"
        plan.refusals.append(
            f"{len(group)} sets of 0.3.x outputs map to the same new outputs "
            f"{group[0].new_prefix}* with different content: "
            + ", ".join(u.label for u in group)
            + ". Keep one (delete or move the others away) and run migrate again."
        )

    all_old = {os.path.normcase(p) for u in plan.units("move") for p in u.files}
    staging_roots = {
        id(u): os.path.join(os.path.dirname(t.journal), "staging")
        for t in plan.trees
        for u in t.units
    }
    for unit in plan.units("move"):
        new_prefix = unit.new_prefix or ""
        for tail, paths in unit.artifacts.items():
            target = new_prefix + tail
            first = os.path.normpath(paths[0])
            txn = _txn_id("mne", first, target)
            staging = os.path.join(staging_roots[id(unit)], txn)
            staged = _staged_txn(plan, first, target) if _is_fif_tail(tail) else None
            if staged is not None:
                rewrite = MneRewrite(
                    tail,
                    [str(p) for p in staged.get("old_parts") or paths],
                    target,
                    txn,
                    staging,
                    [list(p) for p in staged.get("parts") or []],
                )
                if _plan_staged(plan, rewrite):
                    unit.mne.append(rewrite)
                    continue
            if len(paths) > 1 and os.path.basename(target) == os.path.basename(paths[0]):
                _plan_split_move(plan, unit, paths, target, all_old)
                continue
            if len(paths) > 1:  # MNE split recording (main + "-<n>.fif" / ".fif-<n>.gz")
                occupied = [p for p in _split_names(target, len(paths) + 2) if os.path.lexists(p)]
                aliases = [p for p in occupied if any(_same_entry(p, o) for o in paths)]
                if aliases:
                    plan.refusals.append(
                        f"the new name of the MNE split recording {paths[0]} differs only in "
                        f"case and this filesystem ignores case ({aliases[0]} is the old "
                        "file): rename it with a --map name that differs beyond case."
                    )
                elif occupied:
                    plan.refusals.append(
                        f"targets of the MNE split recording {paths[0]} already exist: "
                        f"{', '.join(occupied)}. Remove them (or the old recording) first."
                    )
                unit.mne.append(MneRewrite(tail, list(paths), target, txn, staging))
                continue
            _add_move(plan, unit, paths[0], target, all_old)
        for marker, old in unit.markers.items():
            if marker == "done":
                plan.notes.append(f"left in place (0.4.0 manifest at an old location): {old}")
                continue
            _add_move(plan, unit, old, f"{new_prefix}.{marker}", all_old)


def _plan_split_move(
    plan: MigrationPlan, unit: Unit, paths: list[str], target: str, all_old: set[str]
) -> None:
    """A split recording whose file names do not change (only the folder): moved part by
    part (the parts reference each other by base name).

    The transaction is keyed by the original first part, and its part list is the one
    journaled by the first attempt, so any number of interrupted runs rejoin the same
    recording (parts already moved are scanned at their original location).
    """
    logical = [unit.logical.get(p, p) for p in paths]
    txn = _txn_id("split", logical[0], target)
    journaled = plan._split_txns.get(txn)
    if journaled is not None:
        pairs = [(str(o), str(n)) for o, n in journaled["parts"]]
    else:
        pairs = [(o, os.path.join(os.path.dirname(target), os.path.basename(o))) for o in logical]
    for old in paths:
        new = os.path.join(os.path.dirname(target), os.path.basename(old))
        _add_move(plan, unit, old, new, all_old)
    unit.split_moves.append(
        {
            "txn": txn,
            "parts": pairs,
            "kind": journaled.get("kind") if journaled else None,
            "shape": journaled.get("shape") if journaled else None,
            "journaled": journaled is not None,
        }
    )


def _plan_staged(plan: MigrationPlan, rewrite: MneRewrite) -> bool:
    """Check a resumed rewrite's staged copy and published parts; ``False``: start over.

    A file at a new name belongs to this transaction only when it *is* its staged file
    (hard link) or matches the size and SHA-256 recorded in the journal; anything else is
    refused, never overwritten.
    """
    if not rewrite.staged:
        return False
    directory = os.path.dirname(rewrite.new_path)
    missing = False
    for name, size, digest in rewrite.staged:
        dest, staged = os.path.join(directory, name), os.path.join(rewrite.staging, name)
        if os.path.lexists(dest):
            if not (
                (os.path.lexists(staged) and _same_entry(dest, staged))
                or _matches(dest, size, digest)
            ):
                plan.refusals.append(
                    f"{dest} exists and is not the verified new recording of "
                    f"{rewrite.old_parts[0]} (kept in {rewrite.staging}); move it away and "
                    "run migrate again."
                )
        elif not _matches(staged, size, digest):
            missing = True
    if missing:
        if all(os.path.isfile(p) for p in rewrite.old_parts) and not any(
            os.path.lexists(os.path.join(directory, n)) for n, _, _ in rewrite.staged
        ):
            return False  # nothing published and the staged copy is gone: redo it
        plan.refusals.append(
            f"the verified new recording of {rewrite.old_parts[0]} is incomplete in "
            f"{rewrite.staging} and in {directory}; restore the old recording or remove "
            "the partial new parts by hand."
        )
    return True


def _file_digest(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _fingerprint(path: str) -> list[Any]:
    return [os.path.getsize(path), _file_digest(path)]


def _matches(path: str, size: Any, digest: Any) -> bool:
    """*path* is a regular file with this size and SHA-256."""
    try:
        if not os.path.isfile(path) or os.path.getsize(path) != size:
            return False
        return _file_digest(path) == digest
    except OSError:
        return False


def _drop_link_leftovers(group: list[Unit]) -> list[Unit]:
    """Undo the double count of a move interrupted between ``os.link`` and ``os.unlink``.

    The file then has its old and its new name (one inode). The new name is dropped from
    the in-place unit; the old unit's move finishes by removing the old name.
    """
    inodes: set[tuple[int, int]] = set()
    for unit in group:
        if unit.status == "move":
            for path in unit.files:
                try:
                    st = os.stat(path)
                except OSError:
                    continue
                inodes.add((st.st_dev, st.st_ino))
    if not inodes:
        return group
    kept = []
    for unit in group:
        if unit.status == "identity":
            for tail, paths in list(unit.artifacts.items()):
                try:
                    st = os.stat(paths[0])
                except OSError:
                    continue
                if len(paths) == 1 and (st.st_dev, st.st_ino) in inodes and st.st_nlink > 1:
                    del unit.artifacts[tail]
            for marker, path in list(unit.markers.items()):
                try:
                    st = os.stat(path)
                except OSError:
                    continue
                if (st.st_dev, st.st_ino) in inodes and st.st_nlink > 1:
                    del unit.markers[marker]
            if not unit.artifacts and not unit.markers:
                unit.status = "resumed"
                continue
        kept.append(unit)
    return kept


def _recorded_duplicate(plan: MigrationPlan, keeper: Unit, other: Unit) -> bool:
    """*other* was found identical to *keeper* by an earlier run (journal ``duplicate``).

    The keeper may have been re-saved since (MNE rewrite), so its bytes cannot be compared
    again; *other*'s files must still match the fingerprints recorded then.
    """
    known = plan._duplicates.get(os.path.normcase(os.path.normpath(keeper.new_prefix or "")))
    if not known or keeper.status != "identity":
        return False
    files = other.files
    for path in files:
        fingerprint = known.get(os.path.normcase(os.path.normpath(path)))
        if not isinstance(fingerprint, list) or len(fingerprint) != 2:
            return False
        if not _matches(path, *fingerprint):
            return False
    return bool(files)


def _add_move(plan: MigrationPlan, unit: Unit, old: str, new: str, all_old: set[str]) -> None:
    unit.targets[old] = new
    if os.path.normcase(old) == os.path.normcase(new):
        return
    if os.path.normcase(new) in all_old:
        plan.refusals.append(
            f"{new} is both a move target and a 0.3.x file to be moved; resolve by hand."
        )
        return
    if os.path.lexists(new):
        if _same_entry(old, new):
            # One file: a move interrupted between link and unlink, or a case-only
            # rename on a case-insensitive filesystem; the move finishes it.
            unit.moves.append((old, new))
            return
        try:
            same = filecmp.cmp(old, new, shallow=False)
        except OSError:
            same = False
        if same:
            unit.duplicate_files.append(old)
            unit.duplicate_of = new
            return
        plan.refusals.append(
            f"{new} already exists with different content than {old}; keep one of them "
            "and run migrate again."
        )
        return
    unit.moves.append((old, new))


def _plan_manifests(plan: MigrationPlan) -> None:
    for unit in _primaries(plan):
        artifacts, markers = _group_view(unit)
        unit.manifest_suffixes = None
        if not plan.synthesize:
            unit.manifest_reason = "--no-synthesize"
            continue
        if unit_state(unit.new_prefix or "", unit.derivative).state in (
            "done",
            "skipped",
        ):
            continue  # already complete (resumed or migrated before)
        if "error" in markers:
            unit.manifest_reason = "has an .error marker"
            continue
        if "skip" in markers:
            continue  # .skip is its own state
        info = plan.artifact_sets.get(unit.derivative)
        known = info.get("known") if info else None
        if not known:
            unit.manifest_reason = "artifact set of the derivative is unknown"
            continue
        missing = [s for s in known if s not in artifacts]
        if missing:
            unit.manifest_reason = f"missing artifact(s) {' '.join(missing)} (partial unit)"
            continue
        # The common set only decides completeness; the manifest lists every artifact of
        # the unit (a unit with an extra artifact keeps it, spec 6).
        unit.manifest_suffixes = list(known) + sorted(s for s in artifacts if s not in known)
        unit.manifest_reason = None


def _plan_safety(plan: MigrationPlan) -> None:
    """Refuse unwritable folders and moves across filesystems (spec 10.5)."""
    unwritable: set[str] = set()
    cross: list[str] = []
    for unit in plan.units("move"):
        pairs = list(unit.moves) + [(r.old_parts[0], r.new_path) for r in unit.mne]
        for old, new in pairs:
            src_dir = os.path.dirname(old)
            dst = _existing_ancestor(os.path.dirname(new))
            for directory in (src_dir, dst):
                if not os.access(directory, os.W_OK):
                    unwritable.add(directory)
            try:
                if os.stat(old).st_dev != os.stat(dst).st_dev:
                    cross.append(f"{old} -> {new}")
            except OSError:
                pass
    for tree in plan.trees:
        if tree.kind == "derivatives" and any(u.status == "move" for u in tree.units):
            root = _existing_ancestor(tree.root)
            if not os.access(root, os.W_OK):
                unwritable.add(root)
    for directory in sorted(unwritable):
        plan.refusals.append(f"folder {directory} is not writable.")
    if cross:
        plan.refusals.append(
            f"{len(cross)} move(s) cross filesystems (os.replace cannot move them), e.g. "
            f"{cross[0]}. Copy the tree onto one filesystem first."
        )


def _plan_chained(
    plan: MigrationPlan,
    config: Mapping[str, Any],
    datasets: Mapping[str, Any],
    source: _Source,
) -> None:
    """Patterns referencing 0.3.x output names: printed for the user, never edited."""
    for name, ds in datasets.items():
        for attr in ("file_pattern", "exclude_pattern"):
            values = ds.get(attr)
            values = values.values() if isinstance(values, Mapping) else [values]
            for value in values:
                if isinstance(value, str) and "@" in value:
                    plan.chained_patterns.append(
                        f"{source.path}: datasets.{name}.{attr} = {value!r} matches neurodags "
                        "outputs; if the upstream pipeline's names change on disk, edit it and "
                        "migrate this pipeline with --upstream-map OLDTOKEN=NEWNAME."
                    )
    changed = {t: n for t, n in plan.tokens.items() if t != n}
    if not changed:
        return
    alternatives = "|".join(re.escape(t) for t in sorted(changed, key=len, reverse=True))
    pattern = re.compile(r"@(" + alternatives + r")(?![A-Za-z0-9_])")
    own = {os.path.normcase(plan.pipeline), os.path.normcase(source.path)}
    folders = {os.path.dirname(plan.pipeline), os.path.dirname(source.path)}
    for folder in sorted(folders):
        try:
            names = sorted(os.listdir(folder))
        except OSError:
            continue
        for fname in names:
            path = os.path.join(folder, fname)
            if not fname.endswith((".yml", ".yaml")) or os.path.normcase(path) in own:
                continue
            try:
                with open(path, encoding="utf-8") as handle:
                    lines = handle.read().splitlines()
            except (OSError, UnicodeDecodeError):
                continue
            for number, line in enumerate(lines, 1):
                match = pattern.search(line)
                if match:
                    plan.chained_patterns.append(
                        f"{path}:{number}: {line.strip()}  (0.3.x '@{match.group(1)}' is "
                        f"'@{changed[match.group(1)]}' after this migration; edit by hand)"
                    )


# ---------------------------------------------------------------------------
# Dataframe column map (spec 10.6)
# ---------------------------------------------------------------------------


_COORD_RULE = (
    "0.3.x: '<name><suffix>@<dim>-<value>_...' with dims sorted by name and values "
    "rewritten lossily ('-'->'~', '_'->'|', '@'->'$', ','->'.', empty->'NA'; dim names "
    "unescaped). 0.4.0: '<name><suffix>@<dim>-<value>_...' with dims and values "
    "percent-encoded (neurodags.naming.format_column) and dims sorted by encoded name. "
    "Best effort only: neurodags.migrate.legacy_column_to_new decodes '~', '|', '$' and "
    "'NA' back, but cannot recover ',' (written as '.') nor dims containing '-' or '_'. "
    "Rebuilding the dataframe from the (unchanged) artifacts is the reliable path."
)


def legacy_column_to_new(
    column: str, names: Iterable[str], renames: Mapping[str, str] | None = None
) -> str | None:
    """Best-effort translation of a 0.3.x dataframe column to its 0.4.0 name (spec 10.6).

    Exact for ``name.suffix`` and ``name__error`` columns; for flattened coordinate
    columns the 0.3.x encoding was lossy (see the module's report text), so the result
    is a best guess.

    Parameters
    ----------
    column : str
        A 0.3.x wide column name (or long-format ``derivative`` value).
    names : iterable of str
        The 0.3.x derivative names (YAML spelling).
    renames : mapping, optional
        ``--map`` renames, old name to new name.

    Returns
    -------
    str or None
        The 0.4.0 column, or ``None`` when *column* does not parse.

    Examples
    --------
    >>> legacy_column_to_new("psd__error", ["psd"])
    'psd.error'
    >>> legacy_column_to_new("psd.nc@band-8~12_ch-C3", ["psd"])
    'psd.nc@band-8-12_ch-C3'
    """
    renames = renames or {}
    try:
        if column.endswith("__error"):
            old = column[: -len("__error")]
            if old not in set(names):
                return None
            return error_column(renames.get(old, old))
        head, sep, coords = column.partition("@")
        for old in sorted(names, key=len, reverse=True):
            if head.startswith(old) and head[len(old) :].startswith("."):
                suffix = head[len(old) :]
                break
        else:
            return None
        pairs: list[tuple[str, str]] = []
        if sep:
            for part in coords.split("_"):
                dim, dash, value = part.partition("-")
                if not dash or not dim:
                    return None
                value = value.replace("~", "-").replace("|", "_").replace("$", "@")
                pairs.append((dim, "" if value == "NA" else value))
        return format_column(renames.get(old, old), suffix, pairs)
    except NamingError:
        return None


def _column_map(
    plan: MigrationPlan, old_to_new: Mapping[str, str], defs: Mapping[Any, Any]
) -> dict[str, Any]:
    suffixes: dict[str, set[str]] = {}
    for unit in plan.units():
        suffixes.setdefault(unit.derivative, set()).update(unit.artifacts)
    derivatives: dict[str, Any] = {}
    for key, definition in defs.items():
        old = str(key)
        new = old_to_new.get(old, old) if plan.mode == "legacy" else old
        olds = [o for o, n in old_to_new.items() if n == new] or [old]
        entry: dict[str, Any] = {
            "old_name": olds[0],
            "for_dataframe": bool(
                isinstance(definition, Mapping) and definition.get("for_dataframe", False)
            ),
            "columns": {f"{olds[0]}{s}": f"{new}{s}" for s in sorted(suffixes.get(new, set()))},
            "error_column": {"old": f"{olds[0]}__error", "new": f"{new}.error"},
        }
        derivatives[new] = entry
    return {
        "exact": "name.suffix columns and name__error -> name.error",
        "derivatives": derivatives,
        "coordinates": {"exact": False, "rule": _COORD_RULE},
    }


# ---------------------------------------------------------------------------
# Preview
# ---------------------------------------------------------------------------


def _rel(path: str, root: str) -> str:
    try:
        rel = os.path.relpath(path, root)
    except ValueError:
        return path
    return path if rel.startswith("..") else rel


def _listed(lines: list[str], items: Sequence[str], indent: str = "    ") -> None:
    for item in items[:_MAX_LISTED]:
        lines.append(f"{indent}{item}")
    if len(items) > _MAX_LISTED:
        lines.append(f"{indent}... and {len(items) - _MAX_LISTED} more (see --report)")


def format_preview(plan: MigrationPlan, *, apply: bool = False) -> str:
    """Human-readable preview of *plan* (what ``neurodags migrate`` prints)."""
    lines: list[str] = []
    title = "neurodags migrate: plan" + (
        "" if apply else " (preview: nothing is changed; add --apply to execute)"
    )
    lines += [title, ""]
    lines.append(f"pipeline:    {plan.pipeline}")
    if plan.datasets_file:
        lines.append(f"datasets:    {plan.datasets_file}")
    lines.append(f"mount:       {plan.mount if plan.mount is not None else '(none)'}")
    if plan.mode == "legacy":
        lines.append(
            f"legacy cwd:  {plan.legacy_cwd}  (0.3.x resolved relative dataset paths here; "
            "--legacy-cwd)"
        )
    else:
        lines.append("mode:        configuration already at naming_version 1 (output tree only)")
    lines.append("")

    lines.append("Configuration")
    if not plan.config_changes:
        lines.append("  no changes")
    for change in plan.config_changes:
        lines.append(f"  {change.path}  (backup: {os.path.basename(change.backup)})")
        _listed(lines, change.changes)
        if not change.comments_preserved:
            lines.append(
                "    note: cannot be edited in place; it will be re-serialized and its "
                "comments/formatting are lost (the backup keeps them)"
            )
    if plan.source_roots:
        lines.append("  source_root (confirm; override with --source-root DATASET[:MOUNT]=PATH):")
        for name, mounts in plan.source_roots.items():
            for mount_name, value in mounts.items():
                where = "" if mount_name == "*" else f" [{mount_name}]"
                lines.append(f"    {name}{where}: {value}")
    changed_tokens = sorted((t, n) for t, n in plan.tokens.items() if t != n)
    if changed_tokens:
        lines.append("  on-disk names (0.3.x token -> 0.4.0 verbatim name):")
        _listed(lines, [f"@{t} -> @{n}" for t, n in changed_tokens])
    lines.append("")

    for tree in plan.trees:
        moves = [m for u in tree.units if u.status == "move" for m in u.moves]
        mne = [r for u in tree.units if u.status == "move" for r in u.mne]
        manifests = [
            u for u in tree.units if u.status in ("move", "identity") and u.manifest_suffixes
        ]
        kind = "output tree" if tree.kind == "derivatives" else "next to sources"
        lines.append(f"{kind.capitalize()}: {tree.root}  (datasets: {', '.join(tree.datasets)})")
        counts = Counter(u.status for u in tree.units)
        lines.append(
            f"  units: {counts['move']} to move, {counts['identity']} already in place, "
            f"{counts['orphan']} orphan, {counts['unresolved']} unresolved, "
            f"{counts['ambiguous']} ambiguous, "
            f"{counts['duplicate']} duplicate; {len(tree.unrecognized)} unrecognized file(s)"
        )
        lines.append(
            f"  {len(moves)} file move(s), {len(mne)} MNE split recording(s) to rewrite, "
            f"{len(manifests)} .done manifest(s) to write"
        )
        if moves:
            lines.append("  moves:")
            _listed(lines, [f"{_rel(o, tree.root)} -> {_rel(n, tree.root)}" for o, n in moves])
        if mne:
            lines.append(
                "  MNE split recordings (re-saved with MNE, verified, then old parts deleted):"
            )
            _listed(
                lines,
                [
                    f"{_rel(r.old_parts[0], tree.root)} (+{len(r.old_parts) - 1} part(s)) -> "
                    f"{_rel(r.new_path, tree.root)}"
                    + (" [resume: publish the verified copy]" if r.staged is not None else "")
                    for r in mne
                ],
            )
        orphans = [f for u in tree.units if u.status == "orphan" for f in u.files]
        if orphans:
            lines.append("  orphans (no matching source; left in place):")
            _listed(lines, [_rel(f, tree.root) for f in orphans])
        unresolved = [f for u in tree.units if u.status == "unresolved" for f in u.files]
        if unresolved:
            lines.append("  unresolved chained outputs (no matching upstream source; REFUSED):")
            _listed(lines, [_rel(f, tree.root) for f in unresolved])
        ambiguous = [u for u in tree.units if u.status == "ambiguous"]
        if ambiguous:
            lines.append("  ambiguous (several sources match; REFUSED):")
            _listed(
                lines,
                [
                    f"{_rel(u.label, tree.root)} -> "
                    + ", ".join(f"{c.dataset}: {c.rel}" for c in u.candidates[:4])
                    + (" ..." if len(u.candidates) > 4 else "")
                    for u in ambiguous
                ],
            )
        dups = [u for u in tree.units if u.status == "duplicate" or u.duplicate_files]
        if dups:
            lines.append(
                "  duplicates (identical to the target; left in place, delete when happy):"
            )
            _listed(
                lines,
                [_rel(f, tree.root) for u in dups for f in (u.duplicate_files or u.files)],
            )
        lines.append("")

    if plan.artifact_sets:
        lines.append("Artifact sets (most common per derivative; confirm):")
        for name, info in plan.artifact_sets.items():
            counts = "; ".join(
                f"{' '.join(c['suffixes'])} x{c['units']}" for c in info["counts"][:3]
            )
            known = " ".join(info["known"]) if info["known"] else "unknown (tie)"
            lines.append(f"  {name}: {known}   [{counts}]")
        skipped = [u for u in plan.units("move", "identity") if u.manifest_reason]
        if skipped:
            reasons = Counter(u.manifest_reason for u in skipped)
            lines.append(
                "  without .done (status 'incomplete' or 'errored', recomputed by the next run): "
                + "; ".join(f"{n} {r}" for r, n in reasons.most_common())
            )
        lines.append("")
    if plan.chained_patterns:
        lines.append("Chained pipelines (edit these by hand; migrate never changes them):")
        _listed(lines, plan.chained_patterns, indent="  ")
        lines.append("")
    if plan.notes:
        lines.append("Notes:")
        _listed(lines, plan.notes, indent="  - ")
        lines.append("")
    if plan.refusals:
        lines.append("REFUSED (nothing will be applied):")
        for refusal in plan.refusals:
            lines.append(f"  - {refusal}")
    elif not plan.has_work:
        lines.append("Nothing to migrate: the configuration and the output tree are up to date.")
    else:
        lines.append(
            f"Summary: {len(plan.moves)} move(s), {len(plan.mne_rewrites)} MNE rewrite(s), "
            f"{len(plan.manifests)} manifest(s), {len(plan.config_changes)} configuration "
            f"file(s)."
        )
        if not apply:
            lines.append(
                "Run again with --apply to execute (use --report plan.json to keep the plan)."
            )
    return "\n".join(lines)


def write_report(
    plan: MigrationPlan, path: str | os.PathLike, extra: Mapping[str, Any] | None = None
) -> str:
    """Write the ``--report`` JSON document (plan, column map, and *extra* apply results)."""
    data = plan.to_dict()
    if extra:
        data.update(extra)
    target = os.fspath(path)
    directory = os.path.dirname(os.path.abspath(target))
    os.makedirs(directory, exist_ok=True)
    with open(target, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return target


# ---------------------------------------------------------------------------
# Apply
# ---------------------------------------------------------------------------


@dataclass
class ApplyResult:
    """What :func:`apply_migration` did."""

    moved: list[tuple[str, str]] = field(default_factory=list)
    rewritten: list[dict[str, Any]] = field(default_factory=list)
    manifests: list[str] = field(default_factory=list)
    recompute: list[dict[str, str]] = field(default_factory=list)
    configs: list[str] = field(default_factory=list)
    journals: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "applied": {
                "moved": [{"old": o, "new": n} for o, n in self.moved],
                "mne_rewrites": self.rewritten,
                "manifests": self.manifests,
                "recompute": self.recompute,
                "configs": self.configs,
                "journals": self.journals,
            }
        }


class _Journal:
    """Append-only JSON Lines record of completed operations (resume after interruption)."""

    def __init__(self, path: str, plan: MigrationPlan):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
        self.handle = open(path, "a", encoding="utf-8")
        from neurodags import __version__

        self.write(
            {
                "op": "start",
                "pipeline": plan.pipeline,
                "mount": plan.mount,
                "neurodags": __version__,
            }
        )

    def write(self, entry: dict[str, Any]) -> None:
        entry = {**entry, "time": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
        self.handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self.handle.flush()

    def sync(self) -> None:
        self.handle.flush()
        os.fsync(self.handle.fileno())

    def close(self) -> None:
        self.write({"op": "end"})
        self.sync()
        self.handle.close()


def _occupied(old: str, new: str) -> bool:
    """*new* exists and is not *old* itself (a case-only rename on a case-insensitive FS)."""
    if not os.path.lexists(new):
        return False
    return not _same_entry(old, new)


def _recheck(plan: MigrationPlan) -> list[str]:
    """Preconditions of every planned operation, right before applying (spec 10.5)."""
    problems: list[str] = []
    for unit in plan.units("move"):
        for old, new in unit.moves:
            if not os.path.lexists(old):
                problems.append(f"{old} no longer exists")
            elif _occupied(old, new):
                problems.append(f"{new} appeared since the plan was made")
        for rewrite in unit.mne:
            if rewrite.staged is not None:
                continue  # resume: checked against its fingerprints when published
            missing = [p for p in rewrite.old_parts if not os.path.isfile(p)]
            if missing:
                problems.append(f"MNE split part(s) {', '.join(missing)} no longer exist")
            for new in _split_names(rewrite.new_path, len(rewrite.old_parts) + 2):
                if os.path.lexists(new):
                    problems.append(f"{new} appeared since the plan was made")
    for change in plan.config_changes:
        if _read_text(change.path) != change.original:
            problems.append(f"{change.path} changed since the plan was made")
    return problems


# -- lock ---------------------------------------------------------------------------


def _hostname() -> str:
    try:
        return socket.gethostname()
    except OSError:
        return ""


def _acquire_lock(directory: str) -> tuple[str, str]:
    """Create ``<directory>/migrate.lock`` exclusively; refuse when it exists.

    Serializes ``--apply`` runs on one output tree. A lock is never taken over: a lock
    left by a killed run must be deleted by hand (the refusal says so). Returns the lock
    path and this run's token (only a lock with this token is released).
    """
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, LOCK_NAME)
    token = uuid.uuid4().hex
    info = {
        "pid": os.getpid(),
        "host": _hostname(),
        "time": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "token": token,
    }
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        holder = _read_lock(path)
        who = (
            f"pid {holder.get('pid')} on host {holder.get('host') or '?'}, started "
            f"{holder.get('time') or '?'}"
            if holder
            else "holder unknown"
        )
        raise MigrationRefused(
            [
                f"another 'neurodags migrate --apply' holds the lock {path} ({who}). Wait "
                "for it to finish. If no migrate is running on this output tree (it was "
                "killed, or the machine restarted), delete that file and run again; the "
                "lock is never removed automatically."
            ]
        ) from None
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(info, handle)
        handle.flush()
        os.fsync(handle.fileno())
    return path, token


def _read_lock(path: str) -> dict[str, Any] | None:
    try:
        with open(path, encoding="utf-8") as handle:
            holder = json.load(handle)
    except (OSError, ValueError):
        return None
    return holder if isinstance(holder, dict) else None


def _release_lock(lock: tuple[str, str]) -> None:
    """Remove the lock only when it is still this run's (same token)."""
    path, token = lock
    holder = _read_lock(path)
    if holder is None or holder.get("token") != token:
        log.warning("Migration lock is not this run's; left in place", lock=path)
        return
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass


def _lock_dirs(plan: MigrationPlan) -> list[str]:
    """Folders of every tree's lock (an output tree that does not exist yet has none)."""
    dirs = []
    for tree in plan.trees:
        if tree.kind == "derivatives" and not os.path.isdir(tree.root):
            continue
        dirs.append(os.path.dirname(tree.journal))
    return list(dict.fromkeys(dirs))


# -- publication without replacing ----------------------------------------------------

_NO_HARDLINK = {
    errno.EPERM,
    errno.EXDEV,
    errno.EMLINK,
    errno.ENOSYS,
    getattr(errno, "ENOTSUP", errno.EOPNOTSUPP),
    errno.EOPNOTSUPP,
}


def _link(src: str, dst: str) -> bool:
    """Hard-link *src* as *dst* (``FileExistsError`` if *dst* exists); ``False`` when the
    filesystem has no hard links."""
    try:
        os.link(src, dst, follow_symlinks=False)
    except FileExistsError:
        raise
    except OSError as exc:
        if exc.errno in _NO_HARDLINK:
            return False
        raise
    return True


def _claim(path: str) -> None:
    """Create *path* empty and exclusively (``FileExistsError`` if it exists)."""
    os.close(os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600))


def _publish_move(old: str, new: str) -> None:
    """Move *old* to *new*; never replaces an existing *new* (``FileExistsError``).

    ``os.link`` + ``os.unlink`` (an interruption in between leaves both names of one
    file, which the next plan finishes); without hard links, *new* is claimed with
    ``O_EXCL`` and only then replaced by *old*.
    """
    if os.path.lexists(new):
        if not _same_entry(old, new):
            raise FileExistsError(errno.EEXIST, "target exists", new)
        if _two_names(old, new):
            os.unlink(old)  # an interrupted link/unlink: finish it
        else:
            os.rename(old, new)  # one entry, case-only rename (case-insensitive FS)
        return
    if _link(old, new):
        os.unlink(old)
        return
    _claim(new)
    os.replace(old, new)


def _publish_copy(staged: str, dest: str) -> None:
    """Publish a staged file at *dest* without replacing anything (``FileExistsError``)."""
    if _link(staged, dest):
        os.unlink(staged)
        return
    _claim(dest)
    try:
        os.replace(staged, dest)
    except OSError as exc:
        if exc.errno != errno.EXDEV:
            raise
        shutil.copyfile(staged, dest)  # into our own claimed (empty) file
        with open(dest, "rb") as handle:
            os.fsync(handle.fileno())
        os.unlink(staged)


# -- MNE recordings ----------------------------------------------------------------------


def _read_fif(path: str, preload: bool = True) -> tuple[str, Any]:
    import mne

    try:
        return "raw", mne.io.read_raw_fif(path, preload=preload, verbose="error")
    except Exception as raw_exc:  # not a raw file: try epochs
        try:
            return "epochs", mne.read_epochs(path, preload=preload, verbose="error")
        except Exception as epochs_exc:
            raise RuntimeError(
                f"cannot read {path} as Raw ({raw_exc}) nor as Epochs ({epochs_exc})"
            ) from epochs_exc


def _shape(kind: str, recording: Any) -> list[int]:
    if kind == "raw":
        return [len(recording.ch_names), int(recording.n_times)]
    return [len(recording.events), len(recording.ch_names), len(recording.times)]


def _recording_shape(path: str) -> tuple[str, list[int]]:
    """Kind and data shape from the recording's headers (no data is loaded)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kind, recording = _read_fif(path, preload=False)
    return kind, _shape(kind, recording)


def _next_part(path: str) -> str | None:
    """The next split part a FIF file refers to (``None``: none, or not a readable FIF)."""
    try:
        from pathlib import Path

        from mne._fiff.open import _get_next_fname, fiff_open

        fid, tree, _ = fiff_open(Path(path), verbose="error")
        with fid:
            found = _get_next_fname(fid, Path(path), tree)
    except Exception:
        return None
    return os.fspath(found) if found is not None else None


def _verify_recording(files: Sequence[str], expected: Sequence[Any] | None) -> str | None:
    """Why the FIF artifact *files* is not a complete, loadable recording (``None``: it is).

    A single file must not refer to a next part. A split recording is loaded in full
    (every data buffer is read); a Raw must have read exactly the listed parts, and the
    data shape must equal *expected* (``[kind, shape]`` recorded before it was moved).
    """
    if len(files) == 1:
        following = _next_part(files[0])
        if following is not None:
            return (
                f"{files[0]} is the first part of a split recording whose next part "
                f"{following} is missing"
            )
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kind, recording = _read_fif(files[0], preload=True)
    except Exception as exc:
        return f"split recording {files[0]} does not load: {exc}"
    if kind == "raw":
        listed = {os.path.normcase(os.path.abspath(f)) for f in files}
        read = {os.path.normcase(os.path.abspath(os.fspath(f))) for f in recording.filenames}
        if read != listed:
            return f"split recording {files[0]} consists of other parts than listed"
    if expected:
        want_kind, want_shape = expected
        if kind != want_kind or _shape(kind, recording) != list(want_shape):
            return (
                f"split recording {files[0]} has shape {_shape(kind, recording)}, the "
                f"recording before the move had {list(want_shape)}"
            )
    return None


def _same_recording(kind: str, original: Any, reloaded: Any) -> str | None:
    """``None`` when *reloaded* equals *original* (data and info essentials), else why not."""
    import numpy as np

    checks = [
        ("channel names", original.ch_names == reloaded.ch_names),
        ("channel types", original.get_channel_types() == reloaded.get_channel_types()),
        ("sampling rate", original.info["sfreq"] == reloaded.info["sfreq"]),
        ("bad channels", list(original.info["bads"]) == list(reloaded.info["bads"])),
        ("measurement date", original.info["meas_date"] == reloaded.info["meas_date"]),
    ]
    if kind == "raw":
        checks += [
            ("first sample", original.first_samp == reloaded.first_samp),
            ("number of samples", original.n_times == reloaded.n_times),
            (
                "annotations",
                np.array_equal(original.annotations.onset, reloaded.annotations.onset)
                and np.array_equal(original.annotations.duration, reloaded.annotations.duration)
                and list(original.annotations.description)
                == list(reloaded.annotations.description),
            ),
        ]
    else:
        checks += [
            ("events", np.array_equal(original.events, reloaded.events)),
            ("event ids", original.event_id == reloaded.event_id),
            ("times", np.array_equal(original.times, reloaded.times)),
        ]
    for what, ok in checks:
        if not ok:
            return f"{what} differ"
    a, b = original.get_data(), reloaded.get_data()
    if a.shape != b.shape or not np.array_equal(a, b):
        return "data differ"
    return None


def _empty_dir(directory: str) -> None:
    shutil.rmtree(directory, ignore_errors=True)
    os.makedirs(directory, exist_ok=True)


def _rewrite_mne(old_parts: Sequence[str], staged_target: str) -> tuple[list[str] | None, str]:
    """Re-save a recording at *staged_target* (a private staging folder) and verify it.

    MNE's own split layout is used (its default ``split_size``): the original split size
    is not stored in the files, and MNE reserves room in every part, so a size inferred
    from the part lengths would multiply the parts. The reloaded copy must equal the
    original sample for sample. ``(parts, "")`` or ``(None, why)``.
    """
    from neurodags.writers import save_mne

    staging = os.path.dirname(staged_target)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            kind, original = _read_fif(old_parts[0])
        except Exception as exc:
            return None, str(exc)
        if kind == "raw":
            # The original on-disk sample format; "single" when MNE reports another one.
            fmt = original.orig_format
            fmts = [fmt if fmt in ("single", "double", "int", "short") else "single"]
        else:
            fmts = ["single", "double"]  # epochs keep no orig_format: compare, then double
        reason = "no format reproduced the data"
        for fmt in fmts:
            _empty_dir(staging)
            try:
                written = save_mne(
                    original, staged_target, fmt=fmt, overwrite=True, verbose="error"
                )
                _, reloaded = _read_fif(staged_target)
                problem = _same_recording(kind, original, reloaded)
                del reloaded
            except Exception as exc:
                problem = f"saving with fmt={fmt!r} failed: {exc}"
                written = []
            if problem is None:
                outside = [p for p in written if os.path.dirname(os.path.abspath(p)) != staging]
                if outside:
                    return None, f"MNE wrote outside the staging folder: {', '.join(outside)}"
                return [os.path.abspath(p) for p in written], ""
            reason = f"fmt={fmt!r}: {problem}"
        _empty_dir(staging)
        return None, reason


def _write_config(change: ConfigChange) -> None:
    if not os.path.exists(change.backup):
        shutil.copy2(change.path, change.backup)
    directory = os.path.dirname(change.path)
    fd, tmp = tempfile.mkstemp(dir=directory, prefix="." + os.path.basename(change.path) + ".")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(change.new_text)
            handle.flush()
            os.fsync(handle.fileno())
        shutil.copymode(change.path, tmp)
        os.replace(tmp, change.path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _appeared(path: str, where: str = "") -> MigrationError:
    return MigrationError(
        f"{path} appeared during the migration and was not overwritten{where}. Move it away "
        "(or keep it and remove the old output), then plan again (preview) and re-run "
        "--apply to finish."
    )


def _apply_rewrite(
    unit: Unit, rewrite: MneRewrite, journal: _Journal, result: ApplyResult
) -> list[str] | None:
    """Stage, verify and publish one MNE re-save, then delete the old parts.

    1. ``mne-start``; the recording is saved into ``rewrite.staging`` (private to this
       transaction) and reloaded: it must equal the original sample for sample;
    2. ``mne-staged`` with the size and SHA-256 of every staged part;
    3. each part is published at its new name without replacing anything (``os.link``);
       a file already there is this transaction's only when it matches its fingerprint;
    4. only then the old parts are deleted (continuations first), ``mne-done``.
    """
    if rewrite.staged is None:
        journal.write(
            {
                "op": "mne-start",
                "txn": rewrite.txn,
                "old_parts": rewrite.old_parts,
                "new": rewrite.new_path,
            }
        )
        journal.sync()
        staged_target = os.path.join(rewrite.staging, os.path.basename(rewrite.new_path))
        parts, reason = _rewrite_mne(rewrite.old_parts, staged_target)
        if parts is None:
            shutil.rmtree(rewrite.staging, ignore_errors=True)
            journal.write(
                {"op": "mne-failed", "txn": rewrite.txn, "old_parts": rewrite.old_parts}
                | {"reason": reason}
            )
            result.recompute.append(
                {"unit": unit.old_prefix, "reason": f"MNE rewrite not verified: {reason}"}
            )
            log.warning(
                "MNE rewrite failed; unit left in place", unit=unit.old_prefix, reason=reason
            )
            return None
        rewrite.staged = [[os.path.basename(p), *_fingerprint(p)] for p in parts]
        journal.write(
            {
                "op": "mne-staged",
                "txn": rewrite.txn,
                "old_parts": rewrite.old_parts,
                "new": rewrite.new_path,
                "parts": rewrite.staged,
            }
        )
        journal.sync()
    directory = os.path.dirname(rewrite.new_path)
    os.makedirs(directory, exist_ok=True)
    kept = f"; the verified new recording is kept in {rewrite.staging}"
    new_parts = []
    for name, size, digest in rewrite.staged:
        dest, staged = os.path.join(directory, name), os.path.join(rewrite.staging, name)
        new_parts.append(dest)
        if os.path.lexists(dest):
            if os.path.lexists(staged) and _same_entry(staged, dest):
                os.unlink(staged)  # linked by an interrupted run
                continue
            if _matches(dest, size, digest):
                if os.path.lexists(staged):
                    os.unlink(staged)
                continue
            raise _appeared(dest, kept)
        try:
            _publish_copy(staged, dest)
        except FileExistsError:
            raise _appeared(dest, kept) from None
    # Continuation parts first, the first part last: an interruption leaves the first
    # part, so the next plan finds the unit and finishes from the journal.
    for old in rewrite.old_parts[1:] + rewrite.old_parts[:1]:
        try:
            os.unlink(old)
        except FileNotFoundError:
            pass
    journal.write(
        {
            "op": "mne-done",
            "txn": rewrite.txn,
            "old_parts": rewrite.old_parts,
            "new_parts": new_parts,
        }
    )
    shutil.rmtree(rewrite.staging, ignore_errors=True)
    result.rewritten.append(
        {"old_parts": rewrite.old_parts, "new": rewrite.new_path, "new_parts": new_parts}
    )
    return new_parts


def _apply_member(
    unit: Unit,
    journal: _Journal,
    result: ApplyResult,
    new_artifacts: dict[str, list[str]],
    shapes: dict[str, list[Any]],
) -> bool:
    """MNE rewrites then moves of one unit; ``False`` when it is left in place."""
    for rewrite in unit.mne:
        new_parts = _apply_rewrite(unit, rewrite, journal, result)
        if new_parts is None:
            return False
        new_artifacts[rewrite.suffix] = list(new_parts)
    # Split recordings moved part by part: the original part list, destinations and
    # data shape go to the journal (once per transaction) before the first move.
    for split in unit.split_moves:
        if not split["journaled"]:
            first = split["parts"][0][0]
            try:
                split["kind"], split["shape"] = _recording_shape(first)
            except Exception as exc:
                result.recompute.append(
                    {"unit": unit.old_prefix, "reason": f"split recording does not open: {exc}"}
                )
                log.warning("Split recording does not open; unit left in place", unit=first)
                return False
            journal.write(
                {
                    "op": "split-move-start",
                    "txn": split["txn"],
                    "parts": [list(p) for p in split["parts"]],
                    "kind": split["kind"],
                    "shape": split["shape"],
                }
            )
            split["journaled"] = True
    if unit.split_moves:
        journal.sync()
    for old, new in unit.moves:
        os.makedirs(os.path.dirname(new), exist_ok=True)
        try:
            _publish_move(old, new)
        except FileExistsError:
            raise _appeared(new) from None
        journal.write({"op": "move", "old": old, "new": new})
        result.moved.append((old, new))
    for split in unit.split_moves:
        journal.write({"op": "split-move-done", "txn": split["txn"]})
        tail = next(
            (
                t
                for t, ps in unit.artifacts.items()
                if unit.logical.get(ps[0], ps[0]) == split["parts"][0][0]
            ),
            None,
        )
        if tail is not None and split.get("shape") is not None:
            shapes[tail] = [split["kind"], split["shape"]]
    return True


def _unverified_recordings(
    artifacts: Mapping[str, list[str]], verified: set[str], shapes: Mapping[str, list[Any]]
) -> str | None:
    """Why a FIF artifact of a manifest is not a complete recording (``None``: all are).

    Re-saved recordings (*verified*) were compared sample for sample before publication;
    every other FIF artifact is checked by :func:`_verify_recording`.
    """
    for suffix, files in artifacts.items():
        if not _is_fif_tail(suffix) or suffix in verified:
            continue
        problem = _verify_recording(files, shapes.get(suffix))
        if problem is not None:
            return problem
    return None


def _apply_unit(unit: Unit, journal: _Journal, result: ApplyResult) -> None:
    """Apply one unit (with the other units sharing its target), then write its manifest."""
    new_prefix = unit.new_prefix or ""
    new_artifacts: dict[str, list[str]] = {}
    shapes: dict[str, list[Any]] = {}
    complete = True
    for member in unit.members or [unit]:
        if member.status == "move":
            complete = _apply_member(member, journal, result, new_artifacts, shapes) and complete
    if unit.manifest_suffixes is None or not complete:
        return
    owners = {tail: member for member in unit.members or [unit] for tail in member.artifacts}
    artifacts = {}
    for suffix in unit.manifest_suffixes:
        if suffix in new_artifacts:
            artifacts[suffix] = new_artifacts[suffix]
        else:
            owner = owners[suffix]
            artifacts[suffix] = [owner.targets.get(p, p) for p in owner.artifacts[suffix]]
    problem = _unverified_recordings(artifacts, set(new_artifacts), shapes)
    if problem is not None:
        result.recompute.append({"unit": unit.old_prefix, "reason": problem})
        log.warning("Recording not verified; no .done", unit=new_prefix, reason=problem)
        return
    cand = unit.candidate
    manifest = build_manifest(
        new_prefix,
        unit.derivative,
        dataset=cand.dataset if cand else None,
        rel=cand.rel if cand else None,
        artifacts=artifacts,
        origin="migrate",
    )
    path = write_manifest(new_prefix, manifest)
    journal.write({"op": "manifest", "path": path})
    result.manifests.append(path)


def apply_migration(plan: MigrationPlan) -> ApplyResult:
    """Execute *plan*: MNE rewrites, file moves, manifests, then configuration files.

    Refuses (raising :class:`MigrationRefused`, nothing changed) when the plan has
    refusals, another run holds a tree's lock, or a precondition no longer holds. Every
    completed operation is appended to the tree's journal
    (``<derivatives_path>/.neurodags/migrate-journal.json``, JSON Lines; for datasets
    without ``derivatives_path``, ``.neurodags/`` next to the pipeline), so an
    interrupted run is resumed by planning and applying again. The lock file
    (:data:`LOCK_NAME`, next to each journal) of every tree of the plan is held for the
    whole run, also when only configuration files change. Existing files are never
    replaced: MNE re-saves are staged in ``.neurodags/staging/`` and published with
    ``os.link``. Configuration files are written (with ``.bak-0.3`` backups) only after
    every tree was processed.
    """
    if plan.refusals:
        raise MigrationRefused(plan.refusals)
    work = []
    for tree in plan.trees:
        units = [
            u
            for u in tree.units
            if u.status in ("move", "identity")
            and not u.secondary
            and (u.manifest_suffixes is not None or any(m.moves or m.mne for m in u.members or [u]))
        ]
        if units:
            work.append((tree, units))
    result = ApplyResult()
    if not work and not plan.config_changes:
        return result
    # One --apply per output tree at a time: every tree's lock is held from the recheck
    # until the configuration files are written.
    locks: list[tuple[str, str]] = []
    try:
        for directory in _lock_dirs(plan):
            locks.append(_acquire_lock(directory))
        problems = _recheck(plan)
        if problems:
            raise MigrationRefused([*problems, "plan again (preview) and re-run --apply"])
        for tree, units in work:
            journal = _Journal(tree.journal, plan)
            result.journals.append(tree.journal)
            try:
                # Duplicates left in place, with fingerprints: their keeper may be
                # re-saved (other bytes), and a re-run still recognizes them.
                for dup in (u for u in tree.units if u.status == "duplicate"):
                    journal.write(
                        {
                            "op": "duplicate",
                            "new_prefix": dup.new_prefix,
                            "files": {f: _fingerprint(f) for f in dup.files},
                        }
                    )
                journal.sync()
                for unit in units:
                    _apply_unit(unit, journal, result)
            finally:
                journal.close()
        # A separate datasets YAML first, the pipeline (which declares naming_version: 1)
        # last: interrupted in between, the datasets already have source_root, which a
        # re-run in legacy mode accepts as 0.4.0 paths, so the same command completes.
        for change in sorted(plan.config_changes, key=lambda c: c.role == "pipeline"):
            _write_config(change)
            result.configs.append(change.path)
    finally:
        for lock in reversed(locks):
            _release_lock(lock)
    return result


def format_apply_summary(result: ApplyResult) -> str:
    """Summary printed after ``--apply``."""
    lines = [
        "Applied:",
        f"  {len(result.moved)} file(s) moved",
        f"  {len(result.rewritten)} MNE split recording(s) rewritten and verified",
        f"  {len(result.manifests)} .done manifest(s) written (origin: migrate)",
        f"  {len(result.configs)} configuration file(s) rewritten"
        + (f" (backups: *{BACKUP_SUFFIX})" if result.configs else ""),
    ]
    for journal in result.journals:
        lines.append(f"  journal: {journal}")
    if result.recompute:
        lines.append(f"{len(result.recompute)} unit(s) left in place, to be recomputed:")
        for item in result.recompute:
            lines.append(f"  {item['unit']}: {item['reason']}")
    lines.append(
        "Next: 'neurodags status pipeline.yml' (migrated units show as done (migrated)), "
        "then rebuild dataframes."
    )
    return "\n".join(lines)
