"""Where outputs go and when a unit is done: output layout and ``.done`` manifests.

This module is the single place that maps a (source file, derivative) *unit* to files on
disk (spec sections 4 and 6). Every reader and writer (execution, dry-run, ``status``,
dataframe collection, the TUI) goes through it, so all of them agree on one location and
one definition of "done".

Layout (spec 4.1)
-----------------
For a source file discovered at ``path``, ``rel`` is its path relative to the dataset's
``source_root`` (POSIX separators, from :mod:`neurodags.iterators`). The *reference base*
of the source is::

    <derivatives_path>[/<derivatives_subdir>]/<dirname(rel)>/<basename(rel) with @ -> &>

or ``<dirname(path)>/<basename with @ -> &>`` when the dataset has no
``derivatives_path`` (outputs next to the source). The output *prefix* of a unit is
``<reference base>@<derivative name>``; artifact files are ``prefix + suffix`` and markers
``prefix + ".done" | ".error" | ".skip"``. Nothing here depends on which other files were
discovered, on limits, indices or previous runs. See :func:`output_reference_base`.

Completion (spec 6)
-------------------
A unit is done when ``<prefix>.done`` is a valid manifest: it parses, its
``naming_version`` and ``derivative`` match, and every file it lists exists (see
:func:`manifest_artifacts`). The manifest is written last and atomically, from the paths
the artifact writers reported (:func:`writer_paths`), never from a glob. Files that match
the prefix but have no valid manifest make the unit ``incomplete``.
"""

from __future__ import annotations

import glob
import json
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

from neurodags.loggers import get_logger
from neurodags.naming import output_prefix_basename
from neurodags.utils import DatasetConfigError, effective_derivatives_base, resolve_dataset_field

log = get_logger(__name__)

__all__ = [
    "MANIFEST_VERSION",
    "NAMING_VERSION",
    "UnitState",
    "UnitStatus",
    "build_manifest",
    "check_output_collisions",
    "dataset_derivatives_base",
    "entry_reference_base",
    "manifest_artifacts",
    "marker_path",
    "output_prefix",
    "output_reference_base",
    "publish_marker",
    "read_manifest",
    "remove_markers",
    "unit_state",
    "write_manifest",
    "writer_paths",
]

#: Version of the ``.done`` manifest format written by this module.
MANIFEST_VERSION = 1
#: Naming/layout version the manifests certify (same number as ``naming_version`` in YAML).
NAMING_VERSION = 1

_MARKERS = ("done", "error", "skip")

UnitStatus = Literal["done", "error", "skipped", "incomplete", "missing"]


# ---------------------------------------------------------------------------
# Layout (spec 4.1, 4.3)
# ---------------------------------------------------------------------------


def output_reference_base(rel: str, source_path: str, derivatives_base: str | None) -> str:
    """Return the reference base of one source file (spec 4.1).

    This is the one function that decides where a source file's outputs go; append
    ``"@" + name`` (:func:`output_prefix`) for a unit's prefix.

    Parameters
    ----------
    rel : str
        Path of the source relative to its dataset's ``source_root``, with ``/``
        separators (:attr:`neurodags.iterators.InputEntry.rel`).
    source_path : str
        The source file's path on the active mount; used only when *derivatives_base* is
        ``None`` (outputs next to the source).
    derivatives_base : str or None
        ``derivatives_path`` joined with the optional ``derivatives_subdir``
        (:func:`dataset_derivatives_base`), or ``None``.

    Returns
    -------
    str
        ``<derivatives_base>/<dirname(rel)>/<basename(rel)>`` or
        ``<dirname(source_path)>/<basename(rel)>``, with every ``@`` of the basename
        replaced by ``&``.

    Raises
    ------
    ValueError
        If *rel* is empty, absolute or contains ``..``/``.``/empty components.
    """
    parts = rel.split("/") if isinstance(rel, str) else []
    if not parts or rel.startswith("/") or any(p in ("", ".", "..") for p in parts):
        raise ValueError(
            f"source path {rel!r} must be a non-empty path relative to source_root "
            "(no '..', '.', or empty components)."
        )
    base = parts[-1].replace("@", "&")
    if derivatives_base:
        outdir = os.path.join(derivatives_base, *parts[:-1])
    else:
        outdir = os.path.dirname(source_path)
    return os.path.join(outdir, base)


def dataset_derivatives_base(dataset_config: Any, mount_point: Any = None) -> str | None:
    """Return the resolved ``derivatives_path[/derivatives_subdir]`` of a dataset, or ``None``."""
    derivatives_path = resolve_dataset_field(dataset_config, "derivatives_path", mount_point)
    return effective_derivatives_base(
        derivatives_path, getattr(dataset_config, "derivatives_subdir", None)
    )


def entry_reference_base(entry: Any, dataset_config: Any, mount_point: Any = None) -> str:
    """Absolute reference base of an enumerated input (``entry.rel``, ``entry.path``).

    Absolute against the current working directory, so workers (which may run with a
    stale cwd) receive an already resolved path (#35).
    """
    base = output_reference_base(
        entry.rel, entry.path, dataset_derivatives_base(dataset_config, mount_point)
    )
    return os.path.abspath(base)


def output_prefix(reference_base: str | os.PathLike, name: str) -> str:
    """Return ``<reference_base>@<name>``, the prefix of every file of one unit.

    The name is validated (:func:`neurodags.naming.output_prefix_basename`); any ``@``
    left in the reference base's basename becomes ``&``.
    """
    reference_base = os.fspath(reference_base)
    dirpart, basename = os.path.split(reference_base)
    prefix_name = output_prefix_basename(basename, name)
    return os.path.join(dirpart, prefix_name) if dirpart else prefix_name


def marker_path(prefix: str, marker: str) -> str:
    """Return ``prefix + "." + marker`` for ``done``, ``error`` or ``skip``."""
    if marker not in _MARKERS:
        raise ValueError(f"unknown marker {marker!r}; expected one of {', '.join(_MARKERS)}")
    return f"{prefix}.{marker}"


def check_output_collisions(units: Iterable[tuple[str, str]]) -> None:
    """Refuse two sources that map to the same reference base (spec 4.3).

    Parameters
    ----------
    units : iterable of (label, reference_base)
        One pair per enumerated source; *label* describes the source in the message
        (e.g. ``"study: sub-01/a@b.vhdr"``).

    Raises
    ------
    DatasetConfigError
        Listing every group of sources sharing a reference base (they would read and
        overwrite each other's outputs).
    """
    by_base: dict[str, list[tuple[str, str]]] = {}
    for label, base in units:
        key = os.path.normcase(os.path.normpath(base))
        by_base.setdefault(key, []).append((label, base))
    problems = []
    for group in by_base.values():
        labels = list(dict.fromkeys(label for label, _ in group))
        if len(labels) > 1:
            problems.append(
                "sources "
                + " and ".join(repr(label) for label in labels)
                + f" map to the same outputs ({group[0][1]}@<name>...); '@' in a source "
                "file name is written as '&', and datasets sharing a tree need distinct "
                "derivatives_subdir values. Rename a source or exclude one of them."
            )
    if problems:
        raise DatasetConfigError(problems)


# ---------------------------------------------------------------------------
# Writer contract (spec 6.2)
# ---------------------------------------------------------------------------


def writer_paths(target: str, returned: Any) -> list[str]:
    """Normalize what an artifact writer returned into the list of files it wrote.

    ``Artifact.writer(path)`` returns the paths it wrote, or ``None`` meaning exactly
    *target*. A single path is accepted as a one-element list. Any other return value
    (for example the character count of ``open(path, "w").write(...)``) is treated like
    ``None``. Every reported file must exist afterwards.

    Returns
    -------
    list of str
        Reported paths, relative ones taken relative to the current working directory
        like *target* itself, without duplicates, in the order reported.

    Raises
    ------
    RuntimeError
        If a reported file does not exist (the writer did not write what it reported).
    """
    if returned is None or isinstance(returned, bool | int | float | bytes):
        paths = [target]
    elif isinstance(returned, str | os.PathLike):
        paths = [os.fspath(returned)]
    elif isinstance(returned, Sequence) and all(isinstance(p, str | os.PathLike) for p in returned):
        paths = [os.fspath(p) for p in returned] or [target]
    else:
        log.debug(
            "Artifact writer returned a non-path value; assuming it wrote the target only",
            target=target,
            returned_type=type(returned).__name__,
        )
        paths = [target]
    paths = list(dict.fromkeys(paths))
    missing = [p for p in paths if not os.path.isfile(p)]
    if missing:
        raise RuntimeError(
            f"artifact writer for {target!r} did not produce the file(s) it reported: "
            f"{missing}. A writer must return the paths it wrote, or None after writing "
            "exactly the target path."
        )
    return paths


# ---------------------------------------------------------------------------
# Manifests (spec 6.1)
# ---------------------------------------------------------------------------


def _relative_entry(path: str, manifest_dir: str) -> str:
    rel = os.path.relpath(os.path.abspath(path), os.path.abspath(manifest_dir or os.curdir))
    rel = rel.replace(os.sep, "/")
    if rel.startswith("../") or rel == ".." or os.path.isabs(rel):
        raise RuntimeError(
            f"artifact file {path!r} is outside the output folder {manifest_dir!r}; "
            "writers must write next to their target path."
        )
    return rel


def build_manifest(
    prefix: str,
    name: str,
    *,
    dataset: str | None,
    rel: str | None,
    artifacts: Mapping[str, Sequence[str]],
    companions: Sequence[str] = (),
    origin: str = "run",
) -> dict[str, Any]:
    """Build the ``.done`` manifest of one unit (spec 6.1).

    Parameters
    ----------
    prefix : str
        The unit's prefix (:func:`output_prefix`); the manifest lives at
        ``prefix + ".done"`` and lists files relative to its directory.
    name : str
        Derivative name.
    dataset, rel : str or None
        The source: dataset key and path relative to ``source_root`` (original spelling,
        including any ``@``).
    artifacts : mapping of suffix to paths
        Every file each artifact consists of, as reported by its writer (main file first).
    companions : sequence of str
        Other files belonging to the unit that no artifact suffix covers.
    origin : {"run", "migrate"}
        Who wrote the manifest.
    """
    manifest_dir = os.path.dirname(prefix)
    return {
        "manifest_version": MANIFEST_VERSION,
        "naming_version": NAMING_VERSION,
        "derivative": name,
        "source": {"dataset": dataset, "rel": rel},
        "artifacts": {
            suffix: [_relative_entry(p, manifest_dir) for p in paths]
            for suffix, paths in artifacts.items()
        },
        "companions": [_relative_entry(p, manifest_dir) for p in companions],
        "written": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "origin": origin,
    }


def write_manifest(prefix: str, manifest: Mapping[str, Any]) -> str:
    """Publish ``prefix + ".done"`` atomically (temp file in the same folder + ``os.replace``).

    A crash before the replace leaves no ``.done`` (only a hidden temp file), so the unit
    is recomputed. Returns the manifest path.
    """
    path = marker_path(prefix, "done")
    directory = os.path.dirname(path) or os.curdir
    fd, tmp = tempfile.mkstemp(
        dir=directory, prefix="." + os.path.basename(path) + ".", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return path


def read_manifest(prefix: str) -> dict[str, Any] | None:
    """Return the parsed ``prefix + ".done"``, or ``None`` when absent or not a JSON object."""
    path = marker_path(prefix, "done")
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError:
        return None
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _safe_entry(entry: Any) -> bool:
    if not isinstance(entry, str) or not entry or entry.startswith("/") or "\\" in entry:
        return False
    return all(part not in ("", ".", "..") for part in entry.split("/"))


def _manifest_problem(
    manifest: Mapping[str, Any] | None, prefix: str, name: str
) -> tuple[str | None, dict[str, list[str]]]:
    """Return ``(problem, {suffix: [absolute paths]})``; ``problem`` is None when valid."""
    if manifest is None:
        return "no readable .done manifest", {}
    if manifest.get("manifest_version") != MANIFEST_VERSION:
        return f"unsupported manifest_version {manifest.get('manifest_version')!r}", {}
    if manifest.get("naming_version") != NAMING_VERSION:
        return f"naming_version {manifest.get('naming_version')!r} does not match", {}
    if manifest.get("derivative") != name:
        return f"manifest is for derivative {manifest.get('derivative')!r}, not {name!r}", {}
    artifacts = manifest.get("artifacts")
    companions = manifest.get("companions", [])
    if not isinstance(artifacts, dict) or not isinstance(companions, list):
        return "malformed artifacts/companions", {}
    directory = os.path.dirname(prefix)
    resolved: dict[str, list[str]] = {}
    for suffix, entries in artifacts.items():
        if not isinstance(suffix, str) or not isinstance(entries, list) or not entries:
            return f"malformed entry for artifact {suffix!r}", {}
        if not all(_safe_entry(e) for e in entries):
            return f"unsafe file name listed for artifact {suffix!r}", {}
        resolved[suffix] = [os.path.join(directory, *e.split("/")) for e in entries]
    if not all(_safe_entry(e) for e in companions):
        return "unsafe companion file name", {}
    listed = [p for paths in resolved.values() for p in paths]
    listed += [os.path.join(directory, *e.split("/")) for e in companions]
    missing = [p for p in listed if not os.path.isfile(p)]
    if missing:
        return f"listed file(s) missing: {', '.join(os.path.basename(p) for p in missing)}", {}
    return None, resolved


def manifest_artifacts(prefix: str, name: str) -> dict[str, list[str]] | None:
    """Cache hit check (spec 6.2): the unit's files per suffix, or ``None``.

    The unit is done when ``prefix + ".done"`` parses, its ``naming_version`` and
    ``derivative`` match and every listed file exists. Files matching the prefix but not
    listed are ignored.

    Returns
    -------
    dict of str to list of str, or None
        ``{suffix: [paths]}`` (the main file first, then e.g. MNE split parts), with paths
        joined onto the manifest's directory; ``None`` when there is no valid manifest.
    """
    problem, resolved = _manifest_problem(read_manifest(prefix), prefix, name)
    return None if problem else resolved


@dataclass(frozen=True)
class UnitState:
    """Completion state of one (source, derivative) unit; see :func:`unit_state`."""

    state: UnitStatus
    prefix: str
    #: Files per artifact suffix when ``state == "done"``, else empty.
    artifacts: dict[str, list[str]] = field(default_factory=dict)
    #: ``origin`` of a valid manifest (``"run"`` or ``"migrate"``).
    origin: str | None = None
    #: Why an existing ``.done`` is not valid (``None`` when valid or absent).
    problem: str | None = None
    #: For a done unit: files matching ``prefix.*`` that its manifest does not list (they
    #: are ignored). Empty for other states.
    unlisted: tuple[str, ...] = ()
    error_path: str | None = None
    skip_path: str | None = None


def _prefix_files(prefix: str) -> list[str]:
    # The prefix is a literal path built from source and derivative names: escape it so
    # glob metacharacters ([, ], *, ?) in any component match verbatim. Status only:
    # never used to decide a cache hit.
    marker_names = {marker_path(prefix, m) for m in _MARKERS}
    return [
        p
        for p in sorted(glob.glob(glob.escape(prefix) + ".*"))
        if p not in marker_names and os.path.isfile(p)
    ]


def unit_state(prefix: str, name: str, *, scan: bool = True) -> UnitState:
    """Classify one unit (spec 6.2): ``skipped``, ``done``, ``error``, ``incomplete``, ``missing``.

    Precedence follows what a run does: a ``.skip`` marker wins (the unit is not
    recomputed), then a valid ``.done``, then ``.error``. Without any of those, files
    matching the prefix (or an invalid ``.done``) make it ``incomplete``; otherwise it is
    ``missing``. With *scan* False the directory is not globbed (``unlisted`` stays empty
    and leftover files without a manifest count as ``missing``).
    """
    skip_path = marker_path(prefix, "skip")
    error_path = marker_path(prefix, "error")
    has_skip = os.path.exists(skip_path)
    has_error = os.path.exists(error_path)
    manifest = read_manifest(prefix)
    done_exists = os.path.exists(marker_path(prefix, "done"))
    problem, resolved = _manifest_problem(manifest, prefix, name)
    files = _prefix_files(prefix) if scan else []
    listed = {os.path.normpath(p) for paths in resolved.values() for p in paths}
    unlisted = (
        tuple(p for p in files if os.path.normpath(p) not in listed) if problem is None else ()
    )
    common = {
        "prefix": prefix,
        "error_path": error_path if has_error else None,
        "skip_path": skip_path if has_skip else None,
        "unlisted": unlisted,
        "problem": problem if done_exists else None,
    }
    if has_skip:
        return UnitState("skipped", **common)
    if problem is None:
        return UnitState("done", artifacts=resolved, origin=manifest.get("origin"), **common)
    if has_error:
        return UnitState("error", **common)
    if done_exists or files:
        return UnitState("incomplete", **common)
    return UnitState("missing", **common)


# ---------------------------------------------------------------------------
# Markers (spec 6.2 write order)
# ---------------------------------------------------------------------------


def remove_markers(prefix: str, *markers: str) -> None:
    """Delete the given markers (``done``, ``error``, ``skip``) of a unit if present."""
    for marker in markers:
        path = marker_path(prefix, marker)
        try:
            os.unlink(path)
            log.debug("Removed marker", path=path)
        except FileNotFoundError:
            pass


def publish_marker(prefix: str, marker: str, text: str) -> str:
    """Write a ``.error`` or ``.skip`` marker and remove the unit's other markers.

    A unit carries at most one of ``.done``/``.error``/``.skip``. Returns the marker path.
    """
    if marker not in ("error", "skip"):
        raise ValueError("publish_marker writes 'error' or 'skip'; use write_manifest for done")
    path = marker_path(prefix, marker)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    remove_markers(prefix, *(m for m in _MARKERS if m != marker))
    return path
