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

    <derivatives_path>[/<derivatives_subdir>]/<dirname(rel)>/<basename(rel)>

or ``<dirname(path)>/<basename>`` when the dataset has no
``derivatives_path`` (outputs next to the source). The output *prefix* of a unit is
``<reference base>@<derivative name>``; artifact files are ``prefix + suffix`` and markers
``prefix + ".done" | ".error" | ".skip"``. The source basename is kept verbatim (it may
contain ``@``, e.g. a chained source ``sub-1.fif@Clean.fif``); the last ``@`` of an output
name separates source and derivative. Nothing here depends on which other files were
discovered, on limits, indices or previous runs. See :func:`output_reference_base`.

Completion (spec 6)
-------------------
A unit is done when ``<prefix>.done`` is a valid manifest: it parses, its
``naming_version`` and ``derivative`` match, and every file it lists exists (see
:func:`manifest_artifacts`). The manifest is written last and atomically, from the paths
the artifact writers reported (:func:`writer_paths`), never from a glob. Files that match
the prefix but have no valid manifest make the unit ``incomplete``.

Atomic publication
------------------
Artifact writers never write at the final names. Each (re)computation writes into a hidden
staging folder next to the outputs (:func:`make_staging_dir`,
``.<prefix basename>.tmp~<token>/``, using the final basenames), then moves every file into
place with ``os.replace`` (:func:`publish_staged`): continuation parts first, each main file
last. A reader therefore sees either a complete old file or a complete new one, never a
partially written one. A staging folder left by a killed process is ignored everywhere
(discovery, status, dataframes, the legacy guard) and reported by ``status``
(:attr:`UnitState.temps`).
"""

from __future__ import annotations

import glob
import json
import os
import re
import shutil
import tempfile
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

from neurodags.loggers import get_logger
from neurodags.naming import output_prefix_basename, parse_output_basename
from neurodags.utils import DatasetConfigError, effective_derivatives_base, resolve_dataset_field

log = get_logger(__name__)

__all__ = [
    "MANIFEST_VERSION",
    "NAMING_VERSION",
    "LegacyOutputsError",
    "UnitState",
    "UnitStatus",
    "build_manifest",
    "check_artifact_ownership",
    "check_legacy_chained_outputs",
    "check_output_collisions",
    "dataset_derivatives_base",
    "entry_reference_base",
    "is_staging_name",
    "make_staging_dir",
    "manifest_artifacts",
    "marker_path",
    "output_prefix",
    "output_reference_base",
    "publish_marker",
    "publish_staged",
    "read_manifest",
    "remove_markers",
    "remove_staging_dir",
    "staging_dirs",
    "unit_state",
    "write_manifest",
    "writer_paths",
]

#: Version of the ``.done`` manifest format written by this module.
MANIFEST_VERSION = 1
#: Naming/layout version the manifests certify (same number as ``naming_version`` in YAML).
NAMING_VERSION = 1

_MARKERS = ("done", "error", "skip")
_SPLIT_NAME_RE = re.compile(r"(.+)-([0-9]+)")
#: Marks a staging folder: ``.<prefix basename>.tmp~<token>``. ``~`` is not allowed in a
#: suffix, so the name never parses as an output (:func:`neurodags.naming.parse_output_basename`).
STAGING_TAG = ".tmp~"

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
        ``<dirname(source_path)>/<basename(rel)>``, with the basename verbatim.

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
    base = parts[-1]
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

    The name is validated (:func:`neurodags.naming.output_prefix_basename`); the reference
    base's basename is kept verbatim.
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
                + f" map to the same outputs ({group[0][1]}@<name>...). Datasets sharing "
                "an output tree need distinct derivatives_subdir values, and file names "
                "that differ only in case collide on case-insensitive filesystems. Rename "
                "a source or exclude one of them."
            )
    if problems:
        raise DatasetConfigError(problems)


class LegacyOutputsError(DatasetConfigError):
    """Outputs exist only in the pre-0.4.0 layout; ``neurodags migrate`` converts them."""

    def __init__(self, problems: Sequence[str]):
        self.problems = list(problems)
        lines = "\n".join(f"  - {p}" for p in self.problems)
        ValueError.__init__(self, f"outputs need 'neurodags migrate':\n{lines}")


def check_legacy_chained_outputs(
    datasets: Mapping[str, Any],
    mount_point: Any,
    names: Iterable[str],
    entries: Iterable[Any] = (),
) -> None:
    """Refuse unmigrated 0.3.x outputs of sources whose basename contained ``@``.

    Before 0.4.0 every ``@`` of a source basename was written as ``&`` (a chained source
    ``x@A.fif`` gave ``x&A.fif@B.nc``); 0.4.0 keeps it verbatim (``x@A.fif@B.nc``). Such
    units would silently be recomputed, so this stops with a pointer to ``neurodags
    migrate``.

    One scan per active output tree: every ``derivatives_path`` (whole, so pilot-era flat
    files and folders from before a ``derivatives_subdir`` count; ``code/`` and
    ``.neurodags/`` skipped) and, for datasets writing next to their sources, the folders
    of the enumerated *entries*. A file is legacy when its name parses (last ``@``) as an
    output of one of *names* (or its 0.3.x CamelCase token), its source part contains
    ``&``, and nothing shows it belongs to a 0.4.0 unit:

    - no ``.done`` in the scanned trees has the same source part, in its own spelling or
      with ``@`` written ``&`` (0.3.x never wrote ``.done``; this also covers duplicates
      left by ``migrate``), and
    - no source file of that exact spelling exists where the dataset's ``source_root``
      would hold it (a real source named with ``&``, whatever is selected now).

    Downstream outputs (``x@Clean.fif@Fit.nc@Power.nc``) never count: their source part
    holds no ``&``.

    Parameters
    ----------
    datasets : mapping
        Dataset configurations (``skip: true`` ones are ignored).
    mount_point : str or None
        Active mount.
    names : iterable of str
        Derivative names of the pipeline.
    entries : iterable
        Enumerated inputs (``dataset``, ``path``), for datasets without
        ``derivatives_path``.

    Raises
    ------
    LegacyOutputsError
        Listing the files (a :class:`neurodags.utils.DatasetConfigError`).
    """
    from neurodags._legacy import snake_to_camel

    token_names: dict[str, str] = {}
    for name in names:
        token_names.setdefault(name, name)
        token_names.setdefault(snake_to_camel(name), name)
    if not token_names:
        return

    def norm(path: str) -> str:
        return os.path.normcase(os.path.normpath(os.path.abspath(path)))

    # (output root, its source_root) pairs used to look for real '&' sources.
    contexts: list[tuple[str, str]] = []
    trees: set[str] = set()  # walked recursively
    folders: set[str] = set()  # listed only (outputs next to the sources)
    for name, cfg in datasets.items():
        if getattr(cfg, "skip", False):
            continue
        try:
            source_root = resolve_dataset_field(cfg, "source_root", mount_point)
            derivatives = resolve_dataset_field(cfg, "derivatives_path", mount_point)
            base = dataset_derivatives_base(cfg, mount_point)
        except (ValueError, KeyError):
            continue  # reported by the configuration checks
        if derivatives:
            trees.add(os.path.abspath(derivatives))
            if source_root:
                contexts.append((os.path.abspath(derivatives), os.path.abspath(source_root)))
                if base:
                    contexts.append((os.path.abspath(base), os.path.abspath(source_root)))
        else:
            folders.update(
                os.path.dirname(os.path.abspath(e.path))
                for e in entries
                if getattr(e, "dataset", None) == name
            )

    def files() -> Iterable[tuple[str, str]]:
        for root in sorted(trees):
            for dirpath, dirnames, filenames in os.walk(root):
                dirnames[:] = sorted(
                    d
                    for d in dirnames
                    if (dirpath != root or d not in ("code", ".neurodags"))
                    and not is_staging_name(d)
                )
                for fname in sorted(filenames):
                    yield dirpath, fname
        for folder in sorted(folders - trees):
            try:
                listing = sorted(os.listdir(folder))
            except OSError:
                continue
            for fname in listing:
                yield folder, fname

    # (source part in 0.3.x spelling, name) -> folders holding such a .done
    done_dirs: dict[tuple[str, str], list[str]] = {}
    suspects: list[tuple[str, str, str, str]] = []  # (dir, source part, path, name)
    seen: set[str] = set()
    for dirpath, fname in files():
        if "@" not in fname:
            continue
        path = os.path.join(dirpath, fname)
        key = norm(path)
        if key in seen:
            continue
        seen.add(key)
        parsed = parse_output_basename(fname)
        if parsed is None:
            continue
        name = token_names.get(parsed.name)
        if name is None:
            split = _SPLIT_NAME_RE.fullmatch(parsed.name)
            name = token_names.get(split.group(1)) if split else None
        if name is None:
            continue
        if parsed.marker == "done" and parsed.name == name:
            key = (parsed.source_part.replace("@", "&"), name)
            done_dirs.setdefault(key, []).append(dirpath)
            continue
        if "&" in parsed.source_part and os.path.isfile(path):
            suspects.append((dirpath, parsed.source_part, path, name))

    def rel_in_tree(path: str) -> tuple[str, str] | None:
        for root in trees:
            rel = os.path.relpath(path, root)
            if rel == os.curdir or not rel.startswith(os.pardir):
                return norm(root), rel.replace(os.sep, "/")
        return None

    def claimed(dirpath: str, key: tuple[str, str]) -> bool:
        """A .done of this unit in the same folder, or (same tree) in a folder whose
        path ends with this one: the unit was migrated and this file is a leftover."""
        here = rel_in_tree(dirpath)
        for done_dir in done_dirs.get(key, []):
            if norm(done_dir) == norm(dirpath):
                return True
            there = rel_in_tree(done_dir)
            if here and there and here[0] == there[0]:
                old, new = here[1], there[1]
                if old == "." or new == old or new.endswith("/" + old):
                    return True
        return False

    listed = {norm(f) for f in folders}
    found: list[str] = []
    for dirpath, source_part, path, name in suspects:
        if claimed(dirpath, (source_part, name)):
            continue
        real = [os.path.join(dirpath, source_part)] if norm(dirpath) in listed else []
        for out_root, source_root in contexts:
            rel = os.path.relpath(dirpath, out_root)
            if rel == os.curdir or not rel.startswith(os.pardir):
                real.append(os.path.normpath(os.path.join(source_root, rel, source_part)))
        if any(os.path.isfile(r) for r in real):
            continue  # a real source named with '&' (0.4.0 unit without .done yet)
        found.append(path)
    if found:
        shown = "\n      ".join(found[:5]) + (
            f"\n      ... and {len(found) - 5} more" if len(found) > 5 else ""
        )
        raise LegacyOutputsError(
            [
                f"{len(found)} file(s) look like neurodags 0.3.x outputs of sources whose "
                "name contained '@' (0.3.x wrote it as '&'; 0.4.0 keeps source names "
                "verbatim), and no .done claims them, so their units would be recomputed:"
                f"\n      {shown}\n    Run 'neurodags migrate' on this pipeline to rename "
                "them. If they are not neurodags 0.3.x outputs, or are obsolete, move them "
                "out of the output tree."
            ]
        )


# ---------------------------------------------------------------------------
# Writer contract (spec 6.2)
# ---------------------------------------------------------------------------


def writer_paths(target: str, returned: Any) -> list[str]:
    """Normalize what an artifact writer returned into the list of files it wrote.

    ``Artifact.writer(path)`` returns the paths it wrote, or ``None`` meaning exactly
    *target*. A single path is accepted as a one-element list. Any other return value
    (for example the character count of ``open(path, "w").write(...)``) is treated like
    ``None``. Every reported file must exist afterwards. During a run, *target* lies in the
    unit's staging folder (:func:`make_staging_dir`); :func:`publish_staged` then moves the
    reported files into place.

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


def check_artifact_ownership(written: Mapping[str, Sequence[str]]) -> None:
    """Refuse a file claimed by two artifacts of one unit.

    Two artifacts reporting the same file (for example ``.EO.fif``, whose MNE split part is
    ``<prefix>.EO-1.fif``, and an artifact ``.EO-1.fif``) mean one overwrote the other, so
    the unit must not be certified done.

    Parameters
    ----------
    written : mapping of suffix to paths
        The files each artifact reported (or is about to write).

    Raises
    ------
    RuntimeError
        Naming every file claimed by more than one artifact.
    """
    owners: dict[str, list[str]] = {}
    for suffix, paths in written.items():
        for path in paths:
            key = os.path.normcase(os.path.abspath(os.fspath(path)))
            group = owners.setdefault(key, [])
            if suffix not in group:
                group.append(suffix)
    conflicts = [
        f"{os.path.basename(key)} (artifacts {', '.join(repr(s) for s in suffixes)})"
        for key, suffixes in owners.items()
        if len(suffixes) > 1
    ]
    if conflicts:
        raise RuntimeError(
            "two artifacts of one unit write the same file, so one would overwrite the "
            f"other: {'; '.join(conflicts)}. Rename an artifact suffix (an MNE recording "
            "'<prefix>.X.fif' may also write '<prefix>.X-1.fif', ...)."
        )


# ---------------------------------------------------------------------------
# Atomic publication (spec 6.2)
# ---------------------------------------------------------------------------


def is_staging_name(basename: str) -> bool:
    """Whether *basename* is a staging folder (``.<prefix basename>.tmp~<token>``)."""
    return basename.startswith(".") and STAGING_TAG in basename


def _staging_glob(prefix: str) -> str:
    directory, base = os.path.split(prefix)
    return os.path.join(directory, glob.escape("." + base + STAGING_TAG) + "*")


def staging_dirs(prefix: str) -> list[str]:
    """Staging folders of one unit on disk (live ones of a running job, or left by a killed one)."""
    return sorted(p for p in glob.glob(_staging_glob(prefix)) if os.path.isdir(p))


def make_staging_dir(prefix: str) -> str:
    """Create a fresh hidden staging folder for one (re)computation of a unit.

    The folder is ``<dir>/.<prefix basename>.tmp~<token>``: next to the outputs, so moving its
    files into place is a same-filesystem ``os.replace``, and unique per computation, so
    concurrent computations of the same unit never write into each other's files.
    """
    directory, base = os.path.split(prefix)
    path = os.path.join(directory, "." + base + STAGING_TAG + uuid.uuid4().hex[:16])
    os.mkdir(path)
    return path


def remove_staging_dir(path: str) -> None:
    """Delete a staging folder (after publishing or after a failure)."""
    shutil.rmtree(path, ignore_errors=True)


def publish_staged(
    staging: str, prefix: str, staged: Mapping[str, Sequence[str]]
) -> dict[str, list[str]]:
    """Move the files of every artifact from *staging* into place with ``os.replace``.

    Each staged file keeps its path relative to *staging* (normally just the final basename,
    e.g. ``x@Clean.fif`` and its MNE split part ``x@Clean-1.fif``). Continuation files
    (everything after the first file an artifact reported) are moved first and each main file
    last, so a reader that finds a main file finds the parts it links to (MNE links split
    parts by basename, so moving them keeps them linked).

    Parameters
    ----------
    staging : str
        The staging folder (:func:`make_staging_dir`).
    prefix : str
        The unit's prefix; files land in its folder.
    staged : mapping of suffix to paths
        The files each artifact writer reported, all inside *staging* (main file first).

    Returns
    -------
    dict of str to list of str
        The final paths per suffix, in the reported order.

    Raises
    ------
    RuntimeError
        If a writer reported a file outside *staging*: writers must write into the folder of
        the path they are given (a file written elsewhere cannot be published atomically).
    """
    directory = os.path.dirname(prefix)
    root = os.path.abspath(staging)
    final: dict[str, list[str]] = {}
    moves: list[tuple[str, str]] = []
    mains: list[tuple[str, str]] = []
    for suffix, paths in staged.items():
        final[suffix] = []
        for index, path in enumerate(paths):
            rel = os.path.relpath(os.path.abspath(path), root)
            if rel == os.curdir or rel.startswith(os.pardir) or os.path.isabs(rel):
                raise RuntimeError(
                    f"artifact writer for {suffix!r} reported {path!r}, outside the path it "
                    f"was given ({root}); writers must write next to (or below) their target "
                    "path so the files can be published atomically."
                )
            target = os.path.join(directory, rel)
            final[suffix].append(target)
            (mains if index == 0 else moves).append((path, target))
    for source, target in moves + mains:
        parent = os.path.dirname(target)
        if parent:
            os.makedirs(parent, exist_ok=True)
        os.replace(source, target)
    return final


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
    #: Staging folders of the unit (:func:`staging_dirs`): a job is computing the unit right
    #: now, or one was killed while computing it. They never affect the state.
    temps: tuple[str, ...] = ()


def _prefix_files(prefix: str) -> list[str]:
    # The prefix is a literal path built from source and derivative names: escape it so
    # glob metacharacters ([, ], *, ?) in any component match verbatim. Status only:
    # never used to decide a cache hit.
    # Outputs of a downstream unit whose source is this unit's artifact (for prefix
    # ``x@Clean``: ``x@Clean.fif@Power.nc``) also match the glob; they parse at their last
    # ``@`` to another unit and are not this unit's files.
    marker_names = {marker_path(prefix, m) for m in _MARKERS}
    own = len(os.path.basename(prefix))
    return [
        p
        for p in sorted(glob.glob(glob.escape(prefix) + ".*"))
        if p not in marker_names
        and os.path.isfile(p)
        and not _is_descendant_output(os.path.basename(p), own)
    ]


def _is_descendant_output(basename: str, prefix_len: int) -> bool:
    # This unit's own suffixes never contain ``@`` (SUFFIX_RE), so an ``@`` after the
    # prefix plus a valid last-``@`` parse means the file belongs to a downstream unit.
    return "@" in basename[prefix_len:] and parse_output_basename(basename) is not None


def unit_state(prefix: str, name: str, *, scan: bool = True) -> UnitState:
    """Classify one unit (spec 6.2): ``skipped``, ``done``, ``error``, ``incomplete``, ``missing``.

    Precedence follows what a run does: a ``.skip`` marker wins (the unit is not
    recomputed), then a valid ``.done``, then ``.error``. Without any of those, files
    matching the prefix (or an invalid ``.done``) make it ``incomplete``; otherwise it is
    ``missing``. With *scan* False the directory is not globbed (``unlisted`` and ``temps``
    stay empty and leftover files without a manifest count as ``missing``). Staging folders
    (``temps``) are reported but never change the state.
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
        "temps": tuple(staging_dirs(prefix)) if scan else (),
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
