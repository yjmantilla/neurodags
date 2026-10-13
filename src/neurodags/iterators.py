import glob
import ntpath
import os
import re
from collections.abc import Iterable
from dataclasses import dataclass, field

from neurodags.datasets import get_datasets_and_mount_point_from_pipeline_configuration
from neurodags.definitions import DatasetConfig
from neurodags.loggers import get_logger
from neurodags.utils import (
    DatasetConfigError,
    effective_derivatives_base,
    relative_to_source_root,
    resolve_dataset_field,
)

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Split-FIF detection
# ---------------------------------------------------------------------------
#
# MNE cannot write more than ~2 GB to a single ``.fif``, so large recordings are
# written as a *set* of split files. Only the first (*entry*) file is an
# independent, self-describing recording; the continuations are stitched in
# transparently by ``mne.io.read_raw_fif(entry)`` via the ``next_fname`` pointer
# stored in each FIF header. A naive ``glob`` therefore returns every
# continuation as if it were a separate source file, which makes the pipeline
# run derivatives on partial data and emit duplicate/garbage rows.
#
# Two naming conventions exist. The exact rules below were taken from mne's
# ``_make_split_fnames`` / ``_construct_bids_filename`` (mne/_fiff/utils.py) —
# not guessed:
#
#   * BIDS (mne-bids, ``split_naming="bids"``):
#       entry        ``sub-X_..._split-01_meg.fif``
#       continuation ``sub-X_..._split-02_meg.fif``, ``..._split-03_...`` , ...
#     The index is 1-based (``{part_idx + 1:02}``), zero-padded to at least two
#     digits, and lives in the reserved ``_split-`` BIDS entity. A recording
#     that fits in one file keeps its bare name (no ``_split-`` entity at all),
#     so a lone ``_split-01`` effectively never occurs.
#
#   * neuromag (plain mne, the default ``split_naming="neuromag"``):
#       entry        ``name.fif``
#       continuation ``name-1.fif``, ``name-2.fif``, ...
#     The index is 1-based (``f"{base}-{i:d}{ext}"`` for ``i >= 1``), NOT
#     zero-padded, and the entry has no ``-N`` token at all.
#
# Detection is purely filename-based (cheap — no file reads during scanning).
# The BIDS ``_split-`` entity is reserved, so its rule has no realistic false
# positives. The neuromag ``-N`` tail is more ambiguous, so a continuation is
# only dropped when its computed entry file is *also* present in the scanned
# set; this prevents dropping legitimately named standalone files such as
# ``sub-01_..._run-2_meg.fif`` (whose stem ends in ``_meg``, not ``-<digit>``)
# or a lone ``foo-2.fif`` with no ``foo.fif`` sibling.

_BIDS_SPLIT_RE = re.compile(r"_split-(\d+)_")
_NEUROMAG_TAIL_RE = re.compile(r"^(?P<base>.+)-(?P<idx>\d+)$")


def _is_fif(path: str) -> bool:
    """Return True for FIF files (``.fif`` or ``.fif.gz``, case-insensitive)."""
    low = path.lower()
    return low.endswith(".fif") or low.endswith(".fif.gz")


def _neuromag_entry_for(path: str) -> str | None:
    """Return the neuromag *entry* path if ``path`` is a ``-N`` continuation.

    ``name-1.fif`` -> ``name.fif``; ``name.fif`` (the entry) -> ``None``.
    The index must be ``>= 1`` (mne writes the entry with no ``-N`` suffix;
    ``-0`` is never produced). The continuation is recognised by stripping the
    trailing ``-N`` from the stem and checking that the resulting *entry* is a
    FIF file — this also handles gzipped splits, which mne names
    ``name.fif-1.gz`` with entry ``name.fif.gz``.
    """
    root, ext = os.path.splitext(path)
    m = _NEUROMAG_TAIL_RE.match(root)
    if m is None or int(m.group("idx")) < 1:
        return None
    entry = m.group("base") + ext
    if not _is_fif(entry):
        return None
    return entry


def find_split_continuations(files: list[str]) -> set[str]:
    """Return the subset of ``files`` that are split-FIF *continuations*.

    A continuation is any part of a split recording other than the entry file.
    See the module-level notes for the exact BIDS / neuromag rules. Files that
    are not FIF, or FIF files that are not part of a split set, are never
    returned.

    Parameters
    ----------
    files : list of str
        Candidate file paths (e.g. the raw output of :func:`glob.glob`).

    Returns
    -------
    set of str
        The paths in ``files`` that should be dropped as continuations.
    """
    # The same pathname can be returned several times (e.g. by a pattern with several
    # ``**`` components); repeated occurrences are one file, never extra split members.
    files = list(dict.fromkeys(files))
    fileset = set(files)
    drop: set[str] = set()

    # --- BIDS: keep the lowest _split-NN_ per group, drop the rest ----------
    # Group files that differ only in the _split-NN_ index (same directory,
    # same prefix/suffix) and keep the lowest index as the entry.
    bids_groups: dict[tuple[str, str], list[tuple[int, str]]] = {}
    for f in files:
        if not _is_fif(f):
            continue
        base = os.path.basename(f)
        m = _BIDS_SPLIT_RE.search(base)
        if m is None:
            continue
        idx = int(m.group(1))
        key = (os.path.dirname(f), _BIDS_SPLIT_RE.sub("_split-_", base))
        bids_groups.setdefault(key, []).append((idx, f))
    for members in bids_groups.values():
        if len(members) < 2:
            # A lone _split-NN_ has nothing to stitch onto — keep it as-is.
            continue
        members.sort()  # by index, then path
        for _idx, f in members[1:]:
            drop.add(f)

    # --- neuromag: drop name-N.fif only when its entry name.fif is present ---
    for f in files:
        entry = _neuromag_entry_for(f)
        if entry is not None and entry in fileset:
            drop.add(f)

    return drop


def get_files_from_pattern(
    pattern,
    recursive: bool = True,
    exclude_filter=None,
    drop_split_continuations: bool = True,
) -> list[str]:
    """
    Get a list of file paths matching the given glob pattern.

    Parameters
    ----------
    pattern : str
        The glob pattern to match files.
    recursive : bool, optional
        Whether to search recursively in subdirectories. Default is True.
    exclude_filter : str, optional
        A glob pattern to exclude certain files from the results.
    drop_split_continuations : bool, optional
        When True (default), drop split-FIF *continuation* files, keeping only
        the entry file of each split recording (both BIDS ``_split-`` and plain
        mne ``-N`` conventions). This prevents the pipeline from treating
        partial recordings as independent source files. Set to False to return
        every matched file unchanged. See :func:`find_split_continuations`.

    Returns
    -------
    list of str
        The matching *files* (directories are never returned; symbolic links to files
        are, under their own path), lexically normalized (:func:`os.path.normpath`, no
        symlink resolution) and without duplicates, in glob order.
    """

    log.debug("get_files_from_pattern: called", pattern=pattern, recursive=recursive)
    # Normalize lexically and deduplicate before anything compares paths: a pattern
    # such as ``raw/./**`` or ``raw/**/**/*`` returns other spellings of, or repeated,
    # pathnames for the same file. ``realpath`` is never used: the lexical path is the
    # input's identity (spec 4.1).
    matches = list(
        dict.fromkeys(os.path.normpath(f) for f in glob.glob(pattern, recursive=recursive))
    )
    # Inputs are files: a broad pattern (``raw/**/*``) also matches directories.
    files = [f for f in matches if os.path.isfile(f)]
    log.debug(
        "get_files_from_pattern: found files",
        count=len(files),
        skipped_non_files=len(matches) - len(files),
    )

    # Apply exclude filter if provided (both sides normalized the same way)
    if exclude_filter:
        log.debug("get_files_from_pattern: applying exclude filter", exclude_filter=exclude_filter)
        excluded_files = {
            os.path.normpath(f) for f in glob.glob(exclude_filter, recursive=recursive)
        }
        files = [f for f in files if f not in excluded_files]
        log.debug("get_files_from_pattern: files after exclusion", count=len(files))

    # Drop split-FIF continuations (keep only entry files)
    if drop_split_continuations:
        continuations = find_split_continuations(files)
        if continuations:
            log.info(
                "get_files_from_pattern: dropped split-FIF continuations",
                dropped=len(continuations),
                kept=len(files) - len(continuations),
            )
            files = [f for f in files if f not in continuations]

    return files


# ---------------------------------------------------------------------------
# Enumeration (spec §8): sorted discovery, one enumeration per invocation,
# frozen input lists
# ---------------------------------------------------------------------------

#: First line of a frozen input list written by ``neurodags inputs``.
INPUTS_HEADER = "# neurodags-inputs naming_version=1"


class InputsError(ValueError):
    """Base class for input-selection errors (frozen input lists, ``only_index``)."""


class InputsFileError(InputsError):
    """A frozen input list (``--inputs``) is malformed or does not match the configuration."""


class IndexSelectionError(InputsError):
    """``only_index`` names an index that is unknown, negative or a missing source."""


@dataclass(frozen=True)
class InputEntry:
    """One indexed input of a pipeline invocation."""

    #: 0-based position in the enumeration (what ``--only-index`` and SLURM array task
    #: ids select).
    index: int
    #: Dataset key in the configuration.
    dataset: str
    #: Path of the source file relative to the dataset's ``source_root``, with POSIX
    #: separators (``/``). Independent of the mount point.
    rel: str
    #: Absolute path of the source file on the active mount (``source_root / rel``).
    path: str
    #: ``True`` when the entry comes from a frozen input list and the file no longer
    #: exists (reported as ``missing-source``). Always ``False`` for discovery.
    missing: bool = False


@dataclass
class InputEnumeration:
    """The indexed inputs of one invocation, shared by every derivative it runs."""

    #: Inputs in index order (``entries[i].index == i``).
    entries: list[InputEntry]
    #: Per dataset with inputs: the resolved, normalized ``source_root`` on the active
    #: mount. Output locations are computed from it and each entry's ``rel`` only
    #: (:func:`neurodags.outputs.output_reference_base`), never from the discovered set.
    source_roots: dict[str, str] = field(default_factory=dict)
    #: The frozen input list the entries were read from, ``None`` for discovery.
    inputs_file: str | None = None
    #: With a frozen list: per dataset, the ``rel`` of files found on disk that the list
    #: does not contain (ignored). Empty for discovery.
    not_in_list: dict[str, list[str]] = field(default_factory=dict)
    #: Set once the pre-0.4.0 output guard ran for this enumeration (internal).
    legacy_checked: bool = field(default=False, repr=False, compare=False)

    @property
    def files_per_dataset(self) -> dict[str, list[str]]:
        """Absolute paths per dataset, in index order (datasets without inputs omitted)."""
        out: dict[str, list[str]] = {}
        for entry in self.entries:
            out.setdefault(entry.dataset, []).append(entry.path)
        return out

    @property
    def all_files(self) -> list[tuple[int, str, str]]:
        """``(index, dataset, path)`` tuples, the 0.3.x enumeration shape."""
        return [(e.index, e.dataset, e.path) for e in self.entries]

    @property
    def missing_sources(self) -> list[InputEntry]:
        """Listed entries whose file no longer exists (frozen lists only)."""
        return [e for e in self.entries if e.missing]

    @property
    def n_not_in_list(self) -> int:
        """Number of files on disk that a frozen list does not contain."""
        return sum(len(v) for v in self.not_in_list.values())

    def select(self, only_index: int | Iterable[int] | None = None) -> list[InputEntry]:
        """Return the entries selected by *only_index* (all entries when ``None``).

        Raises
        ------
        IndexSelectionError
            If an index is negative or not in the enumeration, or names a
            ``missing-source`` entry of a frozen input list.
        """
        if only_index is None:
            return list(self.entries)
        requested = [only_index] if isinstance(only_index, int) else list(only_index)
        n = len(self.entries)
        bad = [i for i in requested if isinstance(i, bool) or not isinstance(i, int)]
        if bad:
            raise IndexSelectionError(f"only_index values must be integers, got {bad!r}.")
        negative = sorted({i for i in requested if i < 0})
        unknown = sorted({i for i in requested if i >= n})
        if negative or unknown:
            parts = []
            if negative:
                parts.append(f"negative: {', '.join(map(str, negative))}")
            if unknown:
                parts.append(f"unknown: {', '.join(map(str, unknown))}")
            valid = f"0-{n - 1}" if n else "none (no inputs)"
            raise IndexSelectionError(
                f"invalid --only-index value(s) ({'; '.join(parts)}); valid indices: {valid}."
            )
        wanted = set(requested)
        selected = [e for e in self.entries if e.index in wanted]
        missing = [e for e in selected if e.missing]
        if missing:
            shown = ", ".join(f"{e.index} ({e.dataset}: {e.rel})" for e in missing)
            raise IndexSelectionError(
                f"--only-index selects missing-source input(s): {shown}. The file(s) are "
                f"listed in {self.inputs_file} but no longer exist on disk."
            )
        return selected


def _posix_rel(rel: str) -> str:
    return rel.replace(os.sep, "/") if os.sep != "/" else rel


def _is_below(path: str, directory: str) -> bool:
    """Whether normalized *path* lies strictly below normalized *directory* (lexical)."""
    return path != directory and path.startswith(directory.rstrip(os.sep) + os.sep)


def output_tree_exclusions(
    datasets: dict[str, DatasetConfig], mount_point: str | None = None
) -> dict[str, list[str]]:
    """Directories whose files discovery never returns, per dataset (spec 4.5).

    - ``code/`` and ``.neurodags/`` under every active dataset's ``derivatives_path``
      (configuration snapshots and neurodags bookkeeping);
    - the dataset's own output tree (``derivatives_path[/derivatives_subdir]``), so a
      chained stage whose ``source_root`` contains its outputs never takes them as
      inputs. Not applied when that tree is ``source_root`` itself or one of its parents
      (the sources would all be dropped); outputs written next to the sources are not
      excluded either.

    Paths that cannot be resolved are ignored here (the configuration checks report
    them). Returns ``{dataset: [normalized directories]}``.
    """
    reserved: list[str] = []
    own: dict[str, str] = {}
    for name, cfg in datasets.items():
        if cfg.skip:
            continue
        try:
            derivatives_path = resolve_dataset_field(cfg, "derivatives_path", mount_point)
        except (ValueError, KeyError):
            continue
        if not derivatives_path:
            continue
        derivatives_path = os.path.normpath(derivatives_path)
        reserved += [
            os.path.join(derivatives_path, "code"),
            os.path.join(derivatives_path, ".neurodags"),
        ]
        try:
            base = effective_derivatives_base(derivatives_path, cfg.derivatives_subdir)
        except ValueError:
            base = derivatives_path
        own[name] = os.path.normpath(base)
    result: dict[str, list[str]] = {}
    for name, cfg in datasets.items():
        dirs = list(dict.fromkeys(reserved))
        if name in own:
            try:
                root = resolve_dataset_field(cfg, "source_root", mount_point)
            except (ValueError, KeyError):
                root = None
            root = os.path.normpath(root) if root else None
            if root is None or not (own[name] == root or _is_below(root, own[name])):
                dirs.append(own[name])
        result[name] = dirs
    return result


def output_tree_warnings(
    datasets: dict[str, DatasetConfig], mount_point: str | None = None
) -> list[str]:
    """``validate`` warnings for datasets whose ``source_root`` contains their outputs (spec 4.5).

    Discovery excludes a dataset's own output tree automatically; the warning asks for an
    explicit ``exclude_pattern`` so the configuration says so too.
    """
    warnings: list[str] = []
    for name, cfg in datasets.items():
        if cfg.skip:
            continue
        try:
            root = resolve_dataset_field(cfg, "source_root", mount_point)
            derivatives_path = resolve_dataset_field(cfg, "derivatives_path", mount_point)
            exclude = resolve_dataset_field(cfg, "exclude_pattern", mount_point)
            base = effective_derivatives_base(derivatives_path, cfg.derivatives_subdir)
        except (ValueError, KeyError):
            continue
        if not root or not base:
            continue
        root, base = os.path.normpath(root), os.path.normpath(base)
        if not _is_below(base, root):
            continue
        if exclude and os.path.normpath(exclude).startswith(base.rstrip(os.sep) + os.sep):
            continue
        warnings.append(
            f"datasets.{name}: source_root {root!r} contains this dataset's own outputs "
            f"({base!r}); discovery excludes them automatically, but add "
            f"exclude_pattern: {os.path.join(base, '**', '*')!r} to make this explicit."
        )
    return warnings


def _discover_dataset(
    dataset_name: str,
    dataset_config: DatasetConfig,
    mount_point: str | None,
    drop_split_continuations: bool,
    problems: list[str],
    excluded_dirs: Iterable[str] = (),
) -> list[tuple[str, str]] | None:
    """Discover one dataset: ``[(rel, path), ...]`` sorted by ``rel``.

    Returns ``None`` when the dataset is skipped or has no pattern. Files outside
    ``source_root`` are appended to *problems* (one message per dataset). Files below
    *excluded_dirs* (:func:`output_tree_exclusions`) are dropped.
    """
    if dataset_config.skip:
        log.info("Skipping dataset (skip=True)", dataset=dataset_name)
        return None

    if not dataset_config.file_pattern:
        log.warning(
            "get_all_files_across_datasets: no pattern defined for dataset",
            dataset=dataset_name,
        )
        return None

    # Relative patterns resolve against the directory of the YAML that defined the
    # dataset (or base_dir= for in-memory mappings), after ~/$VAR expansion (§1.3).
    resolved_pattern = resolve_dataset_field(dataset_config, "file_pattern", mount_point)
    # exclude_pattern is a glob like file_pattern, so it gets the same resolution
    # (and fails loudly on unset variables) outside the catch below.
    exclude_pattern = resolve_dataset_field(dataset_config, "exclude_pattern", mount_point)
    source_root = resolve_dataset_field(dataset_config, "source_root", mount_point)
    if source_root is None:
        raise DatasetConfigError(
            [
                f"datasets.{dataset_name}.source_root is required (since 0.4.0): set it to "
                "the root folder of the dataset's source files"
            ]
        )
    log.debug(
        "get_all_files_across_datasets: resolved pattern",
        dataset=dataset_name,
        pattern=resolved_pattern,
    )

    # Per-dataset opt-out ANDed with the global switch: a dataset can turn
    # split dropping off individually, and the caller can force it off for
    # all datasets.
    dataset_drop_splits = getattr(dataset_config, "drop_split_continuations", True)
    try:
        files = get_files_from_pattern(
            resolved_pattern,
            exclude_filter=exclude_pattern,
            drop_split_continuations=drop_split_continuations and dataset_drop_splits,
        )
    except Exception as e:
        log.error(
            "get_all_files_across_datasets: error getting files from pattern",
            dataset=dataset_name,
            error=str(e),
        )
        return []

    # Deduplicate identical paths, check each one lies below source_root, sort by rel.
    by_rel: dict[str, str] = {}
    outside: list[str] = []
    excluded_dirs = list(excluded_dirs)
    n_excluded = 0
    for f in files:
        path = os.path.normpath(f)
        if any(path == d or _is_below(path, d) for d in excluded_dirs):
            n_excluded += 1
            continue
        try:
            rel = _posix_rel(relative_to_source_root(path, source_root))
        except ValueError:
            outside.append(path)
            continue
        by_rel.setdefault(rel, path)
    if outside:
        outside.sort()
        shown = ", ".join(repr(p) for p in outside[:3])
        more = f" (and {len(outside) - 3} more)" if len(outside) > 3 else ""
        problems.append(
            f"datasets.{dataset_name}: {len(outside)} discovered file(s) are not inside "
            f"source_root {os.path.normpath(source_root)!r}: {shown}{more}. Every discovered "
            "file must lie below its dataset's source_root; fix file_pattern or source_root."
        )
    if n_excluded:
        log.info(
            "Excluded files below output trees from discovery",
            dataset=dataset_name,
            excluded=n_excluded,
        )
    # Code-point order of the POSIX rel: no locale, no natural sort (spec §8).
    return sorted(by_rel.items())


def discover_inputs(
    datasets: dict[str, DatasetConfig],
    mount_point: str | None = None,
    max_files_per_dataset: int | None = None,
    drop_split_continuations: bool = True,
) -> InputEnumeration:
    """Enumerate the inputs of a configuration by discovery (spec §8).

    Per dataset: glob ``file_pattern``, apply ``exclude_pattern``, drop split
    continuations, deduplicate, check every file lies below ``source_root`` (a file
    outside it is a configuration error naming the file), sort by ``rel`` (POSIX
    separators, code-point order), then keep the first *max_files_per_dataset*. Files below
    an output tree (:func:`output_tree_exclusions`) are never discovered. Datasets
    are concatenated in configuration (YAML) order, skipping ``skip: true``; the index is
    the 0-based position.

    Parameters
    ----------
    datasets : dict of str to DatasetConfig
        Dataset configurations in YAML order.
    mount_point : str, optional
        The active mount point.
    max_files_per_dataset : int, optional
        Keep only the first N files (after sorting) of every dataset, so a limited run is
        a prefix of the full run per dataset.
    drop_split_continuations : bool, optional
        Global switch (default True) ANDed with each dataset's own
        ``drop_split_continuations`` field. See :func:`find_split_continuations`.

    Returns
    -------
    InputEnumeration

    Raises
    ------
    DatasetConfigError
        If a dataset has no ``source_root`` or discovers files outside it (all datasets
        are checked and reported together).
    ValueError
        If *max_files_per_dataset* is negative.
    """
    if max_files_per_dataset is not None and max_files_per_dataset < 0:
        raise ValueError(f"max_files_per_dataset must be >= 0, got {max_files_per_dataset}.")
    problems: list[str] = []
    per_dataset: dict[str, list[tuple[str, str]]] = {}
    source_roots: dict[str, str] = {}
    exclusions = output_tree_exclusions(datasets, mount_point)
    for dataset_name, dataset_config in datasets.items():
        log.debug("get_all_files_across_datasets: processing dataset", dataset=dataset_name)
        found = _discover_dataset(
            dataset_name,
            dataset_config,
            mount_point,
            drop_split_continuations,
            problems,
            exclusions.get(dataset_name, ()),
        )
        if not found:
            if found is not None:
                log.warning(
                    "get_all_files_across_datasets: no files found for dataset",
                    dataset=dataset_name,
                )
            continue
        source_roots[dataset_name] = os.path.normpath(
            resolve_dataset_field(dataset_config, "source_root", mount_point)
        )
        if max_files_per_dataset is not None:
            found = found[:max_files_per_dataset]
            log.debug(
                "get_all_files_across_datasets: limited files per dataset",
                dataset=dataset_name,
                max_files=max_files_per_dataset,
            )
        if not found:
            continue
        per_dataset[dataset_name] = found
        log.info("Found files in dataset", dataset=dataset_name, file_count=len(found))
    if problems:
        raise DatasetConfigError(problems)

    entries: list[InputEntry] = []
    for dataset_name, found in per_dataset.items():
        for rel, path in found:
            entries.append(InputEntry(len(entries), dataset_name, rel, path))
    log.info(
        "File discovery complete",
        total_datasets=len(per_dataset),
        total_files=len(entries),
    )
    return InputEnumeration(entries=entries, source_roots=source_roots)


def format_inputs(entries: Iterable[InputEntry]) -> str:
    """Return the text of a frozen input list (header + ``index<TAB>dataset<TAB>rel``)."""
    lines = [INPUTS_HEADER]
    for entry in entries:
        for value, what in ((entry.dataset, "dataset name"), (entry.rel, "file path")):
            if any(ch in value for ch in "\t\r\n"):
                raise InputsFileError(
                    f"cannot write {what} {value!r} to an input list: it contains a tab or "
                    "newline."
                )
        lines.append(f"{entry.index}\t{entry.dataset}\t{entry.rel}")
    return "\n".join(lines) + "\n"


def write_inputs_file(enumeration: InputEnumeration, path: str | os.PathLike) -> str:
    """Write *enumeration* as a frozen input list to *path*; return the absolute path."""
    text = format_inputs(enumeration.entries)
    out = os.path.abspath(os.fspath(path))
    parent = os.path.dirname(out)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    return out


def _check_rel(rel: str) -> bool:
    if not rel or rel.startswith("/") or "\\" in rel or ntpath.splitdrive(rel)[0]:
        return False
    return all(part not in ("", ".", "..") for part in rel.split("/"))


def read_inputs_file(
    path: str | os.PathLike, datasets: dict[str, DatasetConfig]
) -> list[tuple[int, str, str]]:
    """Read and validate a frozen input list; return ``(index, dataset, rel)`` by index.

    Checks: the header (``# neurodags-inputs naming_version=1``), three tab-separated
    fields per line, non-negative integer indices without duplicates and contiguous from
    0, dataset names that exist (and are not ``skip: true``) in *datasets*, and canonical
    relative POSIX paths. Blank lines and ``#`` comment lines after the header are
    ignored.

    Raises
    ------
    InputsFileError
        With every problem found (up to 20 are listed).
    """
    path = os.fspath(path)
    try:
        with open(path, encoding="utf-8") as fh:
            lines = fh.read().splitlines()
    except OSError as exc:
        raise InputsFileError(f"cannot read input list {path!r}: {exc}") from exc
    if not lines or lines[0].strip() != INPUTS_HEADER:
        first = lines[0] if lines else ""
        raise InputsFileError(
            f"{path}: not a neurodags input list; the first line must be {INPUTS_HEADER!r} "
            f"(got {first!r}). Write one with 'neurodags inputs'."
        )

    problems: list[str] = []
    rows: dict[int, tuple[int, str, str]] = {}
    seen: dict[tuple[str, str], int] = {}
    for lineno, line in enumerate(lines[1:], start=2):
        if not line.strip() or line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) != 3:
            problems.append(f"line {lineno}: expected index<TAB>dataset<TAB>rel, got {line!r}")
            continue
        raw_index, dataset, rel = fields
        if not raw_index.isdigit():
            problems.append(f"line {lineno}: index {raw_index!r} is not a non-negative integer")
            continue
        index = int(raw_index)
        if dataset not in datasets:
            problems.append(f"line {lineno}: dataset {dataset!r} is not in the configuration")
        elif datasets[dataset].skip:
            problems.append(
                f"line {lineno}: dataset {dataset!r} is skip: true in the configuration"
            )
        if not _check_rel(rel):
            problems.append(
                f"line {lineno}: path {rel!r} must be a relative POSIX path below "
                "source_root (no '..', '.', empty components or backslashes)"
            )
        if index in rows:
            problems.append(f"line {lineno}: duplicate index {index}")
            continue
        if (dataset, rel) in seen:
            problems.append(
                f"line {lineno}: {dataset}/{rel} is already listed as index {seen[(dataset, rel)]}"
            )
        seen[(dataset, rel)] = index
        rows[index] = (index, dataset, rel)
    if rows and not problems:
        expected = set(range(len(rows)))
        if set(rows) != expected:
            gaps = sorted(expected - set(rows))[:10]
            problems.append(
                f"indices must be contiguous from 0 to {len(rows) - 1}; missing: "
                f"{', '.join(map(str, gaps))}"
            )
    if problems:
        shown = "\n".join(f"  - {p}" for p in problems[:20])
        more = f"\n  ... and {len(problems) - 20} more" if len(problems) > 20 else ""
        raise InputsFileError(f"invalid input list {path}:\n{shown}{more}")
    return [rows[i] for i in sorted(rows)]


def load_inputs(
    inputs: str | os.PathLike,
    datasets: dict[str, DatasetConfig],
    mount_point: str | None = None,
    drop_split_continuations: bool = True,
) -> InputEnumeration:
    """Enumerate the inputs of a configuration from a frozen input list (spec §8.6).

    The list replaces discovery for selection and indexing. Each entry's path is the
    dataset's ``source_root`` on the active mount joined with its ``rel``, so one list
    works on every mount. Listed files that no longer exist are kept at their index with
    ``missing=True``. Discovery still runs once to report files on disk that the list
    does not contain (:attr:`InputEnumeration.not_in_list`); they are not processed.

    Parameters
    ----------
    inputs : path-like
        The input list (relative paths resolve against the current working directory).
    datasets, mount_point, drop_split_continuations
        As for :func:`discover_inputs`.

    Raises
    ------
    InputsFileError
        If the list is malformed or does not match the configuration.
    DatasetConfigError
        As for :func:`discover_inputs`.
    """
    inputs_file = os.path.abspath(os.fspath(inputs))
    rows = read_inputs_file(inputs_file, datasets)

    roots: dict[str, str] = {}
    entries: list[InputEntry] = []
    for index, dataset, rel in rows:
        if dataset not in roots:
            root = resolve_dataset_field(datasets[dataset], "source_root", mount_point)
            if root is None:
                raise DatasetConfigError(
                    [f"datasets.{dataset}.source_root is required (since 0.4.0)"]
                )
            roots[dataset] = os.path.normpath(root)
        path = os.path.join(roots[dataset], *rel.split("/"))
        entries.append(InputEntry(index, dataset, rel, path, missing=not os.path.exists(path)))

    # One discovery pass, only to report files the list does not contain.
    discovered = discover_inputs(
        datasets, mount_point, drop_split_continuations=drop_split_continuations
    )
    listed = {(e.dataset, e.rel) for e in entries}
    not_in_list: dict[str, list[str]] = {}
    for entry in discovered.entries:
        if (entry.dataset, entry.rel) not in listed:
            not_in_list.setdefault(entry.dataset, []).append(entry.rel)

    enumeration = InputEnumeration(
        entries=entries,
        source_roots=roots,
        inputs_file=inputs_file,
        not_in_list=not_in_list,
    )
    if enumeration.missing_sources:
        log.warning(
            "Listed input files no longer exist (missing-source); they are not processed",
            inputs_file=inputs_file,
            count=len(enumeration.missing_sources),
            indices=[e.index for e in enumeration.missing_sources],
        )
    if not_in_list:
        log.warning(
            "Files on disk are not in the input list and are ignored",
            inputs_file=inputs_file,
            count=enumeration.n_not_in_list,
            per_dataset={k: len(v) for k, v in not_in_list.items()},
        )
    log.info("Loaded input list", inputs_file=inputs_file, total_files=len(entries))
    return enumeration


def enumerate_inputs(
    datasets: dict[str, DatasetConfig],
    mount_point: str | None = None,
    *,
    max_files_per_dataset: int | None = None,
    drop_split_continuations: bool = True,
    inputs: str | os.PathLike | None = None,
) -> InputEnumeration:
    """Enumerate the inputs of one invocation: a frozen list if given, else discovery.

    See :func:`discover_inputs` and :func:`load_inputs`.

    Raises
    ------
    InputsFileError
        If *inputs* is combined with *max_files_per_dataset*, or the list is invalid.
    """
    if inputs is not None:
        if max_files_per_dataset is not None:
            raise InputsFileError(
                "max_files_per_dataset cannot be combined with an input list (--inputs): "
                "the list already fixes the inputs."
            )
        return load_inputs(
            inputs, datasets, mount_point, drop_split_continuations=drop_split_continuations
        )
    return discover_inputs(
        datasets,
        mount_point,
        max_files_per_dataset=max_files_per_dataset,
        drop_split_continuations=drop_split_continuations,
    )


def enumerate_pipeline_inputs(
    pipeline_input,
    datasets_input=None,
    *,
    max_files_per_dataset: int | None = None,
    drop_split_continuations: bool = True,
    inputs: str | os.PathLike | None = None,
    base_dir=None,
) -> InputEnumeration:
    """Load the datasets of a pipeline configuration and enumerate its inputs.

    Parameters are those of :func:`get_all_files_from_pipeline_configuration` and
    :func:`enumerate_inputs`.
    """
    datasets, mount_point = get_datasets_and_mount_point_from_pipeline_configuration(
        pipeline_input, datasets_input=datasets_input, base_dir=base_dir
    )
    return enumerate_inputs(
        datasets,
        mount_point,
        max_files_per_dataset=max_files_per_dataset,
        drop_split_continuations=drop_split_continuations,
        inputs=inputs,
    )


def get_all_files_across_datasets(
    datasets: dict[str, DatasetConfig],
    mount_point: str | None = None,
    max_files_per_dataset: int | None = None,
    drop_split_continuations: bool = True,
) -> tuple[dict[str, list[str]], list[tuple[int, str, str]], dict[str, str]]:
    """
    Iterate over all datasets and retrieve files based on their patterns.

    Thin wrapper over :func:`discover_inputs` returning the 0.3.x shapes. Files are
    sorted by their path relative to ``source_root`` and the limit is applied after
    sorting (spec §8).

    Parameters
    ----------
    datasets : dict of str to DatasetConfig
        A dictionary mapping dataset names to their configurations.
    mount_point : str, optional
        The mount point to resolve paths if needed.
    max_files_per_dataset : int, optional
        Maximum number of files to retrieve per dataset (the first N after sorting). If
        None, retrieves all files.
    drop_split_continuations : bool, optional
        Global switch (default True) for dropping split-FIF continuation files.
        The effective per-dataset behaviour is this flag AND the dataset's own
        ``drop_split_continuations`` field, so a dataset can opt out
        individually and this argument can force it off for every dataset.
        See :func:`find_split_continuations`.

    Returns
    -------
    dict of str to list of str
        A dictionary mapping dataset names to lists of file paths.
    list of tuple
        A list of tuples (index, dataset_name, file_path) for all files across datasets
    dict of str to str
        Per-dataset resolved ``source_root`` (the root output locations are computed
        from; this was the discovered set's common root before 0.4.0).
    """
    enumeration = discover_inputs(
        datasets,
        mount_point=mount_point,
        max_files_per_dataset=max_files_per_dataset,
        drop_split_continuations=drop_split_continuations,
    )
    return enumeration.files_per_dataset, enumeration.all_files, enumeration.source_roots


def get_all_files_from_pipeline_configuration(
    pipeline_input,
    datasets_input=None,
    max_files_per_dataset=None,
    drop_split_continuations=True,
    *,
    base_dir=None,
):
    """
    Given a pipeline configuration and an optional datasets configuration,
    retrieve all files across datasets.

    Parameters
    ----------
    pipeline_input : dict | path-like
        A dictionary or path to a YAML file containing pipeline configuration.
    datasets_input : dict | path-like, optional
        A dictionary or path to a YAML file containing datasets configuration.
        If provided, it overrides the 'datasets' section of the pipeline configuration.
    max_files_per_dataset : int, optional
        Maximum number of files to retrieve per dataset. If None, retrieves all files.
    drop_split_continuations : bool, optional
        Global switch (default True) for dropping split-FIF continuation files,
        forwarded to :func:`get_all_files_across_datasets`.
    base_dir : path-like, optional
        Directory that relative dataset paths of in-memory mappings resolve against
        (see :func:`get_datasets_and_mount_point_from_pipeline_configuration`).

    Returns
    -------
    dict of str to list of str
        A dictionary mapping dataset names to lists of file paths.
    list of tuple
        A list of tuples (index, dataset_name, file_path) for all files across datasets
    dict of str to str
        A dictionary mapping dataset names to their resolved ``source_root`` (the
        discovered set's common root before 0.4.0).
    """
    log.debug("get_all_files_from_pipeline_configuration: called", pipeline_input=pipeline_input)
    enumeration = enumerate_pipeline_inputs(
        pipeline_input,
        datasets_input,
        max_files_per_dataset=max_files_per_dataset,
        drop_split_continuations=drop_split_continuations,
        base_dir=base_dir,
    )
    return enumeration.files_per_dataset, enumeration.all_files, enumeration.source_roots
