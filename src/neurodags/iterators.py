import glob
import os
import re

from neurodags.datasets import get_datasets_and_mount_point_from_pipeline_configuration
from neurodags.definitions import DatasetConfig
from neurodags.loggers import get_logger
from neurodags.utils import find_unique_root, resolve_dataset_field

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
        A list of file paths matching the pattern.
    """

    log.debug("get_files_from_pattern: called", pattern=pattern, recursive=recursive)
    files = glob.glob(pattern, recursive=recursive)
    log.debug("get_files_from_pattern: found files", count=len(files))

    # Apply exclude filter if provided
    if exclude_filter:
        log.debug("get_files_from_pattern: applying exclude filter", exclude_filter=exclude_filter)
        excluded_files = set(glob.glob(exclude_filter, recursive=recursive))
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


def get_all_files_across_datasets(
    datasets: dict[str, DatasetConfig],
    mount_point: str | None = None,
    max_files_per_dataset: int | None = None,
    drop_split_continuations: bool = True,
) -> dict[str, list[str]]:
    """
    Iterate over all datasets and retrieve files based on their patterns.

    Parameters
    ----------
    datasets : dict of str to DatasetConfig
        A dictionary mapping dataset names to their configurations.
    mount_point : str, optional
        The mount point to resolve paths if needed.
    max_files_per_dataset : int, optional
        Maximum number of files to retrieve per dataset. If None, retrieves all files.
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
    """

    files_per_dataset = {}
    common_roots = {}
    for dataset_name, dataset_config in datasets.items():
        log.debug("get_all_files_across_datasets: processing dataset", dataset=dataset_name)

        if dataset_config.skip:
            log.info(
                "Skipping dataset (skip=True)",
                dataset=dataset_name,
            )
            continue

        pattern = dataset_config.file_pattern
        if not pattern:
            log.warning(
                "get_all_files_across_datasets: no pattern defined for dataset",
                dataset=dataset_name,
            )
            continue

        # Relative patterns resolve against the directory of the YAML that defined the
        # dataset (or base_dir= for in-memory mappings), after ~/$VAR expansion (§1.3).
        resolved_pattern = resolve_dataset_field(dataset_config, "file_pattern", mount_point)
        # exclude_pattern is a glob like file_pattern, so it gets the same resolution
        # (and fails loudly on unset variables) outside the catch below.
        exclude_pattern = resolve_dataset_field(dataset_config, "exclude_pattern", mount_point)
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
            continue

        if max_files_per_dataset is not None:
            files = files[:max_files_per_dataset]
            log.debug(
                "get_all_files_across_datasets: limited files per dataset",
                dataset=dataset_name,
                max_files=max_files_per_dataset,
            )

        if not files:
            log.warning(
                "get_all_files_across_datasets: no files found for dataset", dataset=dataset_name
            )
            continue

        files_per_dataset[dataset_name] = files
        common_root = find_unique_root(files, mode="maximal")
        common_roots[dataset_name] = common_root
        log.debug(
            "get_all_files_across_datasets: common root for dataset",
            dataset=dataset_name,
            common_root=common_root,
        )
        log.info(
            "Found files in dataset",
            dataset=dataset_name,
            file_count=len(files),
        )
    all_files = []
    for dataset, files in files_per_dataset.items():
        log.debug(
            "get_all_files_across_datasets: dataset summary", dataset=dataset, file_count=len(files)
        )
        these_files = [(dataset, f) for f in files]
        all_files.extend(these_files)
    # add index to each file tuple
    all_files = [(i, dataset, f) for i, (dataset, f) in enumerate(all_files)]
    log.info(
        "File discovery complete",
        total_datasets=len(files_per_dataset),
        total_files=len(all_files),
    )
    return files_per_dataset, all_files, common_roots


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
        A dictionary mapping dataset names to their common roots.
    """
    log.debug("get_all_files_from_pipeline_configuration: called", pipeline_input=pipeline_input)
    datasets, mount_point = get_datasets_and_mount_point_from_pipeline_configuration(
        pipeline_input, datasets_input=datasets_input, base_dir=base_dir
    )

    files_per_dataset, all_files, common_roots = get_all_files_across_datasets(
        datasets,
        mount_point=mount_point,
        max_files_per_dataset=max_files_per_dataset,
        drop_split_continuations=drop_split_continuations,
    )
    log.debug(
        "get_all_files_from_pipeline_configuration: completed",
        total_datasets=len(files_per_dataset),
        total_files=len(all_files),
    )

    return files_per_dataset, all_files, common_roots
