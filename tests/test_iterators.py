# Tests for neurodags.iterators split-FIF scanning
from __future__ import annotations

from pathlib import Path

import pytest

from neurodags import iterators
from neurodags.definitions import DatasetConfig
from neurodags.iterators import (
    find_split_continuations,
    get_all_files_across_datasets,
    get_files_from_pattern,
)

# ---------------------------------------------------------------------------
# find_split_continuations — pure filename logic
# ---------------------------------------------------------------------------


def test_bids_split_keeps_entry_drops_continuations():
    files = [
        "/d/sub-01_task-rest_split-01_meg.fif",
        "/d/sub-01_task-rest_split-02_meg.fif",
        "/d/sub-01_task-rest_split-03_meg.fif",
    ]
    drop = find_split_continuations(files)
    assert drop == {
        "/d/sub-01_task-rest_split-02_meg.fif",
        "/d/sub-01_task-rest_split-03_meg.fif",
    }


def test_bids_split_two_digit_index():
    # split-01 is the entry; split-02..split-10 are continuations
    files = [f"/d/x_split-{i:02}_meg.fif" for i in range(1, 11)]
    drop = find_split_continuations(files)
    kept = set(files) - drop
    assert kept == {"/d/x_split-01_meg.fif"}
    assert len(drop) == 9


def test_neuromag_split_keeps_entry_drops_continuations():
    files = ["/d/rest_raw.fif", "/d/rest_raw-1.fif", "/d/rest_raw-2.fif"]
    drop = find_split_continuations(files)
    assert drop == {"/d/rest_raw-1.fif", "/d/rest_raw-2.fif"}


def test_neuromag_gz_split():
    # mne names gz continuations as name.fif-1.gz (entry name.fif.gz)
    files = ["/d/rest.fif.gz", "/d/rest.fif-1.gz", "/d/rest.fif-2.gz"]
    drop = find_split_continuations(files)
    assert drop == {"/d/rest.fif-1.gz", "/d/rest.fif-2.gz"}


def test_no_split_unchanged():
    files = ["/d/sub-01_task-rest_meg.fif", "/d/sub-02_task-rest_meg.fif"]
    assert find_split_continuations(files) == set()


def test_legit_run_entity_not_dropped():
    # run-2 lives before the _meg suffix, so the stem does not end in -<digit>
    files = ["/d/sub-01_run-1_meg.fif", "/d/sub-01_run-2_meg.fif"]
    assert find_split_continuations(files) == set()


def test_standalone_dash_number_without_entry_not_dropped():
    # foo-2.fif with no foo.fif sibling is NOT a continuation
    assert find_split_continuations(["/d/foo-2.fif"]) == set()
    # but with the entry present it IS a continuation
    assert find_split_continuations(["/d/foo.fif", "/d/foo-2.fif"]) == {"/d/foo-2.fif"}


def test_neuromag_only_applies_to_fif():
    # a non-fif -N file with an entry sibling must be left alone
    files = ["/d/img.png", "/d/img-1.png"]
    assert find_split_continuations(files) == set()


def test_lone_bids_split_01_kept():
    # a lone _split-01_ (no siblings) has nothing to stitch to — keep it
    assert find_split_continuations(["/d/sub-01_task-rest_split-01_meg.fif"]) == set()


def test_split_sets_isolated_per_directory():
    # identical split names in different dirs are independent groups
    files = [
        "/a/x_split-01_meg.fif",
        "/a/x_split-02_meg.fif",
        "/b/x_split-01_meg.fif",
        "/b/x_split-02_meg.fif",
    ]
    drop = find_split_continuations(files)
    assert drop == {"/a/x_split-02_meg.fif", "/b/x_split-02_meg.fif"}


# ---------------------------------------------------------------------------
# get_files_from_pattern — end-to-end on disk, incl. opt-out flag
# ---------------------------------------------------------------------------


def _touch(p: Path) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"\x00")
    return p


def test_get_files_from_pattern_drops_bids_splits(tmp_path: Path):
    d = tmp_path / "sub-01"
    for name in (
        "sub-01_task-rest_split-01_meg.fif",
        "sub-01_task-rest_split-02_meg.fif",
        "sub-01_task-rest_split-03_meg.fif",
    ):
        _touch(d / name)

    found = get_files_from_pattern(str(tmp_path / "**" / "*.fif"))
    assert [Path(f).name for f in found] == ["sub-01_task-rest_split-01_meg.fif"]


def test_get_files_from_pattern_drops_neuromag_splits(tmp_path: Path):
    d = tmp_path / "rec"
    for name in ("rest_raw.fif", "rest_raw-1.fif", "rest_raw-2.fif"):
        _touch(d / name)

    found = get_files_from_pattern(str(tmp_path / "**" / "*.fif"))
    assert [Path(f).name for f in found] == ["rest_raw.fif"]


def test_get_files_from_pattern_opt_out_returns_all(tmp_path: Path):
    d = tmp_path / "sub-01"
    for name in (
        "sub-01_task-rest_split-01_meg.fif",
        "sub-01_task-rest_split-02_meg.fif",
    ):
        _touch(d / name)

    found = get_files_from_pattern(str(tmp_path / "**" / "*.fif"), drop_split_continuations=False)
    assert len(found) == 2


def test_get_files_from_pattern_no_split_unchanged(tmp_path: Path):
    d = tmp_path / "raw"
    for name in ("a_meg.fif", "b_meg.fif"):
        _touch(d / name)

    found = get_files_from_pattern(str(tmp_path / "**" / "*.fif"))
    assert sorted(Path(f).name for f in found) == ["a_meg.fif", "b_meg.fif"]


# ---------------------------------------------------------------------------
# get_all_files_across_datasets — flag threading & per-dataset opt-out
# ---------------------------------------------------------------------------


def _make_split_dataset(tmp_path: Path) -> str:
    d = tmp_path / "sub-01"
    for name in (
        "sub-01_task-rest_split-01_meg.fif",
        "sub-01_task-rest_split-02_meg.fif",
        "sub-01_task-rest_split-03_meg.fif",
    ):
        _touch(d / name)
    return str(tmp_path / "**" / "*.fif")


def test_datasets_default_drops_splits(tmp_path: Path):
    pattern = _make_split_dataset(tmp_path)
    datasets = {"ds": DatasetConfig(name="ds", source_root=str(tmp_path), file_pattern=pattern)}
    _, all_files, _ = get_all_files_across_datasets(datasets)
    assert len(all_files) == 1


def test_datasets_global_optout(tmp_path: Path):
    pattern = _make_split_dataset(tmp_path)
    datasets = {"ds": DatasetConfig(name="ds", source_root=str(tmp_path), file_pattern=pattern)}
    _, all_files, _ = get_all_files_across_datasets(datasets, drop_split_continuations=False)
    assert len(all_files) == 3


def test_datasets_per_dataset_optout(tmp_path: Path):
    pattern = _make_split_dataset(tmp_path)
    datasets = {
        "ds": DatasetConfig(
            name="ds",
            source_root=str(tmp_path),
            file_pattern=pattern,
            drop_split_continuations=False,
        )
    }
    _, all_files, _ = get_all_files_across_datasets(datasets)
    assert len(all_files) == 3


# ---------------------------------------------------------------------------
# get_files_from_pattern — exclude_filter
# ---------------------------------------------------------------------------


def test_get_files_from_pattern_exclude_filter(tmp_path: Path):
    d = tmp_path / "raw"
    _touch(d / "keep_meg.fif")
    _touch(d / "drop_meg.fif")

    found = get_files_from_pattern(
        str(tmp_path / "**" / "*.fif"),
        exclude_filter=str(tmp_path / "**" / "drop_*.fif"),
    )
    assert [Path(f).name for f in found] == ["keep_meg.fif"]


# ---------------------------------------------------------------------------
# get_all_files_across_datasets — skip / no-pattern / error / max_files
# ---------------------------------------------------------------------------


def _make_plain_dataset(tmp_path: Path, n: int = 3) -> str:
    d = tmp_path / "raw"
    for i in range(n):
        _touch(d / f"rec-{i}_meg.fif")
    return str(tmp_path / "**" / "*.fif")


def test_datasets_skip_true_is_skipped(tmp_path: Path):
    pattern = _make_plain_dataset(tmp_path)
    datasets = {
        "ds": DatasetConfig(name="ds", source_root=str(tmp_path), file_pattern=pattern, skip=True)
    }
    files_per_dataset, all_files, _ = get_all_files_across_datasets(datasets)
    assert files_per_dataset == {}
    assert all_files == []


def test_datasets_empty_pattern_is_skipped(tmp_path: Path):
    datasets = {"ds": DatasetConfig(name="ds", source_root=str(tmp_path), file_pattern="")}
    files_per_dataset, all_files, _ = get_all_files_across_datasets(datasets)
    assert files_per_dataset == {}
    assert all_files == []


def test_datasets_pattern_error_is_skipped(tmp_path: Path, monkeypatch):
    pattern = _make_plain_dataset(tmp_path)
    datasets = {"ds": DatasetConfig(name="ds", source_root=str(tmp_path), file_pattern=pattern)}

    def boom(*args, **kwargs):
        raise RuntimeError("glob blew up")

    monkeypatch.setattr(iterators, "get_files_from_pattern", boom)
    files_per_dataset, all_files, _ = get_all_files_across_datasets(datasets)
    assert files_per_dataset == {}
    assert all_files == []


def test_datasets_no_files_found_is_skipped(tmp_path: Path):
    # Pattern matches nothing -> dataset produces no files and is skipped
    datasets = {
        "ds": DatasetConfig(
            name="ds", source_root=str(tmp_path), file_pattern=str(tmp_path / "**" / "*.fif")
        )
    }
    files_per_dataset, all_files, _ = get_all_files_across_datasets(datasets)
    assert files_per_dataset == {}
    assert all_files == []


def test_datasets_max_files_per_dataset(tmp_path: Path):
    pattern = _make_plain_dataset(tmp_path, n=3)
    datasets = {"ds": DatasetConfig(name="ds", source_root=str(tmp_path), file_pattern=pattern)}
    files_per_dataset, all_files, _ = get_all_files_across_datasets(
        datasets, max_files_per_dataset=1
    )
    assert len(files_per_dataset["ds"]) == 1
    assert len(all_files) == 1


if __name__ == "__main__":
    pytest.main(["-v", "-s", "-q", "--no-cov", __file__])
