# tests/test_get_num_digits.py
import pytest
import os
from neurodags.utils import get_num_digits, get_path, find_unique_root, replace_bids_suffix
import ntpath
import posixpath
from pathlib import PureWindowsPath, PurePosixPath
from pathlib import Path

# Tests for get_num_digits function


@pytest.mark.parametrize(
    "n, expected",
    [
        (0, 1),
        (5, 1),
        (9, 1),
        (10, 2),
        (123, 3),
        (-1, 1),
        (-999, 3),
    ],
)
def test_safe_mode_basic(n, expected):
    assert get_num_digits(n, method="safe") == expected


@pytest.mark.parametrize(
    "n, expected",
    [
        (1, 1),
        (9, 1),
        (10, 2),
        (12345, 5),
        (-12345, 5),
    ],
)
def test_fast_mode_basic(n, expected):
    assert get_num_digits(n, method="fast") == expected


def test_zero_in_fast_mode():
    assert get_num_digits(0, method="fast") == 1


def test_large_number_consistency():
    n = 10 ** 50  # 51 digits
    safe = get_num_digits(n, method="safe")
    fast = get_num_digits(n, method="fast")
    # Both should agree
    assert safe == fast == 51


def test_invalid_method_raises():
    with pytest.raises(ValueError) as excinfo:
        get_num_digits(123, method="wrong")
    assert "Unknown method" in str(excinfo.value)


def test_negative_numbers_consistency():
    for n in [-1, -12, -123456]:
        safe = get_num_digits(n, method="safe")
        fast = get_num_digits(n, method="fast")
        assert safe == fast
        assert safe == len(str(abs(n)))


# Tests for get_path function


def test_returns_same_path_for_string_no_mount_point():
    path = "/data/files"
    assert get_path(path) == "/data/files"


def test_returns_same_path_for_string_with_mount_point_ignored():
    path = "/data/files"
    assert get_path(path, mount_point="local") == "/data/files"


def test_returns_mount_pointed_value_when_dict_and_mount_point_present():
    paths = {"local": "/mnt/local/data", "remote": "/mnt/remote/data"}
    assert get_path(paths, mount_point="local") == "/mnt/local/data"
    assert get_path(paths, mount_point="remote") == "/mnt/remote/data"


def test_raises_keyerror_when_mount_point_missing_in_dict():
    paths = {"local": "/mnt/local/data"}
    with pytest.raises(KeyError):
        _ = get_path(paths, mount_point="remote")


def test_returns_dict_when_mount_point_is_none_and_path_is_dict():
    # NOTE: Given the current implementation, if `path` is a dict and `mount_point` is None,
    # the function returns the dictionary itself.
    # If you want it to *always* return a string, update the implementation accordingly.
    paths = {"local": "/mnt/local/data"}
    assert get_path(paths, mount_point=None) is paths


# Helper functions for testing find_unique_root


def _pick_flavor(paths) -> tuple:
    """Return (pmod, PurePath, use_windows) based on the inputs."""
    s = [os.fspath(p) for p in paths]
    looks_win = [("\\" in x) or (len(x) >= 2 and x[1] == ":") for x in s]
    use_windows = all(looks_win)
    pmod = ntpath if use_windows else posixpath
    PurePath = PureWindowsPath if use_windows else PurePosixPath
    return pmod, PurePath, use_windows


def _is_prefix_path(prefix, path) -> bool:
    pmod, _, _ = _pick_flavor([prefix, path])
    prefix = pmod.normpath(os.fspath(prefix))
    path = pmod.normpath(os.fspath(path))
    try:
        common = pmod.commonpath([prefix, path])
    except ValueError:
        # e.g., different drives on Windows → no common path
        return False
    return common == prefix


def _relpaths_from_root(root, filepaths) -> list[str]:
    pmod, _, _ = _pick_flavor([root, *filepaths])
    root = pmod.normpath(os.fspath(root))
    return [pmod.relpath(pmod.normpath(os.fspath(p)), root) for p in filepaths]


def _has_unique_relpaths(root, filepaths) -> bool:
    rels = _relpaths_from_root(root, filepaths)
    return len(rels) == len(set(rels))


def _parent_dir(path) -> str | None:
    pmod, PurePath, use_windows = _pick_flavor([path])
    path = pmod.normpath(os.fspath(path))
    parent = pmod.dirname(path)

    # Treat filesystem roots as having no parent
    if parent == path:
        return None
    if use_windows:
        drv = ntpath.splitdrive(path)[0]
        # Drive root like "C:\" (normalized) has no parent
        if path in (drv + "\\", drv + "/"):
            return None
    else:
        if path == "/":
            return None
    return parent
    return None if parent == path else parent


# Tests for find_unique_root function


@pytest.mark.parametrize(
    "filepaths",
    [
        # Same directory, different filenames
        ["/data/p/x.txt", "/data/p/y.txt"],
        # Different directories, same filename
        ["/data/project1/images/cat.png", "/data/project2/images/cat.png"],
        # Deeper structures that only diverge late
        ["/data/A/B/C/file.bin", "/data/A/B/D/file2.bin", "/data/A/E/F/file3.bin"],
        # Mixed files across a broader tree
        ["/mnt/a/x/a.txt", "/mnt/a/y/b.txt", "/mnt/b/x/c.txt", "/mnt/b/z/d.txt"],
    ],
)
def test_minimal_unique_root_properties_posix(filepaths, monkeypatch):
    # Force POSIX-style behavior for consistency on all OSes during tests
    monkeypatch.setenv("PYTHONIOENCODING", "utf-8")

    # Precondition: all absolute POSIX-like paths
    assert all(p.startswith("/") for p in filepaths)

    root = find_unique_root(filepaths)

    # 1) Root must be a common ancestor of all files
    assert all(
        _is_prefix_path(root, p) for p in filepaths
    ), "Returned root is not a common ancestor."

    # 2) Relative paths from root must be unique
    assert _has_unique_relpaths(
        root, filepaths
    ), "Relative paths are not unique from the returned root."

    # 3) Minimality: parent of root must NOT satisfy uniqueness (or root is already top-level)
    parent = _parent_dir(root)
    if parent is not None:
        assert not _has_unique_relpaths(
            parent, filepaths
        ), "Returned root is not minimal: its parent still yields unique relative paths."


def test_identical_paths_edge_case():
    # When inputs contain identical paths, uniqueness is impossible from any higher root.
    # The function should fall back to the full common path (which equals that path).
    filepaths = ["/data/p/x.txt", "/data/p/x.txt"]
    root = find_unique_root(filepaths)

    # Must be a common ancestor (trivially true if equals the path)
    assert _is_prefix_path(root, filepaths[0])

    # From that root, relpaths are not unique — that's okay; the function promises fallback.
    # We only require that the returned root equals the full common path in this scenario.
    common = os.path.commonpath(filepaths)
    assert os.path.normpath(root) == os.path.normpath(common)


def test_single_path_trivial_case():
    # With a single path, the minimal unique root can be the path itself
    # (since there is nothing to disambiguate).
    filepaths = ["/only/one/file.txt"]
    root = find_unique_root(filepaths)

    assert _is_prefix_path(root, filepaths[0])
    # Relative path uniqueness is trivially satisfied
    assert _has_unique_relpaths(root, filepaths)


@pytest.mark.parametrize(
    "filepaths",
    [
        ["C:\\data\\p\\x.txt", "C:\\data\\p\\y.txt"],
        ["C:\\data\\proj1\\images\\cat.png", "C:\\data\\proj2\\images\\cat.png"],
    ],
)
def test_minimal_unique_root_windows_like_paths(filepaths):
    # These are Windows-like paths; the implementation should handle them
    # if it relies on os.path/commonpath properly. If your implementation
    # is POSIX-only, mark these xfail or normalize inputs before calling.
    root = find_unique_root(filepaths)

    # Common-ancestor + uniqueness properties
    assert all(_is_prefix_path(root, p) for p in filepaths)
    assert _has_unique_relpaths(root, filepaths)

    # Minimality: parent should fail uniqueness (or root has no parent)
    parent = _parent_dir(root)
    if parent is not None:
        assert not _has_unique_relpaths(parent, filepaths)


# --- Error conditions ---


def test_empty_input_raises():
    with pytest.raises(ValueError, match="filepaths must be non-empty"):
        find_unique_root([])


def test_mixed_styles_strict_rejects():
    # One POSIX, one Windows → should raise with strict=True
    filepaths = ["/data/x.txt", "C:\\data\\y.txt"]
    with pytest.raises(ValueError, match="Mixed POSIX/Windows styles"):
        find_unique_root(filepaths, style="auto", strict=True)


def test_invalid_style():
    with pytest.raises(ValueError, match="style must be 'auto'"):
        find_unique_root(["/data/x.txt"], style="bogus")


def test_backslash_in_posix_strict_raises():
    # Backslash in POSIX path with strict=True
    filepaths = ["/data\\bad.txt"]
    with pytest.raises(ValueError, match="Backslash found in POSIX path"):
        find_unique_root(filepaths, style="posix", strict=True)


def test_invalid_mode():
    with pytest.raises(ValueError, match="mode must be 'minimal' or 'maximal'"):
        find_unique_root(["/data/x.txt"], mode="wrong")


# --- Explicit style selection ---


def test_explicit_windows_style(monkeypatch):
    # Even if POSIX-looking, force Windows behavior
    filepaths = ["C:\\proj\\a.txt", "C:\\proj\\b.txt"]
    root = find_unique_root(filepaths, style="windows")
    assert root.lower().startswith("c:\\"), f"Got {root} instead of C:\\..."


def test_explicit_posix_style():
    filepaths = ["/a/b/file1.txt", "/a/b/file2.txt"]
    root = find_unique_root(filepaths, style="posix")
    # Minimal unique root can be "/" since relpaths are still unique there
    assert root in ["/", "/a/b"]


# --- Lenient coercion ---


def test_lenient_mixed_styles_coerces_to_windows():
    filepaths = ["/data/x.txt", "C:\\data\\y.txt"]
    root = find_unique_root(filepaths, style="auto", strict=False)
    # Different drives → no common root, function falls back to ""
    assert root == ""


def test_lenient_backslash_coerced_to_slash():
    filepaths = ["/data\\x.txt", "/data\\y.txt"]
    root = find_unique_root(filepaths, style="posix", strict=False)
    # Minimal root is "/" since relpaths are unique from root
    assert root == "/"


# --- Edge: different drives on Windows → no common path ---


def test_different_drives_windows_no_common(monkeypatch):
    filepaths = ["C:\\a\\x.txt", "D:\\b\\y.txt"]
    root = find_unique_root(filepaths, style="windows")
    # Expected fallback: no common prefix → empty string
    assert root == ""


# --- Maximal mode ---


def test_maximal_mode_returns_deepest():
    filepaths = ["/data/project1/a.txt", "/data/project2/b.txt"]
    root_min = find_unique_root(filepaths, mode="minimal")
    root_max = find_unique_root(filepaths, mode="maximal")
    # Minimal should be "/" and maximal should be "/data"
    assert root_min == "/"
    assert root_max == "/data"
    assert root_max != root_min


# Replace BIDS suffix tests


@pytest.mark.parametrize(
    "input_path,new_suffix,new_ext,expected",
    [
        # Case with underscore and multi-part extension
        (
            "sub-01_task-rest_bold.nii.gz",
            "desc-preproc",
            ".nii.gz",
            "sub-01_task-rest_desc-preproc.nii.gz",
        ),
        # Case with single extension
        ("sub-02_eeg.set", "clean", ".fdt", "sub-02_clean.fdt"),
        # No underscore in base
        ("dataset.json", "metadata", ".json", "dataset_metadata.json"),
        # No extension at all
        ("file_without_ext", "suffix", ".txt", "file_without_ext_suffix.txt"),
    ],
)
def test_replace_bids_suffix(input_path, new_suffix, new_ext, expected):
    result = replace_bids_suffix(input_path, new_suffix, new_ext)
    assert isinstance(result, Path)
    assert (
        result.name == expected
    ), f"Failed for input: {input_path}. Result: {result.name}, Expected: {expected}"


def test_handles_multiparts():
    path = "sub-03_ses-01_bold.nii.gz"
    result = replace_bids_suffix(path, "desc-cleaned", ".nii.gz")
    assert result.name == "sub-03_ses-01_desc-cleaned.nii.gz"


def test_no_suffix():
    path = "plainfile"
    result = replace_bids_suffix(path, "extra", ".json")
    assert result.name == "plainfile_extra.json"


if __name__ == "__main__":
    # pytest.main([__file__])
    test_identical_paths_edge_case()
