import math
import ntpath
import os
import posixpath
import re
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Literal

import numpy as np

from neurodags.definitions import PathLike
from neurodags.loggers import get_logger

log = get_logger(__name__)


def get_num_digits(n: int, method: Literal["safe", "fast"] = "safe") -> int:
    """Return the number of decimal digits in an integer.

    Parameters
    ----------
    n : int
        The integer whose digits should be counted.
    method : {"safe", "fast"}, default="safe"
        - "safe": Uses string conversion (`len(str(abs(n)))`).
          Always correct, even for very large integers.
        - "fast": Uses ``log10``.
          Faster for large numbers, but depends on floating-point precision.
          Still safe for most practical integer ranges.

    Returns
    -------
    int
        The number of digits in the absolute value of ``n``.

    Examples
    --------
    >>> get_num_digits(0)
    1
    >>> get_num_digits(12345)
    5
    >>> get_num_digits(-999, method="fast")
    3
    """
    log.debug("get_num_digits: called", n=n, method=method)

    if n == 0:
        log.debug("get_num_digits: zero has one digit")
        return 1

    n_abs = abs(n)

    if method == "safe":
        digits = len(str(n_abs))
        log.debug("get_num_digits: computed (safe)", n=n, digits=digits)
        return digits

    if method == "fast":
        digits = int(math.log10(n_abs)) + 1
        log.debug("get_num_digits: computed (fast)", n=n, digits=digits)
        return digits

    log.debug("get_num_digits: invalid method", method=method)
    raise ValueError(f"Unknown method: {method!r}. Use 'safe' or 'fast'.")


_POSIX_VAR_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)")
_WINDOWS_VAR_RE = re.compile(r"%([A-Za-z_][A-Za-z0-9_]*)%")


def _referenced_env_vars(path: str) -> list[str]:
    """Return the environment-variable names referenced in *path*, in order.

    Recognises ``$NAME`` and ``${NAME}`` everywhere and ``%NAME%`` on Windows.
    A ``$`` that is not followed by an identifier (``a$1``, ``foo$``) is not a
    reference.
    """
    names = [braced or bare for braced, bare in _POSIX_VAR_RE.findall(path)]
    if os.name == "nt":
        names.extend(_WINDOWS_VAR_RE.findall(path))
    return list(dict.fromkeys(names))


def expand_path(path: str, *, field: str = "path") -> str:
    """Expand ``~`` and environment variables in a dataset path, failing loudly.

    The path is expanded with ``os.path.expandvars(os.path.expanduser(path))``.
    ``$NAME`` and ``${NAME}`` are recognised (plus ``%NAME%`` on Windows). Unlike
    :func:`os.path.expandvars`, a referenced variable that is unset **or empty**
    raises instead of being left in place (or silently collapsing), so neurodags
    never creates a directory literally named ``$NAME`` or ``~``.

    Parameters
    ----------
    path : str
        The path (or glob pattern) to expand.
    field : str, default="path"
        Name of the configuration field, used in error messages
        (e.g. ``"derivatives_path"``).

    Returns
    -------
    str
        The expanded path.

    Raises
    ------
    ValueError
        If a referenced environment variable is unset or empty, or if a leading
        ``~`` cannot be expanded.

    Examples
    --------
    With ``SCRATCH=/scratch/me`` exported, ``expand_path("$SCRATCH/derivatives")``
    returns ``'/scratch/me/derivatives'``. A ``$`` not followed by an identifier is
    left alone:

    >>> expand_path("data/a$1")
    'data/a$1'
    """
    unset: list[str] = []
    empty: list[str] = []
    for name in _referenced_env_vars(path):
        value = os.environ.get(name)
        if value is None:
            unset.append(name)
        elif value == "":
            empty.append(name)
    if unset or empty:
        problems = []
        if unset:
            problems.append("unset: " + ", ".join(unset))
        if empty:
            problems.append("empty: " + ", ".join(empty))
        raise ValueError(
            f"{field} {path!r} references environment variable(s) that are "
            f"{'; '.join(problems)}. Set them before running, or write the path "
            "literally. (neurodags refuses to create a directory named after an "
            "unresolved variable.)"
        )

    expanded = os.path.expandvars(os.path.expanduser(path))
    if expanded.startswith("~"):
        raise ValueError(
            f"{field} {path!r} expands to {expanded!r}, which still starts with '~' "
            "(the home directory could not be resolved). Use an absolute path, or "
            "'./~...' if you really mean a directory named '~'."
        )
    if expanded != path:
        log.debug("expand_path: expanded", field=field, path=path, expanded=expanded)
    return expanded


def get_path(
    path: str | dict[str, str],
    mount_point: str | None = None,
    *,
    field: str = "path",
) -> str:
    """
    Retrieve the resolved path based on a given mount_point key or return the path directly.

    The selected string is passed through :func:`expand_path`, so ``~`` and
    ``$VAR``/``${VAR}`` are expanded from the environment and an unset or empty
    variable raises ``ValueError``. Only the value for the active mount point is
    expanded: inactive entries may reference variables that exist only on other
    machines.

    Parameters
    ----------
    path : str or dict of str
        A file path string or a dictionary mapping mount_point names to paths.
    mount_point : str, optional
        The mount_point key to use if ``path`` is a dictionary. If None or if ``path``
        is not a dictionary, the function returns ``path`` directly (a dict is
        returned unchanged when ``mount_point`` is None).
    field : str, default="path"
        Name of the configuration field, used in error messages.

    Returns
    -------
    str
        The resolved path.

    Raises
    ------
    KeyError
        If ``mount_point`` is provided but not found in the dictionary ``path``.
    ValueError
        If the selected path references an unset/empty environment variable or a
        ``~`` that cannot be expanded.

    Examples
    --------
    >>> get_path("/data/files")
    '/data/files'

    >>> paths = {"local": "/mnt/local/data", "remote": "/mnt/remote/data"}
    >>> get_path(paths, mount_point="local")
    '/mnt/local/data'
    """
    log.debug("get_path: called", mount_point=mount_point, path_type=type(path).__name__)
    if isinstance(path, dict) and mount_point is not None:
        try:
            resolved = path[mount_point]
        except KeyError:
            log.debug(
                "get_path: missing mount_point key",
                mount_point=mount_point,
                available=list(path.keys()),
            )
            raise
        log.debug("get_path: resolved via mount_point", mount_point=mount_point, resolved=resolved)
        if isinstance(resolved, str):
            resolved = expand_path(resolved, field=field)
        return resolved
    log.debug("get_path: returning direct path", path=path)
    if isinstance(path, str):
        return expand_path(path, field=field)
    return path


class DatasetConfigError(ValueError):
    """One or more dataset definitions are invalid (all problems are listed together)."""

    def __init__(self, problems: Sequence[str]):
        self.problems = list(problems)
        lines = "\n".join(f"  - {p}" for p in self.problems)
        super().__init__(f"invalid dataset configuration:\n{lines}")


#: Path fields of a dataset that are resolved like ``file_pattern`` (spec §1.3).
DATASET_PATH_FIELDS = ("source_root", "file_pattern", "exclude_pattern", "derivatives_path")

_GLOB_CHARS = ("*", "?", "[")


def resolve_config_path(
    path: str | dict[str, str],
    mount_point: str | None = None,
    *,
    field: str = "path",
    base_dir: PathLike | None = None,
) -> str:
    """Select, expand and absolutize a configured path (spec §1.3).

    The value for the active mount point is selected and ``~``/``$VARS`` are expanded
    (:func:`get_path`, 0.3.1 rules). A path that is absolute after expansion is used as
    is; a relative one is joined onto *base_dir* (the directory of the YAML file that
    defined it, or the ``base_dir=`` given to the Python API) and normalized lexically
    (never ``realpath``).

    Raises
    ------
    ValueError
        If *path* is a mount-point map but no mount point is active, if a variable is
        unset/empty, or if the expanded path is relative and *base_dir* is ``None``.
    KeyError
        If the mount-point map has no entry for *mount_point*.
    """
    if isinstance(path, dict) and mount_point is None:
        raise ValueError(
            f"{field} is a mount-point map ({', '.join(map(str, path))}) but the pipeline "
            "configuration sets no 'mount_point'."
        )
    resolved = get_path(path, mount_point=mount_point, field=field)
    if not isinstance(resolved, str):
        raise ValueError(f"{field} must be a string, got {type(resolved).__name__}.")
    if os.path.isabs(resolved):
        return resolved
    if base_dir is None:
        raise ValueError(
            f"{field} {resolved!r} is a relative path, but the dataset configuration was "
            "given as an in-memory mapping without a base directory. Pass base_dir=... "
            "(relative paths in a mapping are resolved against it) or use absolute paths."
        )
    return os.path.normpath(os.path.join(os.fspath(base_dir), resolved))


def resolve_dataset_field(
    dataset_config: Any, attr: str, mount_point: str | None = None
) -> str | None:
    """Resolve one path field of a dataset (``None`` when unset); see :func:`resolve_config_path`.

    Relative values resolve against the directory recorded on the dataset when it was
    loaded (the datasets YAML, the pipeline YAML for an inline mapping, or ``base_dir=``).
    """
    value = getattr(dataset_config, attr, None)
    if not value:
        return None
    name = getattr(dataset_config, "name", "?")
    return resolve_config_path(
        value,
        mount_point,
        field=f"datasets.{name}.{attr}",
        base_dir=getattr(dataset_config, "_base_dir", None),
    )


def suggest_source_root(file_pattern: str) -> str | None:
    """Suggest a ``source_root`` from an expanded ``file_pattern`` (spec §1.2).

    Returns the leading path components before the first component containing ``*``,
    ``?`` or ``[``; a pattern without wildcards suggests its parent directory. A
    suggestion at a filesystem root (``/``, ``C:\\``) or an empty one is never offered
    (``None``): the user must choose.

    Examples
    --------
    >>> suggest_source_root("/data/study/rawdata/**/*.vhdr")
    '/data/study/rawdata'
    >>> suggest_source_root("/data/study/sub-01/eeg.vhdr")
    '/data/study/sub-01'
    >>> suggest_source_root("/*/x.vhdr") is None
    True
    """
    if not file_pattern:
        return None
    is_windows = "\\" in file_pattern and "/" not in file_pattern
    mod = ntpath if is_windows else posixpath
    sep = "\\" if is_windows else "/"
    drive, rest = mod.splitdrive(file_pattern)
    absolute = rest.startswith(sep)
    parts = [p for p in rest.split(sep) if p]
    fixed: list[str] = []
    for part in parts:
        if any(ch in part for ch in _GLOB_CHARS):
            break
        fixed.append(part)
    else:
        # No wildcard at all: the pattern names one file; suggest its directory.
        fixed = fixed[:-1]
    if not fixed:
        return None
    joined = sep.join(fixed)
    return f"{drive}{sep if absolute else ''}{joined}"


def lexical_path_key(path: PathLike) -> str:
    """The lexical identity of *path*: the key discovery, exclusions and containment compare.

    :func:`os.path.normpath` (``.``/``..`` components and repeated separators removed;
    symbolic links are never resolved). On POSIX, a leading ``//`` (which ``normpath``
    keeps, as POSIX leaves its meaning to the implementation) is also reduced to ``/``:
    Linux and macOS treat it as the root. Windows paths, UNC shares included, are only
    ``normpath``-ed.
    """
    norm = os.path.normpath(os.fspath(path))
    if os.name != "nt" and norm.startswith("//"):
        norm = "/" + norm.lstrip("/")
    return norm


def relative_to_source_root(path: PathLike, source_root: PathLike) -> str:
    """Return *path* relative to *source_root*, lexically (spec §1.2, §4.1).

    Both are normalized with :func:`lexical_path_key` (never ``realpath``). The result
    is non-empty and has no ``..`` component.

    Raises
    ------
    ValueError
        If *path* does not lie strictly below *source_root* (outside it, equal to it, or
        on another drive). The message names the file.
    """
    norm_path = lexical_path_key(path)
    norm_root = lexical_path_key(source_root)
    try:
        rel: str | None = os.path.relpath(norm_path, norm_root)
    except ValueError:  # different drives on Windows
        rel = None
    if (
        rel is None
        or rel == os.curdir
        or rel == os.pardir
        or rel.startswith(os.pardir + os.sep)
        or os.path.isabs(rel)
        or os.path.isabs(norm_path) != os.path.isabs(norm_root)
    ):
        raise ValueError(
            f"file {norm_path!r} is not inside source_root {norm_root!r}. Every discovered "
            "file must lie below its dataset's source_root; fix file_pattern or source_root."
        )
    return rel


def check_derivatives_subdir(value: Any, *, field: str = "derivatives_subdir") -> str | None:
    """Validate a ``derivatives_subdir`` value and return it normalized (``None`` when unset).

    It must be a non-empty relative path (``/``-separated components) with no ``..``,
    no backslash, no absolute or drive-qualified form, and its first component must not
    be ``code`` or ``.neurodags`` (reserved under a ``derivatives_path``). It is used
    literally (no ``~``/``$VAR`` expansion). ``.`` components are dropped.

    Raises
    ------
    ValueError
        With a message naming *field*.
    """
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string, got {value!r}.")
    if "\\" in value:
        raise ValueError(f"{field} {value!r} must use '/' as separator (no backslashes).")
    if posixpath.isabs(value) or ntpath.splitdrive(value)[0]:
        raise ValueError(f"{field} {value!r} must be a relative path below derivatives_path.")
    parts = [p for p in value.split("/") if p not in ("", ".")]
    if not parts:
        raise ValueError(f"{field} {value!r} must name a subfolder of derivatives_path.")
    if ".." in parts:
        raise ValueError(f"{field} {value!r} must not contain '..'.")
    if parts[0] in ("code", ".neurodags"):
        raise ValueError(
            f"{field} {value!r} is reserved: '{parts[0]}/' under derivatives_path is used by "
            "neurodags itself."
        )
    return "/".join(parts)


def effective_derivatives_base(
    derivatives_path: str | None, derivatives_subdir: str | None
) -> str | None:
    """Return ``derivatives_path / derivatives_subdir`` (spec §4.1, §4.4).

    *derivatives_path* is an already resolved path, or ``None`` (outputs go next to the
    sources and the subdir does not apply). *derivatives_subdir* is validated with
    :func:`check_derivatives_subdir`. The ``code/`` snapshot is not affected: it stays at
    ``<derivatives_path>/code/``.
    """
    if not derivatives_path:
        return None
    subdir = check_derivatives_subdir(derivatives_subdir)
    if not subdir:
        return derivatives_path
    return os.path.join(derivatives_path, *subdir.split("/"))


def _format_resolution_error(exc: BaseException, field: str, value: Any, mount_point: Any) -> str:
    if isinstance(exc, KeyError):
        available = ", ".join(map(str, value)) if isinstance(value, dict) else "?"
        return f"{field}: no entry for mount point {mount_point!r} (available: {available})"
    return str(exc)


def inspect_dataset_paths(
    datasets: Mapping[str, Any],
    mount_point: str | None = None,
    *,
    strict: bool = False,
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Resolve and check the path fields of every dataset; collect the problems.

    Checks (spec §1.2, §4.4):

    - ``source_root`` is set for every dataset (active or skipped); when it is missing,
      the problem includes a suggestion (:func:`suggest_source_root`) when one exists;
    - for active datasets, every path field resolves for the active mount point and
      ``source_root`` is an existing directory;
    - ``derivatives_subdir`` is valid (:func:`check_derivatives_subdir`);
    - two active datasets resolving to the same ``derivatives_path`` have different
      ``derivatives_subdir`` values (unset counts as one value).

    Parameters
    ----------
    datasets : mapping of str to DatasetConfig
        Dataset configurations.
    mount_point : str, optional
        The active mount point.
    strict : bool, default False
        Re-raise path-resolution errors (unset variables, missing mount entries,
        relative paths without a base directory) instead of collecting them.

    Returns
    -------
    resolved : dict
        ``{dataset_name: {"source_root", "file_pattern", "exclude_pattern",
        "derivatives_path", "derivatives_subdir", "derivatives_base"}}`` for active
        datasets (``None`` for unset or unresolvable values).
    problems : list of str
        One message per problem, empty when the configuration is valid.
    """
    resolved: dict[str, dict[str, Any]] = {}
    problems: list[str] = []
    for name, cfg in datasets.items():
        skipped = bool(getattr(cfg, "skip", False))
        entry: dict[str, Any] = {}
        for attr in DATASET_PATH_FIELDS:
            entry[attr] = None
            if skipped and attr != "file_pattern":
                continue
            try:
                entry[attr] = resolve_dataset_field(cfg, attr, mount_point)
            except (ValueError, KeyError) as exc:
                if skipped:  # only resolved for the source_root suggestion
                    continue
                if strict:
                    raise
                problems.append(
                    _format_resolution_error(
                        exc, f"datasets.{name}.{attr}", getattr(cfg, attr, None), mount_point
                    )
                )

        if not getattr(cfg, "source_root", None):
            msg = (
                f"datasets.{name}.source_root is required (since 0.4.0): set it to the "
                "root folder of the dataset's source files"
            )
            suggestion = suggest_source_root(entry["file_pattern"] or "")
            if suggestion is not None:
                where = f" for mount '{mount_point}'" if mount_point is not None else ""
                msg += f"; suggested{where}: {suggestion}"
            problems.append(msg)
        if skipped:
            continue

        root = entry["source_root"]
        if root is not None and not os.path.isdir(root):
            problems.append(f"datasets.{name}.source_root {root!r} is not an existing directory")

        try:
            entry["derivatives_subdir"] = check_derivatives_subdir(
                getattr(cfg, "derivatives_subdir", None),
                field=f"datasets.{name}.derivatives_subdir",
            )
        except ValueError as exc:
            problems.append(str(exc))
            entry["derivatives_subdir"] = None
        entry["derivatives_base"] = effective_derivatives_base(
            entry["derivatives_path"], entry["derivatives_subdir"]
        )
        resolved[name] = entry

    problems.extend(_shared_derivatives_path_problems(resolved, datasets))
    return resolved, problems


def _shared_derivatives_path_problems(
    resolved: Mapping[str, Mapping[str, Any]], datasets: Mapping[str, Any]
) -> list[str]:
    """Spec §4.4: datasets sharing a derivatives_path need distinct derivatives_subdir."""
    by_path: dict[str, list[str]] = {}
    for name, entry in resolved.items():
        if entry.get("derivatives_path"):
            # Same lexical identity as discovery (a POSIX leading // is /).
            key = lexical_path_key(entry["derivatives_path"])
            by_path.setdefault(key, []).append(name)
    problems: list[str] = []
    for path, names in by_path.items():
        if len(names) < 2:
            continue
        by_subdir: dict[str | None, list[str]] = {}
        for name in names:
            raw = getattr(datasets[name], "derivatives_subdir", None)
            try:
                subdir = check_derivatives_subdir(raw)
            except ValueError:
                subdir = raw  # already reported as invalid
            by_subdir.setdefault(subdir, []).append(name)
        for subdir, same in by_subdir.items():
            if len(same) > 1:
                shown = "unset" if subdir is None else repr(subdir)
                problems.append(
                    f"datasets {', '.join(same)} share derivatives_path {path!r} with the same "
                    f"derivatives_subdir ({shown}); give each a different derivatives_subdir "
                    "so their outputs cannot collide"
                )
    return problems


def resolve_dataset_paths(
    datasets: Mapping[str, Any],
    mount_point: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Resolve and validate the path fields of every active dataset.

    Preflight run before anything is written to disk: path-resolution errors (an
    unset ``$VAR``, a mount map without the active mount, a relative path without a
    base directory) raise as soon as they are found, then the configuration checks of
    :func:`inspect_dataset_paths` (required ``source_root``, existing source root,
    ``derivatives_subdir``, shared ``derivatives_path``) raise together.

    Parameters
    ----------
    datasets : mapping of str to DatasetConfig
        Dataset configurations (as returned by
        :func:`neurodags.datasets.get_datasets_and_mount_point_from_pipeline_configuration`).
    mount_point : str, optional
        The active mount point.

    Returns
    -------
    dict
        Resolved fields per active dataset, see :func:`inspect_dataset_paths`.
        Skipped datasets are omitted.

    Raises
    ------
    ValueError
        If any active path field references an unset/empty environment variable, or
        is relative without a base directory.
    KeyError
        If a mount-point map lacks the active mount point.
    DatasetConfigError
        If any configuration check fails (all problems are listed).
    """
    resolved, problems = inspect_dataset_paths(datasets, mount_point, strict=True)
    if problems:
        raise DatasetConfigError(problems)
    return resolved


def _resolve_eval_strings(value: Any) -> Any:
    """Recursively interpret ``eval%`` prefixed expressions within nested structures."""

    if isinstance(value, dict):
        return {key: _resolve_eval_strings(sub_value) for key, sub_value in value.items()}
    if isinstance(value, list):
        return [_resolve_eval_strings(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_resolve_eval_strings(item) for item in value)
    if isinstance(value, str) and value.startswith("eval%"):
        expression = value.removeprefix("eval%")
        namespace = {"np": np, "math": math}
        return eval(expression, namespace)
    return value


def replace_bids_suffix(path: PathLike, new_suffix: str, new_ext: str) -> Path:
    """
    Replace the suffix and extension of a BIDS-like filename.

    Parameters
    ----------
    path : PathLike
        Original file path (``str`` or ``os.PathLike``).
    new_suffix : str
        New suffix (without leading underscore).
    new_ext : str
        New extension. Can handle multi-part extensions (e.g. ``".nii.gz"``).
        Must include the leading dot.

    Returns
    -------
    Path
        Path object with replaced suffix and extension.

    .. deprecated:: 0.4.0
        Removed in 0.5. neurodags no longer derives output names from BIDS suffixes.

    Notes
    -----
    - Handles multi-part extensions (e.g., ``.nii.gz``).
    - Splits on the last underscore to isolate suffix **only if the file already
      has an extension**. If no extension is present, keeps the full name.
    """
    warnings.warn(
        "neurodags.utils.replace_bids_suffix is deprecated and will be removed in neurodags "
        "0.5; derivative outputs are named by the 0.4.0 layout (see the naming guide).",
        DeprecationWarning,
        stacklevel=2,
    )
    path = Path(path)
    log.debug("Received path for suffix replacement", path=str(path))

    stem = path.name
    exts = "".join(path.suffixes)  # ".nii.gz" or ""
    base = stem[: -len(exts)] if exts else stem
    log.debug("Split stem into base and exts", base=base, exts=exts)

    prefix = base
    if exts and "_" in base:
        prefix, _ = base.rsplit("_", 1)
        log.debug("Detected existing suffix, isolating prefix", prefix=prefix)
    elif not exts and "_" in base:
        log.debug("No extension present, keeping full base as prefix", prefix=prefix)
    else:
        log.debug("No underscore found, using full base", prefix=prefix)

    new_name = f"{prefix}_{new_suffix}{new_ext}"
    log.debug("Constructed new filename", new_name=new_name)

    return path.with_name(new_name)


def find_unique_root(
    filepaths: Sequence[PathLike],
    *,
    style: str = "auto",  # {'auto','posix','windows'}
    strict: bool = True,  # if True, reject ambiguous/mixed inputs
    mode: str = "minimal",  # {'minimal','maximal'}
) -> str:
    """
    Find the shallowest or deepest root such that relative paths from it are unique.

    Parameters
    ----------
    filepaths : sequence of path-like
        Absolute or relative paths (all expected to use the same style).
    style : {'auto', 'posix', 'windows'}, optional
        Path style rules to apply. 'auto' infers from inputs:
        - any path with a drive letter or backslash ⇒ windows
        - otherwise ⇒ posix
        Mixed styles raise if strict=True.
    strict : bool, optional
        If True, raise on mixed/ambiguous separators for the chosen style.
        If False, coerce separators:
        - Windows: convert ``/`` → ``\\``
        - POSIX: convert ``\\`` → ``/``
    mode : {'minimal', 'maximal'}, optional
        - 'minimal': return the shallowest root that ensures unique relpaths.
        - 'maximal': return the deepest such root.

    Returns
    -------
    str
        Normalized root path as a string.

    Raises
    ------
    ValueError
        If filepaths is empty, styles are mixed with strict=True,
        or mode is invalid.
    """
    if not filepaths:
        log.debug("find_unique_root: empty input")
        raise ValueError("filepaths must be non-empty")

    raw = [os.fspath(p) for p in filepaths]
    log.debug("find_unique_root: called", n_paths=len(raw), style=style, strict=strict, mode=mode)

    def looks_windows(s: str) -> bool:
        # Drive letter or backslash suggests Windows intent
        return ("\\" in s) or (len(s) >= 2 and s[1] == ":")

    # Decide style
    if style == "auto":
        flags = [looks_windows(s) for s in raw]
        if any(flags) and not all(flags):
            if strict:
                log.debug("find_unique_root: mixed styles rejected (strict=True)")
                raise ValueError("Mixed POSIX/Windows styles are not supported (strict=True).")
            # Lenient: pick windows if any looks windows
            use_windows = True
            log.debug("find_unique_root: mixed styles coerced to windows (lenient)")
        else:
            use_windows = all(flags)
            log.debug(
                "find_unique_root: inferred style",
                inferred="windows" if use_windows else "posix",
            )
    elif style == "windows":
        use_windows = True
        log.debug("find_unique_root: explicit style", chosen="windows")
    elif style == "posix":
        use_windows = False
        log.debug("find_unique_root: explicit style", chosen="posix")
    else:
        log.debug("find_unique_root: invalid style", style=style)
        raise ValueError("style must be 'auto', 'posix', or 'windows'.")

    # Pick modules/classes per style
    pmod = ntpath if use_windows else posixpath
    PurePath = PureWindowsPath if use_windows else PurePosixPath

    # Separator coercion if not strict
    paths = []
    for s in raw:
        if strict:
            if not use_windows and "\\" in s:
                log.debug("find_unique_root: backslash in POSIX path with strict=True", path=s)
                raise ValueError(
                    f"Backslash found in POSIX path '{s}' (strict=True). "
                    "Use strict=False to coerce '\\' → '/'."
                )
            paths.append(s)
        else:
            if use_windows:
                paths.append(s.replace("/", "\\"))  # normalize separators
            else:
                paths.append(s.replace("\\", "/"))  # treat backslash as separator
    if not strict:
        log.debug("find_unique_root: coerced separators", use_windows=use_windows)

    # Normalize with the chosen flavor
    paths = [pmod.normpath(s) for s in paths]
    log.debug(
        "find_unique_root: normalized paths sample",
        sample=paths[:3] if len(paths) > 3 else paths,
    )

    # Upper bound common prefix
    try:
        common_prefix = pmod.commonpath(paths)
    except ValueError:
        # e.g., different drives on Windows → no common path
        common_prefix = ""
        log.debug("find_unique_root: no common path (e.g., different drives)")

    split = [PurePath(s).parts for s in paths]
    prefix_len = len(PurePath(common_prefix).parts)
    log.debug("find_unique_root: prefix info", common_prefix=common_prefix, prefix_len=prefix_len)

    # Try shallowest→deepest; collect all candidates. A candidate must be a
    # strict ancestor of every path (non-empty relative tail): otherwise a
    # single-file input would yield the file itself as root and a relpath of
    # "." (#34). With distinct multi-file inputs the common prefix is always a
    # strict ancestor, so their roots are unaffected.
    candidates = []
    for i in range(1, prefix_len + 1):
        rels = []
        for parts in split:
            tail = parts[i:]
            rels.append(pmod.join(*tail) if tail else "")
        if "" in rels:
            log.debug("find_unique_root: candidate is not a strict ancestor", depth=i)
            continue
        unique = len(rels) == len(set(rels))
        log.debug("find_unique_root: candidate check", depth=i, unique=unique)
        if unique:
            head = split[0][:i]
            chosen = pmod.normpath(pmod.join(*head))
            candidates.append((i, chosen))

    if candidates:
        if mode == "minimal":
            chosen = candidates[0][1]  # shallowest
            log.debug("find_unique_root: selected minimal root", root=chosen)
            return chosen
        elif mode == "maximal":
            chosen = candidates[-1][1]  # deepest
            log.debug("find_unique_root: selected maximal root", root=chosen)
            return chosen
        else:
            raise ValueError("mode must be 'minimal' or 'maximal'")

    # Fallback — full common path (may be empty)
    fallback = pmod.normpath(common_prefix) if common_prefix else common_prefix
    log.debug("find_unique_root: fallback to common prefix", root=fallback or "(none)")
    return fallback
