import math
import ntpath
import os
import posixpath
import re
from collections.abc import Iterable, Mapping, Sequence
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


def resolve_dataset_paths(
    datasets: Mapping[str, Any],
    mount_point: str | None = None,
) -> dict[str, dict[str, str | None]]:
    """Resolve and validate the path fields of every active dataset.

    Expands ``file_pattern``, ``derivatives_path`` and ``exclude_pattern`` for the
    active mount point of each non-skipped dataset (see :func:`expand_path`).
    Intended as a preflight check before anything is written to disk, so that an
    unresolved ``$VAR`` fails loudly instead of creating a literal ``$VAR``
    directory.

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
        ``{dataset_name: {"file_pattern": ..., "derivatives_path": ...,
        "exclude_pattern": ...}}`` with resolved values (``None`` when unset).
        Skipped datasets are omitted.

    Raises
    ------
    ValueError
        If any active path field references an unset/empty environment variable.
    KeyError
        If a mount-point map lacks the active mount point.
    """
    resolved: dict[str, dict[str, str | None]] = {}
    for name, cfg in datasets.items():
        if getattr(cfg, "skip", False):
            continue
        entry: dict[str, str | None] = {}
        for attr in ("file_pattern", "derivatives_path", "exclude_pattern"):
            value = getattr(cfg, attr, None)
            entry[attr] = (
                get_path(value, mount_point=mount_point, field=f"datasets.{name}.{attr}")
                if value
                else None
            )
        resolved[name] = entry
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


def snake_to_camel(snake_str):
    """Convert a snake_case string to CamelCase.

    Used to turn a derivative name into the suffix of its on-disk files
    (``<source>@<CamelName>.<ext>``). It keeps the derivative name a single token
    without ``_``, which many file-naming conventions (including BIDS) use as a
    separator. Names without ``_`` are returned unchanged (``-`` is left as is);
    otherwise each ``_``-separated part is title-cased and joined
    (``fit_w_l06`` -> ``FitWL06``). Changing this
    rule would rename existing cached outputs, so it is kept stable; distinct
    derivative names that map to the same token are rejected by
    :func:`check_derivative_names` (#42).

    Parameters
    ----------
    snake_str : str
        The input string in snake_case format.

    Returns
    -------
    str
        The converted string in CamelCase format.
    """
    if "_" not in snake_str:
        return snake_str  # no conversion needed
    components = snake_str.split("_")
    return "".join(x.title() for x in components)


def derivative_file_names(derivative_names: Iterable[Any]) -> dict[str, str]:
    """Map each derivative name to the name its on-disk files use (``@<name>``).

    Parameters
    ----------
    derivative_names : iterable
        Derivative names as written in ``DerivativeDefinitions``.

    Returns
    -------
    dict[str, str]
        ``{yaml_name: on_disk_name}`` in input order, via :func:`snake_to_camel`.
    """
    return {str(name): snake_to_camel(str(name)) for name in derivative_names}


# Characters a derivative name must not contain, and why (#42). Each one breaks
# the ``<source>@<name><artifact suffix>`` layout or the existence-based cache:
_FORBIDDEN_NAME_CHARS = {
    ".": "separates the name from the artifact suffix ('derivative: Name.ext' splits at "
    "the first '.', and the cache glob '@Name.*' would match another derivative's files)",
    "@": "separates the source name from the derivative name (one '@' per file name)",
    "/": "is a path separator",
    "\\": "is a path separator",
    "*": "is a glob wildcard used by the cache lookup",
    "?": "is a glob wildcard used by the cache lookup",
    "[": "is a glob character used by the cache lookup",
    "]": "is a glob character used by the cache lookup",
}
_RESERVED_DERIVATIVE_NAMES = {"SourceFile": "'derivative: SourceFile' means the raw input file"}


def check_derivative_names(derivative_names: Iterable[Any]) -> dict[str, str]:
    """Validate derivative names against the on-disk naming rules.

    Outputs are cached by existence of ``<source>@<on-disk name>.*``, so a name
    that breaks that layout, or two names sharing an on-disk name, silently
    reuse the wrong files (#42). Rejected:

    - an empty name, the reserved name ``SourceFile``, and names containing
      ``.``, ``@``, ``/``, ``\\``, ``*``, ``?``, ``[`` or ``]``;
    - two names whose on-disk names (:func:`snake_to_camel`) are equal, or equal
      up to case (the same file on case-insensitive filesystems), e.g.
      ``band_power`` and ``BandPower`` (both ``@BandPower``).

    Parameters
    ----------
    derivative_names : iterable
        Derivative names as written in ``DerivativeDefinitions``.

    Returns
    -------
    dict[str, str]
        ``{yaml_name: on_disk_name}`` (see :func:`derivative_file_names`).

    Raises
    ------
    ValueError
        Listing every offending name, with both YAML names and the shared
        on-disk name for collisions.
    """
    mapping = derivative_file_names(derivative_names)
    problems: list[str] = []
    for name in mapping:
        if not name:
            problems.append("an empty derivative name is not allowed")
        elif name in _RESERVED_DERIVATIVE_NAMES:
            problems.append(f"{name!r} is reserved: {_RESERVED_DERIVATIVE_NAMES[name]}")
        for char, reason in _FORBIDDEN_NAME_CHARS.items():
            if char in name:
                problems.append(f"{name!r} contains {char!r}, which {reason}")

    by_file_name: dict[str, list[str]] = {}
    for name, file_name in mapping.items():
        by_file_name.setdefault(file_name.casefold(), []).append(name)
    for names in by_file_name.values():
        if len(names) > 1:
            on_disk = sorted({mapping[n] for n in names})
            shown = " / ".join(f"'@{f}'" for f in on_disk)
            note = " (same file on case-insensitive filesystems)" if len(on_disk) > 1 else ""
            problems.append(
                " and ".join(repr(n) for n in names)
                + f" collide on disk: files named {shown}{note}"
            )

    if problems:
        raise ValueError(
            "DerivativeDefinitions: invalid derivative names: "
            + "; ".join(problems)
            + ". Derivative outputs are named '<source>@<OnDiskName><artifact suffix>', where "
            "the on-disk name is the YAML name converted with snake_to_camel "
            "('_'-separated parts title-cased and joined); colliding names would read and "
            "overwrite each other's cached outputs. Rename the derivative(s)."
        )
    return mapping


def analyze_bids_filename(path: PathLike) -> dict[str, bool]:
    """
    Analyze whether a filename follows BIDS-like conventions.

    Parameters
    ----------
    path : PathLike
        The file path to check (``str`` or ``os.PathLike``).

    Returns
    -------
    dict : Dictionary with two boolean entries:
    - ``is_key_value_pairs`` : bool
        True if all underscore-separated parts before the suffix are
        BIDS-like key-value pairs (`key-value`).
    - ``has_suffix`` : bool
        True if there is a final part (before the extension) that is not
        a key-value pair (e.g., "_bold", "_eeg", "_meg").
    """
    path = Path(path)
    filename = path.stem  # remove extension
    parts = filename.split("_")

    def is_key_value(part: str) -> bool:
        if "-" not in part:
            return False
        key, value = part.split("-", 1)
        return key.isalnum() and all(c.isalnum() or c == "-" for c in value)

    # All but possibly the last part must be key-value pairs
    key_value_parts = [is_key_value(p) for p in parts]

    # If last part is not a key-value → treat as suffix
    has_suffix = not key_value_parts[-1]

    # For is_key_value_pairs we require all parts before suffix to be key-value
    is_key_value_pairs = all(key_value_parts[:-1]) if has_suffix else all(key_value_parts)

    return {
        "is_key_value_pairs": is_key_value_pairs,
        "has_suffix": has_suffix,
    }


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

    Notes
    -----
    - Handles multi-part extensions (e.g., ``.nii.gz``).
    - Splits on the last underscore to isolate suffix **only if the file already
      has an extension**. If no extension is present, keeps the full name.
    """
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
