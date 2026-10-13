"""Names, suffixes, references, output basenames and dataframe columns (naming version 1).

This module is the single place that defines how neurodags spells identities: derivative
names, artifact suffixes, references between derivatives, output file basenames and wide
dataframe columns. Every function is pure (no I/O, no logging, standard library only), so
the same rules apply in the CLI, the TUI, the Python API and the migration tool.

Dataframe assembly uses the column functions (:func:`format_column`, :func:`error_column`);
later 0.4.0 changes wire in the rest.

Summary of the grammar
----------------------
- Derivative name: ``^[A-Za-z](?:[A-Za-z0-9_-]*[A-Za-z0-9])?$``, used verbatim everywhere.
  ``SourceFile`` is reserved (any case).
- Artifact suffix: ``^(?:\\.[A-Za-z0-9][A-Za-z0-9_-]*)+$``; the last segment is never
  ``done``, ``error`` or ``skip`` (any case), which mark completion markers.
- Output basename: ``<source basename verbatim>@<name><suffix>``; parsed at the last ``@``
  (names and suffixes never contain ``@``), so chained names such as
  ``sub-1.fif@Clean.fif@Power.nc`` nest.
- Reference: ``SourceFile`` | ``name`` | ``name`` + ``suffix``.
- Wide dataframe column: ``name suffix ["@" dim-value ("_" dim-value)*]`` with dims and
  values percent-encoded, or ``name.error`` for the error column.

Optional key-value convention for names: ``_``-separated parts, each a ``key-value`` pair or
a bare word, with at least one pair and at most one bare word (the base) in any position (for
example ``Fit_alpha-0p5_w-6``, ``alpha-0p5_w-6_Fit`` or ``alpha-0p5_w-6``), with numbers written using ``p`` for the decimal point and ``m`` for
minus (see :func:`encode_number`). Names that do not follow it are equally valid.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from decimal import Decimal
from itertools import pairwise
from numbers import Integral
from typing import Any, NamedTuple

__all__ = [
    "ENTITY_PAIR_RE",
    "ENTITY_WORD_RE",
    "FIELD_RE",
    "MARKER_SEGMENTS",
    "NAME_LENGTH_WARNING",
    "NAME_RE",
    "RESERVED_NAME",
    "SUFFIX_RE",
    "ColumnCollisionError",
    "NamingError",
    "ParsedColumn",
    "ParsedOutput",
    "check_derivative_names",
    "decode_dim",
    "decode_number",
    "decode_value",
    "encode_dim",
    "encode_number",
    "encode_value",
    "entity_pairs",
    "error_column",
    "find_column_collisions",
    "format_column",
    "marker_basename",
    "output_basename",
    "output_prefix_basename",
    "parse_column",
    "parse_entities",
    "parse_output_basename",
    "parse_reference",
    "repeated_entity_keys",
    "validate_artifact_suffix",
    "validate_artifact_suffixes",
    "validate_derivative_name",
]


class NamingError(ValueError):
    """Raised when a name, suffix, reference, basename or column breaks the naming rules.

    The message is meant to be shown to users as is: it names the offending value and says
    how to fix it. :func:`check_derivative_names` and :func:`validate_artifact_suffixes`
    collect every problem into a single error, one problem per line.
    """


class ColumnCollisionError(NamingError):
    """Two origins would produce the same dataframe column (spec section 7.1).

    Raised while assembling a dataframe when two coordinates of one artifact, two artifacts,
    or two derivatives map to the same column (possible only with non-unique coordinate
    indexes or mixed types such as ``1`` and ``"1"``). The message names the column and
    every origin. It is a structural problem of the pipeline's outputs, not a per-file
    failure, so it is raised even when errors are otherwise recorded in ``name.error``
    columns.
    """


# ---------------------------------------------------------------------------
# Regular expressions and constants
# ---------------------------------------------------------------------------

#: Derivative-name grammar (spec section 2.1). Use with ``fullmatch``.
NAME_RE = re.compile(r"[A-Za-z](?:[A-Za-z0-9_-]*[A-Za-z0-9])?")

#: Artifact-suffix grammar (spec section 3). Use with ``fullmatch``.
SUFFIX_RE = re.compile(r"(?:\.[A-Za-z0-9][A-Za-z0-9_-]*)+")

#: Node-field reference ``id.N`` or ``id.N.field`` (spec section 5). Group 1 is the node
#: index, group 2 the optional field. Use with ``fullmatch``.
FIELD_RE = re.compile(r"id\.([0-9]+)(?:\.([A-Za-z_][A-Za-z0-9_]*))?")

#: One ``key-value`` part of the key-value convention (spec section 2.3). Group 1 is the
#: key, group 2 the value. Use with ``fullmatch`` on a single ``_``-separated part.
ENTITY_PAIR_RE = re.compile(r"([A-Za-z][A-Za-z0-9]*)-([A-Za-z0-9]+)")

#: One bare word (the base) of the key-value convention. Use with ``fullmatch`` on a single
#: ``_``-separated part.
ENTITY_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*")

#: Reserved reference to the input file itself; not usable as a derivative name (any case).
RESERVED_NAME = "SourceFile"

#: Last suffix segments that denote completion markers; reserved for artifacts (any case).
MARKER_SEGMENTS = frozenset({"done", "error", "skip"})

#: Names longer than this produce a warning in :func:`check_derivative_names`.
NAME_LENGTH_WARNING = 64

_SPLIT_TAIL_RE = re.compile(r"(.+)-([0-9]+)")

_DIM_SAFE = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789")
_VALUE_SAFE = _DIM_SAFE | frozenset(".+~-")
_PCT_RE = re.compile(r"%([0-9A-F]{2})")

_NUMBER_RE = re.compile(r"(m)?(0|[1-9][0-9]*)(?:p([0-9]*[1-9]))?")
_DECIMAL_STR_RE = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?")

_CHAR_HINTS = {
    ".": "'.' starts an artifact suffix (as in Name.nc) and cannot appear in a name",
    "@": "'@' separates the source file from the derivative name in output files",
    "&": "'&' is not allowed in names",
    ",": "commas are not allowed (write decimals with 'p', e.g. 0p5)",
    "/": "path separators are not allowed",
    "\\": "path separators are not allowed",
}


def _describe_char(ch: str) -> str:
    """Return a readable description of a single character for error messages."""
    if ch.isspace():
        return f"whitespace {ch!r} (U+{ord(ch):04X})"
    if not ch.isascii():
        return f"non-ASCII character {ch!r} (U+{ord(ch):04X})"
    return f"character {ch!r}"


# ---------------------------------------------------------------------------
# Derivative names (spec section 2)
# ---------------------------------------------------------------------------


def _not_a_string_message(name: Any, what: str) -> str:
    kind = "null" if name is None else type(name).__name__
    return (
        f"{what} {name!r} is not a string (YAML loaded it as {kind}). Unquoted YAML keys such "
        "as Yes, No, On, Off, true, 1_000, 0x1F or ~ become booleans, numbers or null; quote "
        f'the key in the YAML (e.g. "{name}":) so it stays a string.'
    )


def _name_problem(name: Any) -> str | None:
    """Return a one-line description of why *name* is invalid, or ``None`` when valid."""
    if not isinstance(name, str):
        return _not_a_string_message(name, "Derivative name")
    if name == "":
        return "Derivative name is empty."
    if name.lower() == RESERVED_NAME.lower():
        return (
            f"Derivative name {name!r} is reserved: {RESERVED_NAME!r} (any case) refers to the "
            "input file itself. Choose another name."
        )
    if NAME_RE.fullmatch(name):
        return None
    for ch in name:
        if not (ch.isascii() and (ch.isalnum() or ch in "_-")):
            hint = _CHAR_HINTS.get(ch, "names use only ASCII letters, digits, '_' and '-'")
            return f"Derivative name {name!r} contains {_describe_char(ch)}: {hint}."
    if not name[0].isalpha():
        return f"Derivative name {name!r} must start with an ASCII letter (found {name[0]!r})."
    return (
        f"Derivative name {name!r} must end with a letter or digit (found {name[-1]!r}); "
        "trailing '-' or '_' is not allowed."
    )


def validate_derivative_name(name: Any) -> None:
    """Check one derivative name against the naming grammar.

    Parameters
    ----------
    name : Any
        The candidate name, typically a ``DerivativeDefinitions`` key as loaded from YAML.

    Raises
    ------
    NamingError
        If *name* is not a string (with advice to quote the YAML key), does not match
        :data:`NAME_RE` (naming the offending character), or is ``SourceFile`` in any case.

    Examples
    --------
    >>> validate_derivative_name("Fit_alpha-0p5_w-6")
    >>> validate_derivative_name("Bad.Name")
    Traceback (most recent call last):
    ...
    neurodags.naming.NamingError: Derivative name 'Bad.Name' contains character '.': ...
    """
    problem = _name_problem(name)
    if problem is not None:
        raise NamingError(problem)


def check_derivative_names(names: Iterable[Any]) -> list[str]:
    """Validate a set of derivative names together and return warnings.

    All problems are collected and raised as one :class:`NamingError` (one per line):

    - grammar, non-string and reserved-name problems (see :func:`validate_derivative_name`);
    - names that differ only in case (``Clean`` and ``clean``), which collide on
      case-insensitive filesystems;
    - the split-file rule: ``X`` and ``X-<digits>`` both defined (compared ignoring case),
      because MNE writes split recordings as ``X-1.fif`` and discovery drops such files.

    Exact repeats of the same spelling are counted once (a mapping cannot hold them; a list
    that repeats an entry still names one derivative).

    Parameters
    ----------
    names : Iterable[Any]
        All derivative names of one configuration.

    Returns
    -------
    list[str]
        Warnings: names longer than :data:`NAME_LENGTH_WARNING` characters and names that
        follow the key-value convention but repeat a key (``Fit_w-6_w-7``).

    Raises
    ------
    NamingError
        Listing every problem found.
    """
    problems: list[str] = []
    warnings: list[str] = []
    valid: list[str] = []
    seen_exact: set[Any] = set()
    for name in names:
        try:
            key: tuple[Any, ...] = (type(name), hash(name), name)
        except TypeError:
            key = (type(name), id(name), None)
        if key in seen_exact:
            continue
        seen_exact.add(key)
        problem = _name_problem(name)
        if problem is not None:
            problems.append(problem)
            continue
        valid.append(name)
        if len(name) > NAME_LENGTH_WARNING:
            warnings.append(
                f"Derivative name {name!r} is {len(name)} characters long (more than "
                f"{NAME_LENGTH_WARNING}); long names make long file names and columns."
            )
        repeated = repeated_entity_keys(name)
        if repeated:
            warnings.append(
                f"Derivative name {name!r} follows the key-value convention but repeats "
                f"key(s) {', '.join(repr(k) for k in repeated)}."
            )

    by_folded: dict[str, list[str]] = {}
    for name in valid:
        by_folded.setdefault(name.lower(), []).append(name)
    for group in by_folded.values():
        if len(group) > 1:
            problems.append(
                "Derivative names differ only in case: "
                + ", ".join(repr(n) for n in group)
                + ". They would collide on case-insensitive filesystems; rename one."
            )

    for name in valid:
        match = _SPLIT_TAIL_RE.fullmatch(name)
        if match is None:
            continue
        for base in by_folded.get(match.group(1).lower(), []):
            problems.append(
                f"Derivative names {base!r} and {name!r} are both defined: MNE writes split "
                f"recordings as '{base}-1.fif', '{base}-2.fif', ..., so {name!r} would be "
                "mistaken for a split part. Rename one of them."
            )

    if problems:
        raise NamingError("\n".join(problems))
    return warnings


# ---------------------------------------------------------------------------
# Key-value convention and numbers (spec section 2.3)
# ---------------------------------------------------------------------------


def entity_pairs(name: str) -> tuple[str | None, list[tuple[str, str]]] | None:
    """Split a name that follows the key-value convention into base and ordered pairs.

    The name is split on ``_``. Each part must be either a pair ``key-value`` (exactly one
    ``-``; key ``[A-Za-z][A-Za-z0-9]*``, value ``[A-Za-z0-9]+``, see :data:`ENTITY_PAIR_RE`)
    or a bare word ``[A-Za-z][A-Za-z0-9]*`` (:data:`ENTITY_WORD_RE`). The name follows the
    convention when it has **at least one pair** and **at most one bare word**, in any
    position; that bare word is the base.

    Parameters
    ----------
    name : str
        A derivative name.

    Returns
    -------
    tuple[str or None, list[tuple[str, str]]] or None
        ``(base, [(key, value), ...])`` with pairs in written order (repeated keys kept) and
        ``base`` ``None`` when there is no bare word; ``None`` when *name* does not follow
        the convention. A non-``None`` result always has at least one pair.

    Examples
    --------
    >>> entity_pairs("Fit_alpha-0p5_w-6")
    ('Fit', [('alpha', '0p5'), ('w', '6')])
    >>> entity_pairs("alpha-0p5_w-6_Fit")
    ('Fit', [('alpha', '0p5'), ('w', '6')])
    >>> entity_pairs("feature-BandPower_w-6")
    (None, [('feature', 'BandPower'), ('w', '6')])
    >>> entity_pairs("Ep_HiguchiFD") is None, entity_pairs("Fit") is None
    (True, True)
    """
    if not isinstance(name, str):
        return None
    base: str | None = None
    pairs: list[tuple[str, str]] = []
    for part in name.split("_"):
        pair = ENTITY_PAIR_RE.fullmatch(part)
        if pair is not None:
            pairs.append((pair.group(1), pair.group(2)))
        elif ENTITY_WORD_RE.fullmatch(part) and base is None:
            base = part
        else:
            return None
    if not pairs:
        return None
    return base, pairs


def repeated_entity_keys(name: str) -> list[str]:
    """Return the keys a structured name repeats, in first-seen order.

    Names that do not follow the convention have no keys and return ``[]``.

    Examples
    --------
    >>> repeated_entity_keys("Fit_w-6_w-7")
    ['w']
    """
    parsed = entity_pairs(name)
    if parsed is None:
        return []
    seen: set[str] = set()
    repeated: list[str] = []
    for key, _ in parsed[1]:
        if key in seen and key not in repeated:
            repeated.append(key)
        seen.add(key)
    return repeated


def parse_entities(name: str) -> tuple[str | None, dict[str, str]] | None:
    """Parse the optional key-value convention into a base and a mapping of entities.

    See :func:`entity_pairs` for the rule. Parsing is lenient: a name that does not follow
    the convention is still a valid derivative name, and this function returns ``None`` for
    it. Values stay strings; use :func:`decode_number` to read numbers. Identity never
    depends on this parse (``Fit_w-6_a-1`` and ``Fit_a-1_w-6`` are different derivatives).

    Parameters
    ----------
    name : str
        A derivative name.

    Returns
    -------
    tuple[str or None, dict[str, str]] or None
        ``None`` when the name does not follow the convention. Otherwise ``(base,
        entities)``: ``base`` is the bare word or ``None`` when there is none, and
        ``entities`` is a non-empty dict in written order.

    Raises
    ------
    NamingError
        If the name follows the convention but repeats a key: a mapping cannot represent it
        without dropping a value.

    Examples
    --------
    >>> parse_entities("Fit_alpha-0p5_w-6")
    ('Fit', {'alpha': '0p5', 'w': '6'})
    >>> parse_entities("alpha-0p5_w-6")
    (None, {'alpha': '0p5', 'w': '6'})
    >>> parse_entities("Ep_HiguchiFD") is None
    True
    """
    parsed = entity_pairs(name)
    if parsed is None:
        return None
    base, pairs = parsed
    repeated = repeated_entity_keys(name)
    if repeated:
        raise NamingError(
            f"Derivative name {name!r} repeats key(s) {', '.join(map(repr, repeated))}; "
            "its entities cannot be parsed into one value per key."
        )
    return base, dict(pairs)


def encode_number(x: int | float | Decimal | str) -> str:
    """Encode a number for use in a name value: ``p`` = decimal point, ``m`` = minus.

    The result is canonical: no exponent, no leading zeros in the integer part (``0`` when
    it is zero), no trailing zeros after the point, and zero is always ``0`` (never ``m0``).
    Nothing is rounded.

    Parameters
    ----------
    x : int, float, Decimal or str
        The number. ``int`` (and other :class:`numbers.Integral`, e.g. NumPy integers) and
        :class:`~decimal.Decimal` are used exactly. A ``str`` must be a plain decimal literal
        (``"0.50"``, ``"-1"``, ``"1e-3"``; no whitespace, ``_`` or ``,``) and is used exactly.
        A ``float`` is converted through its shortest ``repr`` (``0.1`` -> ``0p1``, not the
        full binary expansion); pass a ``str`` or ``Decimal`` when that matters.

    Returns
    -------
    str
        The encoded number, matching ``m?(0|[1-9][0-9]*)(p[0-9]*[1-9])?``.

    Raises
    ------
    TypeError
        For ``bool`` and other types.
    NamingError
        For NaN, infinities, or strings that are not decimal literals (including commas).

    Examples
    --------
    >>> encode_number(0.5), encode_number(-1), encode_number("-0.05"), encode_number(1e-3)
    ('0p5', 'm1', 'm0p05', '0p001')
    >>> encode_number(Decimal("1.00"))
    '1'
    """
    if isinstance(x, bool):
        raise TypeError("encode_number does not accept booleans.")
    if isinstance(x, Decimal):
        d = x
    elif isinstance(x, Integral):
        d = Decimal(int(x))
    elif isinstance(x, float):
        d = Decimal(repr(float(x)))
    elif isinstance(x, str):
        if "," in x:
            raise NamingError(
                f"Cannot encode {x!r}: commas are not allowed; use '.' as the decimal point."
            )
        if not _DECIMAL_STR_RE.fullmatch(x):
            raise NamingError(f"Cannot encode {x!r}: not a plain decimal number.")
        d = Decimal(x)
    else:
        raise TypeError(
            f"encode_number expects int, float, Decimal or str, not {type(x).__name__}."
        )
    if not d.is_finite():
        raise NamingError(f"Cannot encode {x!r}: only finite numbers are allowed.")
    text = format(d, "f")
    negative = text.startswith("-")
    text = text.lstrip("+-")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    text = text.lstrip("0")
    if text.startswith(".") or text == "":
        text = "0" + text
    if text == "0":
        return "0"
    return ("m" if negative else "") + text.replace(".", "p")


def decode_number(s: str) -> int | Decimal:
    """Decode a value written by :func:`encode_number`.

    Only canonical encodings are accepted, so ``encode_number(decode_number(s)) == s``.

    Parameters
    ----------
    s : str
        Encoded number such as ``"0p5"``, ``"m1"`` or ``"m0p25"``.

    Returns
    -------
    int or Decimal
        ``int`` when there is no fractional part, otherwise an exact ``Decimal``
        (never a float, so nothing is rounded).

    Raises
    ------
    NamingError
        If *s* is not a canonical encoding (``m0``, ``01``, ``1p50``, ``1p``, ``-1``, ...).

    Examples
    --------
    >>> decode_number("m1"), decode_number("0p5")
    (-1, Decimal('0.5'))
    """
    match = _NUMBER_RE.fullmatch(s) if isinstance(s, str) else None
    if match is None or (match.group(1) and match.group(2) == "0" and not match.group(3)):
        raise NamingError(
            f"{s!r} is not an encoded number: expected an optional 'm' (minus), digits without "
            "leading zeros, and optionally 'p' plus digits without trailing zeros "
            "(e.g. 0p5, m1, m0p25; zero is 0)."
        )
    sign, whole, frac = match.groups()
    if frac is None:
        value = int(whole)
        return -value if sign else value
    return Decimal(f"{'-' if sign else ''}{whole}.{frac}")


# ---------------------------------------------------------------------------
# Artifact suffixes (spec section 3)
# ---------------------------------------------------------------------------


def _suffix_problem(suffix: Any) -> str | None:
    if not isinstance(suffix, str):
        return f"Artifact suffix {suffix!r} is not a string ({type(suffix).__name__})."
    if not SUFFIX_RE.fullmatch(suffix):
        return (
            f"Artifact suffix {suffix!r} is invalid: it must be one or more segments, each a "
            "'.' followed by a letter or digit and then letters, digits, '_' or '-' "
            "(e.g. '.nc', '.report.html', '.fif.gz')."
        )
    last = suffix.rsplit(".", 1)[1]
    if last.lower() in MARKER_SEGMENTS:
        return (
            f"Artifact suffix {suffix!r} ends in {last!r}, which is reserved for completion "
            "markers (.done, .error, .skip, any case)."
        )
    return None


def validate_artifact_suffix(suffix: Any) -> None:
    """Check one artifact suffix (a ``NodeResult.artifacts`` key).

    Raises
    ------
    NamingError
        If *suffix* does not match :data:`SUFFIX_RE` or its last segment is ``done``,
        ``error`` or ``skip`` in any case.
    """
    problem = _suffix_problem(suffix)
    if problem is not None:
        raise NamingError(problem)


def validate_artifact_suffixes(suffixes: Iterable[Any]) -> None:
    """Check all artifact suffixes of one node result together.

    Parameters
    ----------
    suffixes : Iterable[Any]
        The keys of ``NodeResult.artifacts``.

    Raises
    ------
    NamingError
        Listing every invalid suffix (see :func:`validate_artifact_suffix`) and every group
        of suffixes that differ only in case (``.fif`` and ``.FIF``).
    """
    problems: list[str] = []
    by_folded: dict[str, list[str]] = {}
    for suffix in suffixes:
        problem = _suffix_problem(suffix)
        if problem is not None:
            problems.append(problem)
            continue
        group = by_folded.setdefault(suffix.lower(), [])
        if suffix not in group:
            group.append(suffix)
    for group in by_folded.values():
        if len(group) > 1:
            problems.append(
                "Artifact suffixes differ only in case: "
                + ", ".join(repr(s) for s in group)
                + ". They would collide on case-insensitive filesystems."
            )
    if problems:
        raise NamingError("\n".join(problems))


# ---------------------------------------------------------------------------
# Output basenames (spec sections 4.1 and 4.2)
# ---------------------------------------------------------------------------


class ParsedOutput(NamedTuple):
    """An output basename split into its parts (see :func:`parse_output_basename`)."""

    source_part: str
    """Source basename, verbatim (everything before the last ``@``)."""
    name: str
    """Derivative name."""
    suffix: str
    """Everything from the first ``.`` after the name, e.g. ``.nc`` or ``.done``."""
    is_marker: bool
    """Whether the file is a completion marker (last segment done/error/skip)."""
    marker: str | None
    """``"done"``, ``"error"`` or ``"skip"`` (lowercase) for markers, else ``None``."""


def output_prefix_basename(source_basename: str, name: str) -> str:
    """Return the basename prefix shared by all outputs of one (source, derivative) unit.

    The source basename is kept verbatim, including any ``@`` or ``&`` it contains
    (neurodags is agnostic to input naming conventions). The separator is the last ``@``
    of the result, because derivative names and suffixes never contain ``@``.

    Parameters
    ----------
    source_basename : str
        Basename of the source file (no directory part).
    name : str
        Derivative name (validated).

    Returns
    -------
    str
        ``<source>@<name>``; append an artifact suffix or a marker suffix.

    Raises
    ------
    NamingError
        If *name* is invalid, or *source_basename* is empty or contains a path separator.

    Examples
    --------
    >>> output_prefix_basename("sub-01.fif@Clean.fif", "Power")
    'sub-01.fif@Clean.fif@Power'
    """
    if not isinstance(source_basename, str) or source_basename in ("", ".", ".."):
        raise NamingError(f"Invalid source basename {source_basename!r}.")
    if "/" in source_basename or "\\" in source_basename or "\0" in source_basename:
        raise NamingError(
            f"Source basename {source_basename!r} must not contain a path separator or NUL."
        )
    validate_derivative_name(name)
    return source_basename + "@" + name


def output_basename(source_basename: str, name: str, suffix: str) -> str:
    """Return the basename of one artifact file: prefix plus validated artifact suffix.

    See :func:`output_prefix_basename`; raises :class:`NamingError` for an invalid suffix.
    """
    validate_artifact_suffix(suffix)
    return output_prefix_basename(source_basename, name) + suffix


def marker_basename(source_basename: str, name: str, marker: str) -> str:
    """Return the basename of a completion marker (``marker`` is done, error or skip)."""
    if marker not in MARKER_SEGMENTS:
        raise NamingError(f"Unknown marker {marker!r}; expected one of done, error, skip.")
    return output_prefix_basename(source_basename, name) + "." + marker


def parse_output_basename(basename: str) -> ParsedOutput | None:
    """Split an output basename into source part, derivative name and suffix.

    The split is ``source_part, rest = basename.rsplit("@", 1)`` and
    ``name, _, s = rest.partition(".")``. The source part is the immediate source basename,
    verbatim; it may itself contain ``@`` (a chained output parses again the same way). Treat
    it as opaque: the ``.done`` manifest, not the file name, records the dataset-relative
    source path and completion.

    Note that an MNE split part ``src@Clean-1.fif`` parses as name ``Clean-1``; the
    split-file rule of :func:`check_derivative_names` keeps that unambiguous.

    Parameters
    ----------
    basename : str
        A file name (no directory part).

    Returns
    -------
    ParsedOutput or None
        ``None`` when the file is not a neurodags output: no ``@``, empty source part,
        invalid derivative name, or missing/invalid suffix.

    Examples
    --------
    >>> parse_output_basename("sub-01_eeg.vhdr@Fit_w-6.nc")
    ParsedOutput(source_part='sub-01_eeg.vhdr', name='Fit_w-6', suffix='.nc', is_marker=False, marker=None)
    """
    if not isinstance(basename, str) or "@" not in basename:
        return None
    source_part, rest = basename.rsplit("@", 1)
    name, _, s = rest.partition(".")
    if not source_part or not s or _name_problem(name) is not None:
        return None
    suffix = "." + s
    if not SUFFIX_RE.fullmatch(suffix):
        return None
    last = s.rsplit(".", 1)[-1].lower()
    if last in MARKER_SEGMENTS:
        return ParsedOutput(source_part, name, suffix, True, last)
    return ParsedOutput(source_part, name, suffix, False, None)


# ---------------------------------------------------------------------------
# References (spec section 5)
# ---------------------------------------------------------------------------


def parse_reference(ref: Any) -> tuple[str, str | None]:
    """Parse a derivative reference: ``SourceFile``, ``name`` or ``name`` + ``suffix``.

    Parameters
    ----------
    ref : Any
        The reference as written in the YAML.

    Returns
    -------
    tuple[str, str or None]
        ``("SourceFile", None)``, ``(name, None)`` or ``(name, suffix)``. Whether a bare
        ``name`` is allowed (single artifact) is decided by the caller.

    Raises
    ------
    NamingError
        For non-strings, invalid names (including ``SourceFile`` with a suffix or in another
        case) and invalid or reserved suffixes.

    Examples
    --------
    >>> parse_reference("Clean.EO_baseline.fif")
    ('Clean', '.EO_baseline.fif')
    """
    if not isinstance(ref, str):
        raise NamingError(_not_a_string_message(ref, "Reference"))
    if ref == RESERVED_NAME:
        return RESERVED_NAME, None
    name, dot, s = ref.partition(".")
    if name == RESERVED_NAME:
        raise NamingError(
            f"Reference {ref!r} is invalid: {RESERVED_NAME!r} refers to the input file and "
            "takes no artifact suffix."
        )
    problem = _name_problem(name)
    if problem is not None:
        raise NamingError(f"Reference {ref!r} is invalid. {problem}")
    if not dot:
        return name, None
    suffix = "." + s
    problem = _suffix_problem(suffix)
    if problem is not None:
        raise NamingError(f"Reference {ref!r} is invalid. {problem}")
    return name, suffix


# ---------------------------------------------------------------------------
# Dataframe columns (spec section 7.1)
# ---------------------------------------------------------------------------


def _stringify(value: Any) -> str:
    return value if isinstance(value, str) else str(value)


def _pct_encode(text: str, safe: frozenset[str]) -> str:
    out: list[str] = []
    for ch in text:
        if ch in safe:
            out.append(ch)
        else:
            out.extend(f"%{b:02X}" for b in ch.encode("utf-8"))
    return "".join(out)


def _pct_decode(text: str, safe: frozenset[str], what: str) -> str:
    if not isinstance(text, str):
        raise NamingError(f"Encoded {what} {text!r} is not a string.")
    raw = bytearray()
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == "%":
            match = _PCT_RE.match(text, i)
            if match is None:
                raise NamingError(
                    f"Encoded {what} {text!r}: '%' at position {i} must be followed by two "
                    "uppercase hex digits."
                )
            raw.append(int(match.group(1), 16))
            i += 3
        elif ch in safe:
            raw.append(ord(ch))
            i += 1
        else:
            raise NamingError(f"Encoded {what} {text!r}: {_describe_char(ch)} must be escaped.")
    try:
        decoded = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise NamingError(f"Encoded {what} {text!r} is not valid UTF-8.") from exc
    if _pct_encode(decoded, safe) != text:
        raise NamingError(
            f"Encoded {what} {text!r} is not canonical (escapes a character that is written "
            "literally)."
        )
    return decoded


def encode_dim(dim: Any) -> str:
    """Percent-encode a dimension name for a column (keeps only ``[A-Za-z0-9]`` literal).

    Non-string dims are converted with ``str()`` (``None`` -> ``"None"``).

    Examples
    --------
    >>> encode_dim("freq_band")
    'freq%5Fband'
    """
    return _pct_encode(_stringify(dim), _DIM_SAFE)


def encode_value(value: Any) -> str:
    """Percent-encode a coordinate value (keeps ``[A-Za-z0-9.+~-]`` literal).

    ``_``, ``@``, ``%``, ``,``, space and non-ASCII characters are escaped as UTF-8 bytes
    with uppercase hex. Non-string values are converted with ``str()`` (``1`` -> ``"1"``,
    ``None`` -> ``"None"``, ``0.5`` -> ``"0.5"``), so ``1`` and ``"1"`` give the same text;
    callers detect such collisions (:func:`find_column_collisions`).

    Examples
    --------
    >>> encode_value("8-12"), encode_value("a_b"), encode_value("a b")
    ('8-12', 'a%5Fb', 'a%20b')
    """
    return _pct_encode(_stringify(value), _VALUE_SAFE)


def decode_dim(text: str) -> str:
    """Inverse of :func:`encode_dim`; rejects invalid, lowercase or unnecessary escapes."""
    return _pct_decode(text, _DIM_SAFE, "dimension")


def decode_value(text: str) -> str:
    """Inverse of :func:`encode_value`; rejects invalid, lowercase or unnecessary escapes."""
    return _pct_decode(text, _VALUE_SAFE, "value")


class ParsedColumn(NamedTuple):
    """A generated wide dataframe column split into its parts (see :func:`parse_column`)."""

    name: str
    """Derivative name."""
    suffix: str
    """Artifact suffix, or ``".error"`` for the error column."""
    coords: tuple[tuple[str, str], ...]
    """Decoded ``(dim, value)`` pairs in column order (sorted by encoded dim); ``()`` when
    scalar."""
    is_error: bool
    """Whether this is the ``name.error`` column."""


def error_column(name: str) -> str:
    """Return the error column of a derivative: ``name.error`` (replaces ``name__error``)."""
    validate_derivative_name(name)
    return name + ".error"


def format_column(name: str, suffix: str, coords: Sequence[tuple[Any, Any]] | None = None) -> str:
    """Build a wide dataframe column ``name suffix ["@" dim-value ("_" dim-value)*]``.

    Parameters
    ----------
    name : str
        Derivative name.
    suffix : str
        Artifact suffix.
    coords : sequence of (dim, value) or None
        Coordinates of one flattened element. ``None`` or empty means a scalar column.
        Dims and values are encoded with :func:`encode_dim` / :func:`encode_value` and the
        pairs are sorted by encoded dim, so the input order does not matter.

    Returns
    -------
    str
        The column name; :func:`parse_column` maps it back.

    Raises
    ------
    NamingError
        For an invalid name or suffix, or when two coords have the same encoded dim.

    Examples
    --------
    >>> format_column("Psd", ".nc", [("freq", "8-12"), ("ch", "C3")])
    'Psd.nc@ch-C3_freq-8-12'
    """
    validate_derivative_name(name)
    validate_artifact_suffix(suffix)
    column = name + suffix
    if not coords:
        return column
    encoded = sorted((encode_dim(d), encode_value(v)) for d, v in coords)
    for (d1, _), (d2, _) in pairwise(encoded):
        if d1 == d2:
            raise NamingError(f"Column {column!r}: dimension {decode_dim(d1)!r} appears twice.")
    return column + "@" + "_".join(f"{d}-{v}" for d, v in encoded)


def parse_column(col: str) -> ParsedColumn | None:
    """Parse a wide dataframe column produced by :func:`format_column` or :func:`error_column`.

    Parameters
    ----------
    col : str
        Column name.

    Returns
    -------
    ParsedColumn or None
        ``None`` for metadata columns (no ``.``, e.g. ``index``, ``dataset``,
        ``file_path``).

    Raises
    ------
    NamingError
        When the column contains ``.`` but is not a canonical generated column (invalid
        name, suffix or encoding, unsorted or repeated dims, coords on an error column).
        Round trip: ``format_column(*parse_column(c)[:3]) == c``.
    """
    if not isinstance(col, str) or "." not in col:
        return None
    head, at, coord_text = col.partition("@")
    name, _, s = head.partition(".")
    problem = _name_problem(name)
    if problem is not None:
        raise NamingError(f"Column {col!r} is invalid. {problem}")
    if s == "error":
        if at:
            raise NamingError(f"Column {col!r} is invalid: an error column has no coordinates.")
        return ParsedColumn(name, ".error", (), True)
    suffix = "." + s
    problem = _suffix_problem(suffix)
    if problem is not None:
        raise NamingError(f"Column {col!r} is invalid. {problem}")
    if not at:
        return ParsedColumn(name, suffix, (), False)
    coords: list[tuple[str, str]] = []
    encoded_dims: list[str] = []
    for pair in coord_text.split("_"):
        dim, dash, value = pair.partition("-")
        if not dash:
            raise NamingError(f"Column {col!r}: coordinate {pair!r} has no '-' separator.")
        coords.append((decode_dim(dim), decode_value(value)))
        encoded_dims.append(dim)
    if any(a >= b for a, b in pairwise(encoded_dims)):
        raise NamingError(f"Column {col!r}: dimensions must be unique and sorted.")
    return ParsedColumn(name, suffix, tuple(coords), False)


def find_column_collisions(
    columns: Iterable[tuple[Any, str]],
) -> dict[str, list[Any]]:
    """Find generated columns produced by more than one origin.

    Parameters
    ----------
    columns : iterable of (origin, column)
        Each generated column with a description of where it came from (for example
        ``("Psd.nc", ("ch", 1))``).

    Returns
    -------
    dict[str, list]
        Column -> list of origins, only for columns produced more than once (in input
        order). Empty when there is no collision.

    Examples
    --------
    >>> find_column_collisions([("int 1", "X.nc@k-1"), ("str '1'", "X.nc@k-1")])
    {'X.nc@k-1': ['int 1', "str '1'"]}
    """
    origins: dict[str, list[Any]] = {}
    for origin, column in columns:
        origins.setdefault(column, []).append(origin)
    return {col: found for col, found in origins.items() if len(found) > 1}
