"""Tests for neurodags.naming (naming version 1, pure functions)."""

import itertools
import random
from decimal import Decimal

import pytest

from neurodags.naming import (
    FIELD_RE,
    NAME_RE,
    SUFFIX_RE,
    NamingError,
    ParsedColumn,
    ParsedOutput,
    check_derivative_names,
    decode_dim,
    decode_number,
    decode_value,
    encode_dim,
    encode_number,
    encode_value,
    entity_pairs,
    error_column,
    find_column_collisions,
    format_column,
    marker_basename,
    output_basename,
    output_prefix_basename,
    parse_column,
    parse_entities,
    parse_output_basename,
    parse_reference,
    repeated_entity_keys,
    validate_artifact_suffix,
    validate_artifact_suffixes,
    validate_derivative_name,
)

# ---------------------------------------------------------------------------
# Derivative names
# ---------------------------------------------------------------------------

VALID_NAMES = [
    "A",
    "a",
    "Clean",
    "clean_eeg",
    "Ep_HiguchiFD",
    "Fit_alpha-0p5_w-6",
    "Fit_alpha-0p5_w-m6",
    "X1",
    "a-b",
    "a__b",
    "a--b",
    "A_1",
    "Psd2",
    "x" * 200,
]

INVALID_NAMES = [
    ("", "empty"),
    ("1abc", "start with an ASCII letter"),
    ("_abc", "start with an ASCII letter"),
    ("-abc", "start with an ASCII letter"),
    ("abc-", "must end with a letter or digit"),
    ("abc_", "must end with a letter or digit"),
    ("a.b", "'.'"),
    ("a@b", "'@'"),
    ("a&b", "'&'"),
    ("a b", "whitespace"),
    ("a\tb", "whitespace"),
    ("a,b", "commas"),
    ("a/b", "path separator"),
    ("a\\b", "path separator"),
    ("a*b", "'*'"),
    ("a?b", "'?'"),
    ("a[b]", "'['"),
    ("a:b", "':'"),
    ("café", "non-ASCII"),
    ("\u0436ord", "non-ASCII"),
    ("abc\n", "whitespace"),
    ("SourceFile", "reserved"),
    ("sourcefile", "reserved"),
    ("SOURCEFILE", "reserved"),
]


@pytest.mark.parametrize("name", VALID_NAMES)
def test_valid_names(name):
    validate_derivative_name(name)
    assert NAME_RE.fullmatch(name)


@pytest.mark.parametrize(("name", "fragment"), INVALID_NAMES)
def test_invalid_names(name, fragment):
    with pytest.raises(NamingError, match=None) as excinfo:
        validate_derivative_name(name)
    assert fragment in str(excinfo.value)
    assert NAME_RE.fullmatch(name) is None or name.lower() == "sourcefile"


@pytest.mark.parametrize("key", [True, False, None, 1000, 31, 1.5, 0])
def test_non_string_names_say_quote_the_key(key):
    # YAML 1.1: Yes/On -> True, No/Off -> False, ~ -> None, 1_000 -> 1000, 0x1F -> 31
    with pytest.raises(NamingError) as excinfo:
        validate_derivative_name(key)
    msg = str(excinfo.value)
    assert "quote the key" in msg
    assert "YAML" in msg


def test_naming_error_is_value_error():
    assert issubclass(NamingError, ValueError)


def test_check_names_ok_and_warnings():
    assert check_derivative_names(["Clean", "Psd", "Fit_w-6"]) == []
    warnings = check_derivative_names(["A" * 65, "Fit_w-6_w-7", "B" * 64])
    assert len(warnings) == 2
    assert "65 characters" in warnings[0]
    assert "'w'" in warnings[1]


def test_check_names_reports_all_problems_together():
    with pytest.raises(NamingError) as excinfo:
        check_derivative_names(["Clean", "clean", "a.b", True, "SourceFile", "Psd", "Psd-1"])
    msg = str(excinfo.value)
    lines = msg.splitlines()
    assert len(lines) == 5
    assert "differ only in case" in msg
    assert "'.'" in msg
    assert "quote the key" in msg
    assert "reserved" in msg
    assert "'Psd' and 'Psd-1'" in msg


def test_check_names_exact_repeats_count_once():
    assert check_derivative_names(["Clean", "Clean"]) == []


@pytest.mark.parametrize(
    "names",
    [
        ["Clean", "Clean-1"],
        ["Clean", "Clean-12"],
        ["clean", "Clean-1"],
        ["Fit_w-6", "Fit_w-6-2"],
        ["A-1", "A-1-1"],
    ],
)
def test_split_file_rule(names):
    with pytest.raises(NamingError, match="split"):
        check_derivative_names(names)


@pytest.mark.parametrize("names", [["Clean-1"], ["Clean", "Clean-a"], ["Clean", "Clean_1"]])
def test_split_file_rule_allows(names):
    check_derivative_names(names)


def test_check_names_unhashable():
    with pytest.raises(NamingError, match="not a string"):
        check_derivative_names([["a"]])


# ---------------------------------------------------------------------------
# Key-value convention and numbers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Fit_alpha-0p5_w-6", ("Fit", {"alpha": "0p5", "w": "6"})),
        ("Fit_w-6_alpha-0p5", ("Fit", {"w": "6", "alpha": "0p5"})),
        ("Fit", ("Fit", {})),
        ("Ep_HiguchiFD", ("Ep_HiguchiFD", {})),
        ("Fit_w-6_x", ("Fit_w-6_x", {})),
        ("Fit_w-6-7", ("Fit_w-6-7", {})),
        ("Fit_w_6", ("Fit_w_6", {})),
        ("Fit_1w-6", ("Fit_1w-6", {})),
        ("Fit_w-m1", ("Fit", {"w": "m1"})),
        ("a-b", ("a-b", {})),
    ],
)
def test_parse_entities(name, expected):
    assert parse_entities(name) == expected


def test_entity_order_and_identity():
    assert list(parse_entities("Fit_w-6_alpha-0p5")[1]) == ["w", "alpha"]
    assert entity_pairs("Fit_w-6_w-7") == ("Fit", [("w", "6"), ("w", "7")])
    assert entity_pairs("Ep_HiguchiFD") is None


def test_repeated_keys():
    assert repeated_entity_keys("Fit_w-6_w-7_w-8_a-1_a-2") == ["w", "a"]
    assert repeated_entity_keys("Fit_w-6") == []
    assert repeated_entity_keys("Ep_HiguchiFD") == []
    with pytest.raises(NamingError, match="repeats"):
        parse_entities("Fit_w-6_w-7")


@pytest.mark.parametrize(
    ("value", "encoded"),
    [
        (0.5, "0p5"),
        (-1, "m1"),
        (-0.05, "m0p05"),
        (Decimal("1.00"), "1"),
        (1e-3, "0p001"),
        (0, "0"),
        (0.0, "0"),
        (-0.0, "0"),
        (Decimal("-0.000"), "0"),
        (Decimal("0E-5"), "0"),
        (100, "100"),
        (Decimal("1E+3"), "1000"),
        (12.25, "12p25"),
        ("0.50", "0p5"),
        ("-1", "m1"),
        ("+2", "2"),
        ("007", "7"),
        ("1e-3", "0p001"),
        (".5", "0p5"),
        ("5.", "5"),
        (0.1, "0p1"),
        (1e-20, "0p00000000000000000001"),
        (10**30, "1" + "0" * 30),
        (Decimal("-123.456000"), "m123p456"),
    ],
)
def test_encode_number(value, encoded):
    assert encode_number(value) == encoded


@pytest.mark.parametrize(
    "bad", ["1,5", "nan", "inf", "-inf", " 1", "1_000", "", "abc", "0x1F", "1.2.3"]
)
def test_encode_number_rejects_strings(bad):
    with pytest.raises(NamingError):
        encode_number(bad)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), Decimal("NaN"), Decimal("-Infinity")])
def test_encode_number_rejects_non_finite(bad):
    with pytest.raises(NamingError, match="finite"):
        encode_number(bad)


@pytest.mark.parametrize("bad", [True, False, None, [1], 1 + 2j])
def test_encode_number_rejects_types(bad):
    with pytest.raises(TypeError):
        encode_number(bad)


def test_encode_number_numpy_integer():
    np = pytest.importorskip("numpy")
    assert encode_number(np.int64(-3)) == "m3"
    assert encode_number(np.float64(0.25)) == "0p25"


@pytest.mark.parametrize(
    ("encoded", "value"),
    [
        ("0", 0),
        ("1", 1),
        ("m1", -1),
        ("0p5", Decimal("0.5")),
        ("m0p25", Decimal("-0.25")),
        ("m0p05", Decimal("-0.05")),
        ("120", 120),
        ("12p001", Decimal("12.001")),
    ],
)
def test_decode_number(encoded, value):
    decoded = decode_number(encoded)
    assert decoded == value
    assert type(decoded) is type(value)


@pytest.mark.parametrize(
    "bad", ["m0", "01", "1p50", "1p", "p5", "-1", "1.5", "1,5", "", "m", "mm1", "M1", "1P5", "0p0"]
)
def test_decode_number_rejects(bad):
    with pytest.raises(NamingError):
        decode_number(bad)


def test_decode_number_non_string():
    with pytest.raises(NamingError):
        decode_number(5)


def _number_cases():
    rng = random.Random(1234)
    cases = []
    for _ in range(2000):
        whole = rng.choice([0, 0, 1, 7, 10, 999, rng.randrange(10**12)])
        frac_digits = rng.choice([0, 0, 1, 2, 5, 12])
        frac = "".join(rng.choice("0123456789") for _ in range(frac_digits))
        sign = rng.choice(["", "-"])
        text = f"{sign}{whole}" + (f".{frac}" if frac else "")
        cases.append(text)
    return cases


def test_number_round_trips():
    for text in _number_cases():
        enc = encode_number(text)
        dec = decode_number(enc)
        assert Decimal(dec) == Decimal(text), text
        assert encode_number(dec) == enc
        assert encode_number(Decimal(text)) == enc
    for x in [0.5, -0.25, 3.0, 1e-7, 123.456, -9.75]:
        assert Decimal(decode_number(encode_number(x))) == Decimal(repr(x))


def test_encoded_numbers_are_valid_entity_values():
    for text in _number_cases()[:200]:
        name = f"Fit_w-{encode_number(text)}"
        validate_derivative_name(name)
        assert parse_entities(name)[1]["w"] == encode_number(text)


# ---------------------------------------------------------------------------
# Artifact suffixes
# ---------------------------------------------------------------------------

VALID_SUFFIXES = [".nc", ".fif", ".report.html", ".EO_baseline.fif", ".fif.gz", ".1", ".a-b"]
INVALID_SUFFIXES = [
    ".done",
    ".DONE",
    ".x.ERROR",
    ".Skip",
    ".a.error",
    "fif",
    "..x",
    ".a/b",
    ".",
    "",
    ".x.",
    "._x",
    ".-x",
    ".a b",
    ".a@b",
    ".café",
    ".a,b",
]


@pytest.mark.parametrize("suffix", VALID_SUFFIXES)
def test_valid_suffixes(suffix):
    validate_artifact_suffix(suffix)
    validate_artifact_suffixes([suffix])
    assert SUFFIX_RE.fullmatch(suffix)


@pytest.mark.parametrize("suffix", INVALID_SUFFIXES)
def test_invalid_suffixes(suffix):
    with pytest.raises(NamingError):
        validate_artifact_suffix(suffix)


def test_suffix_marker_message():
    with pytest.raises(NamingError, match="reserved for completion markers"):
        validate_artifact_suffix(".x.ERROR")


def test_suffixes_together():
    validate_artifact_suffixes(VALID_SUFFIXES)
    with pytest.raises(NamingError) as excinfo:
        validate_artifact_suffixes([".fif", ".FIF", ".done", "nc", 3])
    lines = str(excinfo.value).splitlines()
    assert len(lines) == 4
    assert any("differ only in case" in line for line in lines)


# ---------------------------------------------------------------------------
# Output basenames
# ---------------------------------------------------------------------------

TRICKY_SOURCES = [
    "sub-01_task-rest_eeg.vhdr",
    "a@b.vhdr",
    "a&b.vhdr",
    "a@@b@.vhdr",
    "file.with.dots.fif",
    "rec[1].edf",
    "my recording.edf",
    "noext",
    ".hidden",
    "café.set",
    "x@Name.nc",
]


@pytest.mark.parametrize("source", TRICKY_SOURCES)
@pytest.mark.parametrize("name", ["Clean", "Fit_alpha-0p5_w-6", "Clean-1"])
@pytest.mark.parametrize("suffix", [".nc", ".report.html", ".EO_baseline.fif", ".fif.gz"])
def test_output_basename_round_trip(source, name, suffix):
    base = output_basename(source, name, suffix)
    assert base == output_prefix_basename(source, name) + suffix
    assert base.count("@") == 1
    parsed = parse_output_basename(base)
    assert parsed == ParsedOutput(source.replace("@", "&"), name, suffix, False, None)


@pytest.mark.parametrize("source", TRICKY_SOURCES)
@pytest.mark.parametrize("marker", ["done", "error", "skip"])
def test_marker_basename_round_trip(source, marker):
    base = marker_basename(source, "Clean", marker)
    parsed = parse_output_basename(base)
    assert parsed == ParsedOutput(source.replace("@", "&"), "Clean", "." + marker, True, marker)


def test_prefix_rewrites_at():
    assert output_prefix_basename("a@b.vhdr", "X") == "a&b.vhdr@X"
    assert output_prefix_basename("a&b.vhdr", "X") == "a&b.vhdr@X"


@pytest.mark.parametrize("source", ["", ".", "..", "a/b.vhdr", "a\\b.vhdr", "a\0b", None])
def test_prefix_rejects_bad_sources(source):
    with pytest.raises(NamingError):
        output_prefix_basename(source, "X")


def test_prefix_and_basename_reject_bad_parts():
    with pytest.raises(NamingError):
        output_prefix_basename("a.vhdr", "a.b")
    with pytest.raises(NamingError):
        output_basename("a.vhdr", "X", ".done")
    with pytest.raises(NamingError):
        marker_basename("a.vhdr", "X", "DONE")


@pytest.mark.parametrize(
    "basename",
    [
        "sub-01.vhdr",
        "no_at_here.nc",
        "@Name.nc",
        "src@Name",
        "src@Name.",
        "src@1Name.nc",
        "src@Name..nc",
        "src@Name.a b",
        "src@.nc",
    ],
)
def test_parse_output_basename_non_outputs(basename):
    assert parse_output_basename(basename) is None


def test_parse_output_basename_details():
    assert parse_output_basename("a&b.vhdr@Clean-1.fif") == ParsedOutput(
        "a&b.vhdr", "Clean-1", ".fif", False, None
    )
    parsed = parse_output_basename("src@X.y.DONE")
    assert parsed is not None
    assert parsed.is_marker
    assert parsed.marker == "done"
    assert parse_output_basename(None) is None


# ---------------------------------------------------------------------------
# References
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("ref", "expected"),
    [
        ("SourceFile", ("SourceFile", None)),
        ("Clean", ("Clean", None)),
        ("Clean.fif", ("Clean", ".fif")),
        ("Clean.EO_baseline.fif", ("Clean", ".EO_baseline.fif")),
        ("Fit_alpha-0p5_w-6.nc", ("Fit_alpha-0p5_w-6", ".nc")),
        ("Report.report.html", ("Report", ".report.html")),
    ],
)
def test_parse_reference(ref, expected):
    assert parse_reference(ref) == expected


@pytest.mark.parametrize(
    "ref",
    [
        "SourceFile.fif",
        "sourcefile",
        "SOURCEFILE.fif",
        "Clean.",
        "Clean..fif",
        "Clean.done",
        "Clean.x.error",
        "1Clean.fif",
        ".fif",
        "",
        "Clean@x.fif",
        "Clean.a b",
    ],
)
def test_parse_reference_invalid(ref):
    with pytest.raises(NamingError):
        parse_reference(ref)


@pytest.mark.parametrize("ref", [True, None, 1])
def test_parse_reference_non_string(ref):
    with pytest.raises(NamingError, match="quote the key"):
        parse_reference(ref)


@pytest.mark.parametrize(
    ("ref", "groups"),
    [("id.0", ("0", None)), ("id.12.field", ("12", "field")), ("id.3._x1", ("3", "_x1"))],
)
def test_field_re(ref, groups):
    match = FIELD_RE.fullmatch(ref)
    assert match is not None
    assert match.groups() == groups


@pytest.mark.parametrize("ref", ["id.", "id.x", "id.1.", "id.1.2a", "id.1.a.b", "id.1.é", "x.1"])
def test_field_re_rejects(ref):
    assert FIELD_RE.fullmatch(ref) is None


# ---------------------------------------------------------------------------
# Dataframe columns
# ---------------------------------------------------------------------------

TRICKY_VALUES = [
    "8-12",
    "8~12",
    "a_b",
    "a|b",
    "a@b",
    "%",
    "%41",
    ",",
    " ",
    "a b",
    "",
    "café",
    "α-band",
    "\U0001f600",
    "C3",
    "1.5",
    "+1",
    "a.b.c",
    "a-b",
    "a~b",
    "_",
    "-",
    "a/b",
    "line\nbreak",
    "None",
]

TRICKY_DIMS = ["ch", "freq_band", "a-b", "a~b", "x", "dim 1", "α", "", "Z9"]


@pytest.mark.parametrize("value", TRICKY_VALUES)
def test_value_round_trip(value):
    enc = encode_value(value)
    assert set(enc) <= set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.+~-%")
    assert "_" not in enc
    assert "@" not in enc
    assert decode_value(enc) == value


@pytest.mark.parametrize("dim", TRICKY_DIMS + TRICKY_VALUES)
def test_dim_round_trip(dim):
    enc = encode_dim(dim)
    assert set(enc) <= set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789%")
    assert decode_dim(enc) == dim


@pytest.mark.parametrize(
    ("raw", "enc"),
    [
        ("8-12", "8-12"),
        ("8~12", "8~12"),
        ("a_b", "a%5Fb"),
        ("a@b", "a%40b"),
        ("%", "%25"),
        (",", "%2C"),
        (" ", "%20"),
        ("é", "%C3%A9"),
        ("", ""),
    ],
)
def test_encode_value_examples(raw, enc):
    assert encode_value(raw) == enc


def test_encode_dim_examples():
    assert encode_dim("freq_band") == "freq%5Fband"
    assert encode_dim("a-b") == "a%2Db"
    assert encode_dim("a.b") == "a%2Eb"


@pytest.mark.parametrize(
    ("value", "text"), [(1, "1"), (None, "None"), (0.5, "0.5"), (True, "True"), (-2, "-2")]
)
def test_encode_value_non_str(value, text):
    assert encode_value(value) == text
    assert encode_dim(None) == "None"


@pytest.mark.parametrize(
    "bad",
    [
        "%",  # incomplete escape
        "%4",
        "%zz",
        "%5f",  # lowercase hex
        "%41",  # unnecessary escape of 'A'
        "%2E",  # unnecessary escape of '.'
        "a b",  # literal space
        "a_b",  # literal underscore
        "a@b",
        "%C3",  # truncated UTF-8
        "%FF",  # invalid UTF-8
        "é",  # literal non-ASCII
    ],
)
def test_decode_value_rejects(bad):
    with pytest.raises(NamingError):
        decode_value(bad)


@pytest.mark.parametrize("bad", ["a-b", "a.b", "a~b", "%61", "%2d"])
def test_decode_dim_rejects(bad):
    with pytest.raises(NamingError):
        decode_dim(bad)


def test_decode_rejects_non_string():
    with pytest.raises(NamingError):
        decode_value(1)


def test_format_column_scalar_and_error():
    assert format_column("Psd", ".nc") == "Psd.nc"
    assert format_column("Psd", ".nc", []) == "Psd.nc"
    assert error_column("Psd") == "Psd.error"
    assert parse_column("Psd.error") == ParsedColumn("Psd", ".error", (), True)
    assert parse_column("Psd.nc") == ParsedColumn("Psd", ".nc", (), False)
    with pytest.raises(NamingError):
        error_column("a.b")


def test_format_column_sorts_dims():
    col = format_column("Psd", ".nc", [("freq", "8-12"), ("ch", "C3")])
    assert col == "Psd.nc@ch-C3_freq-8-12"
    assert col == format_column("Psd", ".nc", [("ch", "C3"), ("freq", "8-12")])
    # Sorted by encoded dim: "%" (0x25) sorts before digits and letters.
    assert format_column("X", ".nc", [("a", 1), ("_a", 2)]) == "X.nc@%5Fa-2_a-1"


def test_format_column_rejects():
    with pytest.raises(NamingError, match="appears twice"):
        format_column("X", ".nc", [("ch", "a"), ("ch", "b")])
    with pytest.raises(NamingError):
        format_column("X.y", ".nc")
    with pytest.raises(NamingError):
        format_column("X", ".error")
    with pytest.raises(NamingError):
        format_column("X", "nc")


@pytest.mark.parametrize("column", ["index", "dataset", "file_path", "subject", "", "a@b", None])
def test_parse_column_metadata(column):
    assert parse_column(column) is None


@pytest.mark.parametrize(
    "column",
    [
        "a.b.c@",  # empty coords
        "Psd.nc@ch",  # pair without '-'
        "Psd.nc@ch-C3_",  # empty pair
        "Psd.nc@freq-1_ch-2",  # unsorted
        "Psd.nc@ch-1_ch-2",  # repeated dim
        "Psd.nc@ch-a b",  # unencoded space
        "Psd.nc@ch-%41",  # unnecessary escape
        "Psd.error@ch-1",  # coords on error column
        "1Psd.nc",
        "Psd.done",
        ".nc",
        "Psd..nc",
    ],
)
def test_parse_column_invalid(column):
    with pytest.raises(NamingError):
        parse_column(column)


def test_parse_column_escaped_dash_in_dim():
    # '-' is escaped in dims, so the first literal '-' always separates dim and value.
    assert parse_column("Psd.nc@c%2Dh-1-2") == ParsedColumn("Psd", ".nc", (("c-h", "1-2"),), False)


def _column_cases():
    rng = random.Random(42)
    alphabet = [*"aZ09_-~.+@%, |&", "é", "α", "\U0001f600", "/", ":"]
    cases = []
    for _ in range(1500):
        n_dims = rng.choice([0, 1, 2, 3, 4])
        dims = set()
        coords = []
        while len(coords) < n_dims:
            dim = "".join(rng.choice(alphabet) for _ in range(rng.randrange(0, 5)))
            if dim in dims:
                continue
            dims.add(dim)
            value = "".join(rng.choice(alphabet) for _ in range(rng.randrange(0, 6)))
            coords.append((dim, value))
        name = rng.choice(["Psd", "Fit_alpha-0p5_w-6", "a-b", "X1"])
        suffix = rng.choice([".nc", ".report.html", ".EO_baseline.fif"])
        cases.append((name, suffix, coords))
    return cases


def test_column_round_trip_generated():
    seen = {}
    for name, suffix, coords in _column_cases():
        col = format_column(name, suffix, coords)
        parsed = parse_column(col)
        assert parsed is not None
        assert parsed.name == name
        assert parsed.suffix == suffix
        assert not parsed.is_error
        assert sorted(parsed.coords, key=lambda p: encode_dim(p[0])) == list(parsed.coords)
        assert dict(parsed.coords) == dict(coords)
        assert format_column(*parsed[:3]) == col
        # Injectivity: one column per distinct (name, suffix, coordinate set).
        identity = (name, suffix, frozenset(coords))
        assert seen.setdefault(col, identity) == identity


@pytest.mark.parametrize(("v1", "v2"), [("a-b", "a~b"), ("a_b", "a|b"), ("a,b", "a.b"), ("", " ")])
def test_previously_colliding_values_are_distinct(v1, v2):
    c1 = format_column("Psd", ".nc", [("band", v1)])
    c2 = format_column("Psd", ".nc", [("band", v2)])
    assert c1 != c2
    assert parse_column(c1).coords == (("band", v1),)
    assert parse_column(c2).coords == (("band", v2),)


def test_pairwise_injective_over_tricky_values():
    cols = {}
    for dim, value in itertools.product(TRICKY_DIMS, TRICKY_VALUES):
        col = format_column("X", ".nc", [(dim, value)])
        assert cols.setdefault(col, (dim, value)) == (dim, value)
    assert len(cols) == len(TRICKY_DIMS) * len(TRICKY_VALUES)


def test_error_column_never_collides_with_artifact_columns():
    # ".error" is reserved as an artifact suffix, so name.error is only the error column.
    with pytest.raises(NamingError):
        format_column("Psd", ".error")
    assert parse_column(error_column("Psd")).is_error


def test_find_column_collisions():
    entries = [
        (("X.nc", 1), format_column("X", ".nc", [("k", 1)])),
        (("X.nc", "1"), format_column("X", ".nc", [("k", "1")])),
        (("X.nc", 2), format_column("X", ".nc", [("k", 2)])),
    ]
    assert find_column_collisions(entries) == {"X.nc@k-1": [("X.nc", 1), ("X.nc", "1")]}
    assert find_column_collisions(entries[1:]) == {}
