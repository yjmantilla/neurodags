"""Discovery and indices (0.4.0 spec §8): sorted enumeration, one enumeration per
invocation, ``--only-index`` errors and frozen input lists (``neurodags inputs``)."""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from neurodags import iterators
from neurodags.cli import main
from neurodags.definitions import DatasetConfig
from neurodags.iterators import (
    INPUTS_HEADER,
    IndexSelectionError,
    InputsFileError,
    discover_inputs,
    enumerate_inputs,
    load_inputs,
    read_inputs_file,
    write_inputs_file,
)
from neurodags.orchestrators import build_derivative_dataframe, run_pipeline
from neurodags.utils import DatasetConfigError, lexical_path_key


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("")
    return path


def _dataset(root: Path, pattern: str = "**/*.vhdr", **kwargs) -> DatasetConfig:
    return DatasetConfig(
        name=root.name, source_root=str(root), file_pattern=str(root / pattern), **kwargs
    )


_NAMES = [
    "sub-2/rec.vhdr",
    "sub-10/rec.vhdr",
    "sub-1/rec.vhdr",
    "B/rec.vhdr",
    "a/rec.vhdr",
    "a.b/rec.vhdr",
    "a/b/rec.vhdr",
]


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    root = tmp_path / "raw"
    for name in _NAMES:
        _touch(root / name)
    return root


# ---------------------------------------------------------------------------
# Sorting, limits, dataset order
# ---------------------------------------------------------------------------


def test_discovery_sorted_by_rel_code_point_order(tree: Path):
    enum = discover_inputs({"ds": _dataset(tree)})
    rels = [e.rel for e in enum.entries]
    assert rels == sorted(_NAMES)  # code-point order of the POSIX rel
    # no natural sort: sub-10 sorts before sub-2; upper case before lower case
    assert rels.index("sub-10/rec.vhdr") < rels.index("sub-2/rec.vhdr")
    assert rels.index("B/rec.vhdr") < rels.index("a/rec.vhdr")
    assert [e.index for e in enum.entries] == list(range(len(_NAMES)))
    for e in enum.entries:
        assert e.path == os.path.join(str(tree), *e.rel.split("/"))
        assert not e.missing


def test_discovery_order_independent_of_filesystem_order(tree: Path, monkeypatch):
    datasets = {"ds": _dataset(tree)}
    expected = [(e.rel, e.path) for e in discover_inputs(datasets).entries]

    real_glob = iterators.glob.glob
    for order in (reversed, lambda xs: xs[3:] + xs[:3]):

        def fake_glob(pattern, recursive=False, _order=order):
            return list(_order(sorted(real_glob(pattern, recursive=recursive))))

        monkeypatch.setattr(iterators.glob, "glob", fake_glob)
        assert [(e.rel, e.path) for e in discover_inputs(datasets).entries] == expected


def test_discovery_deduplicates_identical_paths(tree: Path, monkeypatch):
    real_glob = iterators.glob.glob

    def doubled(pattern, recursive=False):
        found = real_glob(pattern, recursive=recursive)
        return found + [f.replace(os.sep, os.sep + os.sep, 1) for f in found]

    monkeypatch.setattr(iterators.glob, "glob", doubled)
    enum = discover_inputs({"ds": _dataset(tree)})
    assert len(enum.entries) == len(_NAMES)


def _rels(datasets) -> list[str]:
    return [e.rel for e in discover_inputs(datasets).entries]


def test_duplicate_glob_hits_keep_bids_split_entry(tmp_path: Path):
    # A pattern with several ``**`` returns the same pathname more than once; the repeated
    # entry file must not count as a second split member (and so drop itself).
    raw = tmp_path / "raw"
    for name in ("x_split-01_meg.fif", "x_split-02_meg.fif"):
        _touch(raw / "sub" / name)
    assert _rels({"d": _dataset(raw, "**/*.fif")}) == ["sub/x_split-01_meg.fif"]
    assert _rels({"d": _dataset(raw, "**/**/*.fif")}) == ["sub/x_split-01_meg.fif"]


@pytest.mark.parametrize(
    ("file_pattern", "exclude_pattern"),
    [("./**/*.txt", "**/*.txt"), ("**/*.txt", "./**/*.txt"), ("**/*.txt", "sub//../sub/*.txt")],
)
def test_exclude_pattern_matches_equivalent_spellings(
    tmp_path: Path, file_pattern, exclude_pattern
):
    raw = tmp_path / "raw"
    _touch(raw / "sub" / "x.txt")
    ds = DatasetConfig(
        name="d",
        source_root=str(raw),
        file_pattern=str(raw) + "/" + file_pattern,
        exclude_pattern=str(raw) + "/" + exclude_pattern,
    )
    assert _rels({"d": ds}) == []


def test_discovery_returns_files_only_and_excludes_output_root(tmp_path: Path):
    raw = tmp_path / "raw"
    _touch(raw / "sub" / "x.txt")
    ds = DatasetConfig(
        name="d",
        source_root=str(raw),
        file_pattern=str(raw / "**" / "*"),
        derivatives_path=str(raw / "out"),
    )
    before = [(e.index, e.rel) for e in discover_inputs({"d": ds}).entries]
    assert before == [(0, "sub/x.txt")]
    # Creating the output tree (with an output in it) must not shift source indices.
    _touch(raw / "out" / "generated.txt")
    assert [(e.index, e.rel) for e in discover_inputs({"d": ds}).entries] == before
    assert discover_inputs({"d": ds}, max_files_per_dataset=1).entries[0].rel == "sub/x.txt"


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="needs symlinks")
def test_discovery_keeps_symlinked_files_under_their_lexical_path(tmp_path: Path):
    raw = tmp_path / "raw"
    target = _touch(tmp_path / "elsewhere" / "real.txt")
    (raw / "sub").mkdir(parents=True)
    os.symlink(target, raw / "sub" / "link.txt")
    os.symlink(tmp_path / "elsewhere", raw / "dirlink")
    ds = DatasetConfig(name="d", source_root=str(raw), file_pattern=str(raw / "*" / "*.txt"))
    entries = discover_inputs({"d": ds}).entries
    # Files behind a directory symlink keep their lexical path below source_root.
    assert [e.rel for e in entries] == ["dirlink/real.txt", "sub/link.txt"]
    assert entries[1].path == str(raw / "sub" / "link.txt")
    # A symlink to a directory is not an input itself.
    ds.file_pattern = str(raw / "*")
    assert _rels({"d": ds}) == []


@pytest.fixture
def recordings(tmp_path: Path) -> Path:
    """CTF ``.ds`` and EGI ``.mff`` recordings (directories), a file and a plain folder."""
    raw = tmp_path / "raw"
    _touch(raw / "sub-01_task-rest_meg.ds" / "sub-01_task-rest_meg.meg4")
    _touch(raw / "sub-01_task-rest_meg.ds" / "hz.ds" / "hz.meg4")  # nested: still inside
    _touch(raw / "sub-02_eeg.mff" / "signal1.bin")
    _touch(raw / "UPPER.DS" / "x.meg4")  # extensions are case-insensitive
    _touch(raw / "plain.vhdr")
    _touch(raw / "sub" / "x.txt")
    _touch(raw / "notes.dsx" / "y.txt")  # not a container: an ordinary folder
    return raw


@pytest.mark.parametrize(
    ("pattern", "expected"),
    [
        ("**/*.ds", ["sub-01_task-rest_meg.ds"]),  # the glob itself is case-sensitive
        ("*.mff", ["sub-02_eeg.mff"]),
        (
            "**/*",
            [
                "UPPER.DS",
                "notes.dsx/y.txt",
                "plain.vhdr",
                "sub-01_task-rest_meg.ds",
                "sub-02_eeg.mff",
                "sub/x.txt",
            ],
        ),
        ("**/*.vhdr", ["plain.vhdr"]),
    ],
)
def test_recording_containers_are_single_inputs(recordings: Path, pattern, expected):
    # Containers are one input each (never their inner files); other folders are no input.
    assert _rels({"d": _dataset(recordings, pattern)}) == expected


def test_recording_container_constant():
    assert iterators.RECORDING_CONTAINER_EXTENSIONS == (".ds", ".mff")


def test_output_root_named_like_a_container_is_not_an_input(recordings: Path):
    ds = _dataset(recordings, "**/*", derivatives_path=str(recordings / "out.ds"))
    _touch(recordings / "out.ds" / "plain.vhdr@A.txt")
    _touch(recordings / "out.ds" / "inner.mff" / "z.bin")
    assert "out.ds" not in _rels({"d": ds})
    assert not any(r.startswith("out.ds") for r in _rels({"d": ds}))


def test_cli_counts_container_recordings(recordings: Path, tmp_path: Path):
    pipe = tmp_path / "pipeline.yml"
    pipe.write_text(
        yaml.safe_dump(
            {
                "naming_version": 1,
                "datasets": {
                    "d": {
                        "name": "d",
                        "source_root": str(recordings),
                        "file_pattern": str(recordings / "**" / "*.ds"),
                    }
                },
            }
        )
    )
    # In a subprocess: the first in-process main() would bind logging to capsys' stream.
    result = subprocess.run(
        [sys.executable, "-m", "neurodags.cli", "count-inputs", str(pipe)],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(Path(iterators.__file__).parent.parent)},
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "1"  # the one *.ds recording, not its inner files


def test_frozen_list_with_containers_round_trips(recordings: Path, tmp_path: Path):
    datasets = {"d": _dataset(recordings, "**/*")}
    out = write_inputs_file(discover_inputs(datasets), tmp_path / "inputs.tsv")
    loaded = load_inputs(out, datasets)
    assert loaded.entries == discover_inputs(datasets).entries
    assert loaded.missing_sources == []


def test_frozen_list_rejects_entries_that_are_not_inputs(recordings: Path, tmp_path: Path):
    datasets = {"d": _dataset(recordings, "**/*")}
    out = tmp_path / "inputs.tsv"
    out.write_text(f"{INPUTS_HEADER}\n0\td\tplain.vhdr\n1\td\tsub\n2\td\tsub-02_eeg.mff\n")
    with pytest.raises(InputsFileError, match=r"1 \(d: sub\)"):
        load_inputs(out, datasets)
    # A listed container that disappeared is missing-source, like a file.
    out.write_text(f"{INPUTS_HEADER}\n0\td\tplain.vhdr\n1\td\tgone.mff\n")
    assert [e.index for e in load_inputs(out, datasets).missing_sources] == [1]


@pytest.mark.skipif(os.name == "nt", reason="POSIX leading-slash rule")
def test_exclude_pattern_with_leading_double_slash(tmp_path: Path):
    raw = tmp_path / "raw"
    _touch(raw / "x.txt")
    for file_pattern, exclude in [
        (str(raw / "*.txt"), "/" + str(raw / "*.txt")),
        ("/" + str(raw / "*.txt"), str(raw / "*.txt")),
    ]:
        ds = DatasetConfig(
            name="d", source_root=str(raw), file_pattern=file_pattern, exclude_pattern=exclude
        )
        assert _rels({"d": ds}) == []
    # And a pattern spelled with a leading // is still below a /-spelled source_root.
    ds = DatasetConfig(name="d", source_root=str(raw), file_pattern="/" + str(raw / "*.txt"))
    assert _rels({"d": ds}) == ["x.txt"]


@pytest.mark.parametrize(
    ("path", "key"),
    [
        ("/a//b/./c/../d", "/a/b/d"),
        ("//a/b", "/a/b" if os.name != "nt" else os.path.normpath("//a/b")),
        ("///a", os.path.normpath("///a")),
        ("rel/./x", os.path.normpath("rel/x")),
    ],
)
def test_lexical_path_key(path, key):
    assert lexical_path_key(path) == key


def test_limit_after_sort_is_a_prefix_per_dataset(tmp_path: Path):
    r1, r2 = tmp_path / "one", tmp_path / "two"
    for i in (3, 1, 2, 10):
        _touch(r1 / f"sub-{i}" / "x.vhdr")
    for i in (5, 4):
        _touch(r2 / f"sub-{i}" / "x.vhdr")
    datasets = {"one": _dataset(r1), "two": _dataset(r2)}
    full = discover_inputs(datasets)
    per_full = {d: [e.rel for e in full.entries if e.dataset == d] for d in datasets}
    for n in range(0, 6):
        limited = discover_inputs(datasets, max_files_per_dataset=n)
        for d in datasets:
            assert [e.rel for e in limited.entries if e.dataset == d] == per_full[d][:n]
        assert [e.index for e in limited.entries] == list(range(len(limited.entries)))
        # the layout root (source_root) does not depend on the limit
        if n:
            assert limited.source_roots == full.source_roots == {"one": str(r1), "two": str(r2)}


def test_datasets_concatenated_in_yaml_order_skipping_skipped(tmp_path: Path):
    roots = {name: tmp_path / name for name in ("zeta", "alpha", "mid")}
    for root in roots.values():
        _touch(root / "a.vhdr")
        _touch(root / "b.vhdr")
    datasets = {
        "zeta": _dataset(roots["zeta"], "*.vhdr"),
        "mid": _dataset(roots["mid"], "*.vhdr", skip=True),
        "alpha": _dataset(roots["alpha"], "*.vhdr"),
    }
    enum = discover_inputs(datasets)
    assert [(e.index, e.dataset, e.rel) for e in enum.entries] == [
        (0, "zeta", "a.vhdr"),
        (1, "zeta", "b.vhdr"),
        (2, "alpha", "a.vhdr"),
        (3, "alpha", "b.vhdr"),
    ]


def test_file_outside_source_root_is_a_configuration_error(tmp_path: Path):
    _touch(tmp_path / "raw" / "in.vhdr")
    _touch(tmp_path / "other" / "out.vhdr")
    ds = DatasetConfig(
        name="d", source_root=str(tmp_path / "raw"), file_pattern=str(tmp_path / "**/*.vhdr")
    )
    with pytest.raises(DatasetConfigError, match=r"out\.vhdr"):
        discover_inputs({"d": ds})


def test_negative_limit_rejected(tree: Path):
    with pytest.raises(ValueError, match="max_files_per_dataset"):
        discover_inputs({"ds": _dataset(tree)}, max_files_per_dataset=-1)


# ---------------------------------------------------------------------------
# --only-index selection
# ---------------------------------------------------------------------------


def test_select_rejects_unknown_and_negative_indices(tree: Path):
    enum = discover_inputs({"ds": _dataset(tree)})
    assert [e.index for e in enum.select([3, 1])] == [1, 3]
    assert [e.index for e in enum.select(2)] == [2]
    with pytest.raises(IndexSelectionError, match="unknown: 99"):
        enum.select([1, 99])
    with pytest.raises(IndexSelectionError, match="negative: -1"):
        enum.select(-1)
    assert isinstance(IndexSelectionError("x"), ValueError)


# ---------------------------------------------------------------------------
# Frozen input lists
# ---------------------------------------------------------------------------


def test_inputs_file_round_trip(tree: Path, tmp_path: Path):
    datasets = {"ds": _dataset(tree)}
    enum = discover_inputs(datasets)
    out = write_inputs_file(enum, tmp_path / "lists" / "inputs.tsv")
    lines = Path(out).read_text().splitlines()
    assert lines[0] == INPUTS_HEADER == "# neurodags-inputs naming_version=1"
    assert lines[1:] == [f"{e.index}\tds\t{e.rel}" for e in enum.entries]
    assert read_inputs_file(out, datasets) == [(e.index, "ds", e.rel) for e in enum.entries]
    loaded = load_inputs(out, datasets)
    assert loaded.entries == enum.entries
    assert loaded.inputs_file == out
    assert loaded.not_in_list == {}
    assert loaded.missing_sources == []


def test_inputs_added_and_missing_files(tree: Path, tmp_path: Path):
    datasets = {"ds": _dataset(tree)}
    enum = discover_inputs(datasets)
    out = write_inputs_file(enum, tmp_path / "inputs.tsv")
    gone = enum.entries[2]
    os.remove(gone.path)
    _touch(tree / "sub-0" / "rec.vhdr")  # would sort first if rediscovered

    loaded = enumerate_inputs(datasets, inputs=out)
    assert [(e.index, e.rel) for e in loaded.entries] == [(e.index, e.rel) for e in enum.entries]
    assert [e.index for e in loaded.missing_sources] == [gone.index]
    assert loaded.not_in_list == {"ds": ["sub-0/rec.vhdr"]}
    assert loaded.n_not_in_list == 1
    # selecting the missing entry explicitly is an error; others are fine
    with pytest.raises(IndexSelectionError, match="missing-source"):
        loaded.select([gone.index])
    assert [e.index for e in loaded.select([0, 1])] == [0, 1]


def test_inputs_mount_independence(tmp_path: Path):
    local, hpc = tmp_path / "local" / "raw", tmp_path / "scratch" / "raw"
    for root in (local, hpc):
        for name in _NAMES:
            _touch(root / name)
    ds = DatasetConfig(
        name="d",
        source_root={"local": str(local), "hpc": str(hpc)},
        file_pattern={"local": str(local / "**/*.vhdr"), "hpc": str(hpc / "**/*.vhdr")},
    )
    datasets = {"d": ds}
    out = write_inputs_file(discover_inputs(datasets, "local"), tmp_path / "inputs.tsv")
    on_local = load_inputs(out, datasets, "local")
    on_hpc = load_inputs(out, datasets, "hpc")
    assert [(e.index, e.rel) for e in on_local.entries] == [
        (e.index, e.rel) for e in on_hpc.entries
    ]
    assert all(e.path.startswith(str(hpc) + os.sep) for e in on_hpc.entries)
    assert all(e.path.startswith(str(local) + os.sep) for e in on_local.entries)
    assert not on_hpc.missing_sources
    assert not on_hpc.not_in_list


@pytest.mark.parametrize(
    ("body", "match"),
    [
        ("0\tds\ta/rec.vhdr\n", "first line"),  # no header
        (f"{INPUTS_HEADER}\n0\tnope\ta/rec.vhdr\n", "not in the configuration"),
        (f"{INPUTS_HEADER}\n0\tds\ta/rec.vhdr\n0\tds\tB/rec.vhdr\n", "duplicate index 0"),
        (f"{INPUTS_HEADER}\n0\tds\ta/rec.vhdr\n2\tds\tB/rec.vhdr\n", "contiguous"),
        (f"{INPUTS_HEADER}\n1\tds\ta/rec.vhdr\n", "contiguous"),
        (f"{INPUTS_HEADER}\n-1\tds\ta/rec.vhdr\n", "non-negative"),
        (f"{INPUTS_HEADER}\n0\tds\t../x.vhdr\n", "relative POSIX path"),
        (f"{INPUTS_HEADER}\n0\tds\t/abs/x.vhdr\n", "relative POSIX path"),
        (f"{INPUTS_HEADER}\n0\tds\n", "expected index"),
        (f"{INPUTS_HEADER}\n0\tds\ta/rec.vhdr\n1\tds\ta/rec.vhdr\n", "already listed"),
        ("# neurodags-inputs naming_version=2\n0\tds\ta/rec.vhdr\n", "first line"),
    ],
)
def test_inputs_file_validation(tree: Path, tmp_path: Path, body: str, match: str):
    path = tmp_path / "inputs.tsv"
    path.write_text(body)
    with pytest.raises(InputsFileError, match=match):
        load_inputs(path, {"ds": _dataset(tree)})


def test_inputs_rejects_skipped_dataset(tree: Path, tmp_path: Path):
    path = tmp_path / "inputs.tsv"
    path.write_text(f"{INPUTS_HEADER}\n0\tds\ta/rec.vhdr\n")
    with pytest.raises(InputsFileError, match="skip"):
        load_inputs(path, {"ds": _dataset(tree, skip=True)})


def test_inputs_with_max_files_is_an_error(tree: Path, tmp_path: Path):
    datasets = {"ds": _dataset(tree)}
    out = write_inputs_file(discover_inputs(datasets), tmp_path / "inputs.tsv")
    with pytest.raises(InputsFileError, match="max_files_per_dataset"):
        enumerate_inputs(datasets, inputs=out, max_files_per_dataset=1)


# ---------------------------------------------------------------------------
# Orchestrators: one enumeration per invocation, only_index errors, inputs=
# ---------------------------------------------------------------------------


@pytest.fixture
def glob_calls(monkeypatch):
    calls: list[str] = []
    real = iterators.get_files_from_pattern

    def spy(pattern, *args, **kwargs):
        calls.append(pattern)
        return real(pattern, *args, **kwargs)

    monkeypatch.setattr(iterators, "get_files_from_pattern", spy)
    return calls


def test_run_pipeline_enumerates_once(dummy_pipeline, glob_calls):
    cfg = dummy_pipeline["config"]
    result = run_pipeline(cfg, derivatives=["BasicPrep", "Spectrum"], dry_run=True)
    assert set(result["derivative"]) == {"BasicPrep", "Spectrum"}
    assert len(glob_calls) == 1  # one dataset, discovered once for both derivatives


def test_build_dataframe_enumerates_once(dummy_pipeline, glob_calls):
    cfg = dummy_pipeline["config"]
    build_derivative_dataframe(cfg, max_files_per_dataset=1)
    assert len(glob_calls) == 1


def test_run_pipeline_unknown_index_raises_before_anything(dummy_pipeline, tmp_path):
    cfg = dummy_pipeline["config"]
    with pytest.raises(ValueError, match="unknown: 5"):
        run_pipeline(cfg, derivatives=["BasicPrep"], dry_run=True, only_index=[0, 5])
    with pytest.raises(ValueError, match="negative: -1"):
        run_pipeline(cfg, derivatives=["BasicPrep"], dry_run=True, only_index=-1)
    plan = run_pipeline(cfg, derivatives=["BasicPrep"], dry_run=True, only_index=[1])
    assert list(plan["index"]) == [1]


def test_only_index_refers_to_limited_enumeration(dummy_pipeline):
    cfg = dummy_pipeline["config"]
    with pytest.raises(ValueError, match="unknown: 1"):
        run_pipeline(
            cfg, derivatives=["BasicPrep"], dry_run=True, max_files_per_dataset=1, only_index=1
        )


def _write_pipeline(dummy_pipeline, tmp_path: Path) -> Path:
    cfg = copy.deepcopy(dummy_pipeline["config"])
    path = tmp_path / "pipeline.yml"
    path.write_text(yaml.safe_dump(cfg))
    return path


@pytest.fixture
def frozen(dummy_pipeline, tmp_path, monkeypatch, capsys):
    """pipeline.yml + inputs.tsv written by the CLI; cwd = tmp_path."""
    monkeypatch.chdir(tmp_path)
    pipeline = _write_pipeline(dummy_pipeline, tmp_path)
    assert main(["inputs", str(pipeline), "-o", "inputs.tsv"]) == 0
    assert "wrote 2 input(s)" in capsys.readouterr().out
    rows = Path("inputs.tsv").read_text().splitlines()
    assert rows[0] == INPUTS_HEADER
    assert len(rows) == 3
    return pipeline


def test_cli_inputs_prints_to_stdout(dummy_pipeline, tmp_path, capsys):
    pipeline = _write_pipeline(dummy_pipeline, tmp_path)
    assert main(["inputs", str(pipeline)]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0] == INPUTS_HEADER
    assert [line.split("\t")[:2] for line in out[1:]] == [
        ["0", "test_dataset"],
        ["1", "test_dataset"],
    ]


def test_cli_count_inputs_with_inputs(frozen, dummy_pipeline, capsys):
    _touch(dummy_pipeline["data_dir"] / "sub-new" / "x.vhdr")  # not in list
    assert main(["count-inputs", str(frozen)]) == 0
    assert capsys.readouterr().out.strip() == "3"  # discovery sees the new file
    assert main(["count-inputs", str(frozen), "--inputs", "inputs.tsv"]) == 0
    assert capsys.readouterr().out.strip() == "2"  # list entries


def _drop_first_listed(frozen: Path) -> tuple[int, str]:
    rel = Path("inputs.tsv").read_text().splitlines()[1].split("\t")[2]
    source = Path(yaml.safe_load(frozen.read_text())["datasets"]["test_dataset"]["source_root"])
    os.remove(source / rel)
    return 0, rel


def test_cli_status_and_dry_run_with_inputs(frozen, dummy_pipeline, capsys):
    idx, _ = _drop_first_listed(frozen)
    _touch(dummy_pipeline["data_dir"] / "sub-new" / "x.vhdr")

    rc = main(["status", str(frozen), "--derivative", "BasicPrep", "--inputs", "inputs.tsv"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "1 file(s) on disk are not in the input list" in out
    assert f"missing-source), not processed: indices {idx}" in out

    rc = main(
        [
            "status",
            str(frozen),
            "--derivative",
            "BasicPrep",
            "--inputs",
            "inputs.tsv",
            "--format",
            "json",
        ]
    )
    data = json.loads(capsys.readouterr().out)
    assert data["missing_source_indices"] == [idx]
    assert data["not_in_list"] == 1
    assert data["inputs"].endswith("inputs.tsv")
    assert data["grand_total"]["missing-source"] == 1
    assert data["missing_indices"] == [1]

    # --format indices lists only resubmittable indices (not missing-source)
    rc = main(
        [
            "status",
            str(frozen),
            "--derivative",
            "BasicPrep",
            "--inputs",
            "inputs.tsv",
            "--format",
            "indices",
            "--exit-zero",
        ]
    )
    assert rc == 0
    assert capsys.readouterr().out.strip() == "1"

    assert (
        main(
            [
                "dry-run",
                str(frozen),
                "--derivative",
                "BasicPrep",
                "--inputs",
                "inputs.tsv",
                "--output",
                "plan.csv",
            ]
        )
        == 0
    )
    import pandas as pd

    plan = pd.read_csv("plan.csv")
    assert sorted(plan["index"]) == [0, 1]
    assert list(plan.loc[plan["index"] == idx, "source_status"]) == ["missing-source"]


def test_cli_run_with_inputs_skips_missing_source(frozen, dummy_pipeline, capsys):
    idx, _ = _drop_first_listed(frozen)
    rc = main(["run", str(frozen), "--derivative", "BasicPrep", "--inputs", "inputs.tsv"])
    assert rc == 0
    out_dir = dummy_pipeline["out_dir"]
    assert len(list(out_dir.rglob("*@BasicPrep.fif"))) == 1  # only the existing source
    capsys.readouterr()

    rc = main(
        [
            "run",
            str(frozen),
            "--derivative",
            "BasicPrep",
            "--inputs",
            "inputs.tsv",
            "--only-index",
            str(idx),
        ]
    )
    err = capsys.readouterr().err
    assert rc == 2
    assert "missing-source" in err
    assert "Traceback" not in err

    with pytest.raises(RuntimeError, match="missing-source"):
        run_pipeline(
            str(frozen),
            derivatives=["BasicPrep"],
            inputs="inputs.tsv",
            raise_on_error=True,
        )


def test_dataframe_with_inputs_uses_list_indices(frozen, dummy_pipeline):
    cfg = dummy_pipeline["config"]
    # Add a file that would sort first under discovery: the list keeps the old indices.
    data_dir = dummy_pipeline["data_dir"]
    first = sorted(data_dir.rglob("*.vhdr"))[0]
    df_full = build_derivative_dataframe(cfg, include_derivatives=["BandPowerMean"])
    _touch(data_dir / "aaa" / first.name)  # empty file, never processed with --inputs
    df = build_derivative_dataframe(cfg, include_derivatives=["BandPowerMean"], inputs="inputs.tsv")
    assert list(df["index"]) == list(df_full["index"]) == [0, 1]
    assert list(df["file_path"]) == list(df_full["file_path"])
    with pytest.raises(InputsFileError, match="max_files_per_dataset"):
        build_derivative_dataframe(cfg, inputs="inputs.tsv", max_files_per_dataset=1)


@pytest.mark.parametrize(
    "argv",
    [
        ["dry-run", "--only-index", "9"],
        ["dry-run", "--only-index", "-1"],
        ["dataframe", "--only-index", "9"],
        ["run", "--inputs", "nope.tsv"],
        ["status", "--inputs", "bad.tsv"],
        ["count-inputs", "--inputs", "bad.tsv"],
        ["dry-run", "--inputs", "inputs.tsv", "--max-files-per-dataset", "1"],
    ],
)
def test_cli_selection_errors_are_clean(frozen, capsys, argv):
    Path("bad.tsv").write_text("not a list\n")
    rc = main([argv[0], str(frozen), *argv[1:]])
    err = capsys.readouterr().err
    assert rc == 2
    assert err.startswith("neurodags: error:")
    assert "Traceback" not in err


@pytest.mark.parametrize("pattern", ["per-file", "flat", "chained"])
def test_slurm_templates_use_frozen_lists(dummy_pipeline, tmp_path, capsys, pattern):
    pipeline = _write_pipeline(dummy_pipeline, tmp_path)
    assert main(["slurm-script", str(pipeline), "--pattern", pattern]) == 0
    out = capsys.readouterr().out
    assert f"neurodags inputs {pipeline} -o inputs.tsv" in out
    assert f"count-inputs {pipeline} --inputs inputs.tsv" in out
    assert 'inputs=os.path.abspath("inputs.tsv")' in out
    assert "`" not in out  # unquoted heredocs: no command substitution
