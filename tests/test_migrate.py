"""`neurodags migrate` (spec 0.4.0 section 10): config rewrite, moves, MNE splits, manifests."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import yaml

from neurodags import migrate
from neurodags._legacy import snake_to_camel
from neurodags.cli import main
from neurodags.loaders import load_pipeline_configuration
from neurodags.migrate import (
    MigrationRefused,
    apply_migration,
    legacy_column_to_new,
    plan_migration,
    read_journal,
)
from neurodags.outputs import manifest_artifacts, unit_state
from neurodags.utils import find_unique_root

# ---------------------------------------------------------------------------
# Building 0.3.x projects
# ---------------------------------------------------------------------------

SIMPLE = {"nodes": [{"id": 0, "derivative": "SourceFile"}]}


def _sources(root: Path, rels) -> list[str]:
    out = []
    for rel in rels:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"source {rel}")
        out.append(str(path))
    return out


def _legacy_write(
    derivs: Path, old_root: str, source: str, name: str, tail: str, content: str | None = None
) -> Path:
    """Write one output exactly where 0.3.x put it (root from find_unique_root, CamelCase)."""
    rel = os.path.relpath(source, old_root)
    base = derivs / rel
    base = base.parent / base.name.replace("@", "&")
    path = Path(f"{base}@{snake_to_camel(name)}{tail}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content if content is not None else f"{name}{tail} of {rel}")
    return path


def _legacy_tree(
    derivs: Path, sources: list[str], outputs: dict[str, list[str]], root: str | None = None
) -> list[Path]:
    """0.3.x outputs for every source: ``{name: [tails]}`` (root over the given sources)."""
    root = root or find_unique_root(sources, mode="maximal")
    written = []
    for source in sources:
        for name, tails in outputs.items():
            for tail in tails:
                written.append(_legacy_write(derivs, root, source, name, tail))
    return written


PIPELINE_TEXT = """\
# Study pipeline (0.3.x)
datasets: ds.yml   # the datasets
mount_point: null

DerivativeDefinitions:
  # spectral features
  band_power:
    for_dataframe: true
    nodes:
      - {id: 0, derivative: SourceFile}
  Ep_HiguchiFD:
    for_dataframe: true
    nodes:
      - id: 0
        derivative: band_power.nc
  Clean:
    nodes:
      - {id: 0, derivative: SourceFile}

DerivativeList:
  - band_power
  - Ep_HiguchiFD
  - Clean
"""

DATASETS_TEXT = """\
# datasets of the study
study:
  name: Study A
  file_pattern: raw/**/*.txt   # every recording
  derivatives_path: derivatives
"""


def _project(tmp_path: Path, pipeline: str = PIPELINE_TEXT, datasets: str = DATASETS_TEXT) -> Path:
    (tmp_path / "pl.yml").write_text(pipeline)
    (tmp_path / "ds.yml").write_text(datasets)
    return tmp_path / "pl.yml"


RELS = [
    "sub-01/ses-1/eeg/sub-01_ses-1_eeg.txt",
    "sub-02/ses-1/eeg/sub-02_ses-1_eeg.txt",
    "sub-03/ses-1/eeg/sub-03_ses-1_eeg.txt",
]


def _snapshot(root: Path) -> dict[str, bytes]:
    return {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def _plan(pipe: Path, tmp_path: Path, **kwargs):
    kwargs.setdefault("legacy_cwd", tmp_path)
    return plan_migration(pipe, **kwargs)


@pytest.fixture
def full_project(tmp_path):
    pipe = _project(tmp_path)
    sources = _sources(tmp_path / "raw", RELS)
    _legacy_tree(
        tmp_path / "derivatives",
        sources,
        {"band_power": [".nc"], "Ep_HiguchiFD": [".nc"], "Clean": [".fif"]},
    )
    return pipe, sources


# ---------------------------------------------------------------------------
# Preview and apply
# ---------------------------------------------------------------------------


def test_preview_changes_nothing(full_project, tmp_path, capsys):
    pipe, _ = full_project
    before = _snapshot(tmp_path)

    assert main(["migrate", str(pipe), "--legacy-cwd", str(tmp_path)]) == 0
    out = capsys.readouterr().out

    assert _snapshot(tmp_path) == before
    assert "preview: nothing is changed" in out
    assert "@BandPower -> @band_power" in out
    assert "@EpHiguchifd -> @Ep_HiguchiFD" in out
    assert "6 file move(s)" in out
    assert "3 already in place" in out  # Clean: same token and name
    assert f"legacy cwd:  {tmp_path}" in out


def test_apply_full_tree_moves_renames_and_manifests(full_project, tmp_path, capsys):
    pipe, _ = full_project
    plan = _plan(pipe, tmp_path)
    assert plan.refusals == []
    planned = list(plan.moves)
    assert len(planned) == 6  # Clean needs no move (no "_": same spelling)
    # The 0.3.x root was raw/ (common to the 3 subjects): folders are unchanged, names verbatim.
    assert (
        str(tmp_path / "derivatives/sub-01/ses-1/eeg/sub-01_ses-1_eeg.txt@EpHiguchifd.nc"),
        str(tmp_path / "derivatives/sub-01/ses-1/eeg/sub-01_ses-1_eeg.txt@Ep_HiguchiFD.nc"),
    ) in planned

    result = apply_migration(plan)

    assert sorted(result.moved) == sorted(planned)
    for old, new in planned:
        assert not os.path.exists(old)
        assert os.path.isfile(new)
    for rel in RELS:
        base = tmp_path / "derivatives" / rel
        for name, suffix in (("band_power", ".nc"), ("Ep_HiguchiFD", ".nc"), ("Clean", ".fif")):
            state = unit_state(f"{base}@{name}", name)
            assert state.state == "done"
            assert state.origin == "migrate"
            assert list(state.artifacts) == [suffix]
            manifest = json.loads(Path(f"{base}@{name}.done").read_text())
            assert manifest["source"] == {"dataset": "study", "rel": rel}

    # Configuration rewritten in place, comments kept, backups written.
    assert (tmp_path / "pl.yml.bak-0.3").read_text() == PIPELINE_TEXT
    assert (tmp_path / "ds.yml.bak-0.3").read_text() == DATASETS_TEXT
    new_pipeline = (tmp_path / "pl.yml").read_text()
    assert new_pipeline.startswith("# Study pipeline (0.3.x)\nnaming_version: 1\n")
    assert "# spectral features" in new_pipeline
    new_datasets = (tmp_path / "ds.yml").read_text()
    assert "  source_root: raw\n  file_pattern: raw/**/*.txt   # every recording" in new_datasets
    config = load_pipeline_configuration(str(pipe))
    assert config["naming_version"] == 1

    # Idempotent: a second run (now in tree-only mode) has nothing to do.
    again = _plan(pipe, tmp_path)
    assert again.mode == "tree-only"
    assert again.refusals == []
    assert not again.has_work
    snapshot = _snapshot(tmp_path)
    assert main(["migrate", str(pipe), "--apply"]) == 0
    assert "Nothing to migrate" in capsys.readouterr().out
    assert _snapshot(tmp_path) == snapshot


def test_status_shows_done_migrated(full_project, tmp_path, capsys):
    pipe, _ = full_project
    apply_migration(_plan(pipe, tmp_path))

    assert main(["status", str(pipe), "--format", "json", "--include-files"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["grand_total"]["done"] == 9
    assert data["grand_total"]["migrated"] == 9
    assert data["derivatives"]["band_power"]["migrated"] == 3
    assert {f["label"] for f in data["files"]} == {"done (migrated)"}
    assert {f["status"] for f in data["files"]} == {"done"}

    assert main(["status", str(pipe)]) == 0
    out = capsys.readouterr().out
    assert "(migrated)" in out
    assert "done (migrated)" in out


def test_pilot_era_flat_tree_moves_into_subject_folders(tmp_path):
    """A single-subject 0.3.x run rooted at the subject's folder: outputs at the tree's top."""
    pipe = _project(tmp_path)
    sources = _sources(tmp_path / "raw", RELS)
    pilot = _legacy_tree(tmp_path / "derivatives", sources[:1], {"band_power": [".nc"]})
    assert pilot[0] == tmp_path / "derivatives/sub-01_ses-1_eeg.txt@BandPower.nc"

    plan = _plan(pipe, tmp_path)
    assert plan.moves == [
        (
            str(pilot[0]),
            str(tmp_path / "derivatives/sub-01/ses-1/eeg/sub-01_ses-1_eeg.txt@band_power.nc"),
        )
    ]
    apply_migration(plan)
    new = tmp_path / "derivatives/sub-01/ses-1/eeg/sub-01_ses-1_eeg.txt@band_power"
    assert unit_state(str(new), "band_power").state == "done"


def test_pre_033_single_file_form(tmp_path):
    """Before 0.3.3 a single-file dataset wrote ``<derivatives_path>@Deriv.*`` beside it."""
    pipe = _project(tmp_path)
    _sources(tmp_path / "raw", RELS[:1])
    old = tmp_path / "derivatives@BandPower.nc"
    old.write_text("pilot")

    plan = _plan(pipe, tmp_path)
    target = tmp_path / "derivatives/sub-01/ses-1/eeg/sub-01_ses-1_eeg.txt@band_power.nc"
    assert plan.moves == [(str(old), str(target))]
    apply_migration(plan)
    assert target.read_text() == "pilot"


def test_pre_033_single_file_form_with_several_sources_is_ambiguous(tmp_path):
    pipe = _project(tmp_path)
    _sources(tmp_path / "raw", RELS)
    (tmp_path / "derivatives@BandPower.nc").write_text("pilot")

    plan = _plan(pipe, tmp_path)
    assert len(plan.units("ambiguous")) == 1
    assert any("ambiguous" in r for r in plan.refusals)


def test_markers_move_with_their_unit(tmp_path):
    pipe = _project(tmp_path)
    sources = _sources(tmp_path / "raw", RELS[:2])
    root = find_unique_root(sources, mode="maximal")
    _legacy_write(tmp_path / "derivatives", root, sources[0], "Clean", ".error", "boom")
    _legacy_write(tmp_path / "derivatives", root, sources[1], "Clean", ".skip", "n/a")
    _legacy_write(tmp_path / "derivatives", root, sources[1], "band_power", ".nc")
    _legacy_write(tmp_path / "derivatives", root, sources[1], "band_power", ".error", "old")

    apply_migration(_plan(pipe, tmp_path))

    b0 = tmp_path / "derivatives" / RELS[0]
    b1 = tmp_path / "derivatives" / RELS[1]
    assert Path(f"{b0}@Clean.error").read_text() == "boom"
    assert unit_state(f"{b0}@Clean", "Clean").state == "error"
    assert unit_state(f"{b1}@Clean", "Clean").state == "skipped"
    # A unit with an .error marker gets no .done, even with its artifacts present.
    assert not Path(f"{b1}@band_power.done").exists()
    assert unit_state(f"{b1}@band_power", "band_power").state == "error"


def test_partial_multi_artifact_unit_gets_no_manifest(tmp_path):
    pipe = _project(tmp_path)
    sources = _sources(tmp_path / "raw", RELS)
    root = find_unique_root(sources, mode="maximal")
    for i, source in enumerate(sources):
        _legacy_write(tmp_path / "derivatives", root, source, "band_power", ".nc")
        if i < 2:
            _legacy_write(tmp_path / "derivatives", root, source, "band_power", ".report.json")

    plan = _plan(pipe, tmp_path)
    assert plan.artifact_sets["band_power"]["known"] == [".nc", ".report.json"]
    apply_migration(plan)

    states = [
        unit_state(str(tmp_path / "derivatives" / f"{rel}@band_power"), "band_power")
        for rel in RELS
    ]
    assert [s.state for s in states] == ["done", "done", "incomplete"]
    assert set(states[0].artifacts) == {".nc", ".report.json"}


def test_tie_between_artifact_sets_synthesizes_nothing(tmp_path):
    pipe = _project(tmp_path)
    sources = _sources(tmp_path / "raw", RELS[:2])
    root = find_unique_root(sources, mode="maximal")
    _legacy_write(tmp_path / "derivatives", root, sources[0], "band_power", ".nc")
    _legacy_write(tmp_path / "derivatives", root, sources[1], "band_power", ".json")

    plan = _plan(pipe, tmp_path)
    assert plan.artifact_sets["band_power"]["known"] is None
    assert plan.manifests == []


def test_no_synthesize_moves_without_manifests(full_project, tmp_path):
    pipe, _ = full_project
    plan = _plan(pipe, tmp_path, synthesize=False)
    apply_migration(plan)
    base = tmp_path / "derivatives" / RELS[0]
    assert unit_state(f"{base}@band_power", "band_power").state == "incomplete"


def test_orphans_are_left_in_place(full_project, tmp_path, capsys):
    pipe, sources = full_project
    os.unlink(sources[2])  # source deleted since the 0.3.x run

    plan = _plan(pipe, tmp_path)
    orphan = tmp_path / "derivatives" / f"{RELS[2]}@BandPower.nc"
    assert str(orphan) in plan.to_dict()["orphans"]
    apply_migration(plan)
    assert orphan.exists()
    assert main(["migrate", str(pipe)]) == 0
    assert "orphans (no matching source; left in place)" in capsys.readouterr().out


def test_unrecognized_files_are_reported_not_moved(full_project, tmp_path):
    pipe, _ = full_project
    stray = tmp_path / "derivatives" / "notes@Unknown.txt"
    stray.write_text("x")
    plan = _plan(pipe, tmp_path)
    assert str(stray) in plan.to_dict()["unrecognized"]
    apply_migration(plan)
    assert stray.exists()


def test_code_and_neurodags_folders_are_not_scanned(full_project, tmp_path):
    pipe, _ = full_project
    snap = tmp_path / "derivatives/code/x@BandPower.nc"
    snap.parent.mkdir(parents=True)
    snap.write_text("snapshot")
    plan = _plan(pipe, tmp_path)
    assert all(str(snap) not in u.files for u in plan.units())


# ---------------------------------------------------------------------------
# Chained pipelines and '&' source names
# ---------------------------------------------------------------------------

CHAINED_DATASETS = """\
stage2:
  name: Stage 2
  file_pattern: upstream/**/*@CleanPrep.fif
  derivatives_path: derivatives
"""


def test_chained_source_names_with_at(tmp_path, capsys):
    """Sources that are themselves outputs ('@' in the name) were written with '&' by 0.3.x;
    they are matched by that legacy key and moved to the verbatim (multi-'@') name."""
    pipe = _project(tmp_path, datasets=CHAINED_DATASETS)
    sources = _sources(
        tmp_path / "upstream", ["sub-01/x.vhdr@CleanPrep.fif", "sub-02/y.vhdr@CleanPrep.fif"]
    )
    outputs = _legacy_tree(tmp_path / "derivatives", sources, {"band_power": [".nc"]})
    assert outputs[0].name == "x.vhdr&CleanPrep.fif@BandPower.nc"  # authentic 0.3.x name

    plan = _plan(pipe, tmp_path)
    assert plan.refusals == []
    assert [u.status for u in plan.units()] == ["move", "move"]
    assert [os.path.basename(n) for _, n in plan.moves] == [
        "x.vhdr@CleanPrep.fif@band_power.nc",
        "y.vhdr@CleanPrep.fif@band_power.nc",
    ]
    # Patterns referencing 0.3.x output names are printed, never edited.
    assert any("*@CleanPrep.fif" in p for p in plan.chained_patterns)
    apply_migration(plan)
    assert "*@CleanPrep.fif" in (tmp_path / "ds.yml").read_text()
    prefix = tmp_path / "derivatives/sub-01/x.vhdr@CleanPrep.fif@band_power"
    manifest = json.loads(Path(f"{prefix}.done").read_text())
    assert manifest["source"]["rel"] == "sub-01/x.vhdr@CleanPrep.fif"
    assert manifest["artifacts"] == {".nc": ["x.vhdr@CleanPrep.fif@band_power.nc"]}
    state = unit_state(str(prefix), "band_power")
    assert (state.state, state.origin) == ("done", "migrate")
    assert not outputs[0].exists()
    assert not _plan(pipe, tmp_path).has_work


def test_literal_ampersand_sources_keep_their_ampersand(tmp_path):
    """'&' is never decoded: a source really named with '&' matches its own key."""
    datasets = CHAINED_DATASETS.replace("*@CleanPrep.fif", "*.fif")
    pipe = _project(tmp_path, datasets=datasets)
    sources = _sources(tmp_path / "upstream", ["sub-01/x&A.fif", "sub-02/x&literal@CleanPrep.fif"])
    outputs = _legacy_tree(tmp_path / "derivatives", sources, {"band_power": [".nc"]})
    assert [o.name for o in outputs] == [
        "x&A.fif@BandPower.nc",
        "x&literal&CleanPrep.fif@BandPower.nc",
    ]
    plan = _plan(pipe, tmp_path)
    assert plan.refusals == []
    assert [os.path.basename(n) for _, n in plan.moves] == [
        "x&A.fif@band_power.nc",
        "x&literal@CleanPrep.fif@band_power.nc",
    ]
    apply_migration(plan)
    prefix = tmp_path / "derivatives/sub-01/x&A.fif@band_power"
    assert unit_state(str(prefix), "band_power").state == "done"


def test_at_and_ampersand_sources_sharing_a_legacy_name_are_ambiguous(tmp_path, capsys):
    """0.3.x wrote x@A.fif and x&A.fif to the same outputs: ownership cannot be told."""
    datasets = CHAINED_DATASETS.replace("*@CleanPrep.fif", "*.fif")
    pipe = _project(tmp_path, datasets=datasets)
    sources = _sources(tmp_path / "upstream", ["sub-01/x@A.fif", "sub-01/x&A.fif"])
    _legacy_tree(tmp_path / "derivatives", sources[:1], {"band_power": [".nc"]})
    plan = _plan(pipe, tmp_path)
    assert [u.status for u in plan.units()] == ["ambiguous"]
    assert any("ambiguous" in r for r in plan.refusals)
    with pytest.raises(MigrationRefused):
        apply_migration(plan)
    out = _refused_cli(pipe, tmp_path, capsys)
    assert "ambiguous" in out


def test_unresolved_chained_outputs_are_refused(tmp_path, capsys):
    """A 0.3.x chained output whose upstream source is not discovered is not silently
    left as an orphan (it would be recomputed): the plan is refused."""
    pipe = _project(tmp_path, datasets=CHAINED_DATASETS)
    _sources(tmp_path / "upstream", ["sub-01/x.vhdr@CleanPrep.fif"])
    gone = [str(tmp_path / "upstream/sub-02/y.vhdr@CleanPrep.fif")]
    up = str(tmp_path / "upstream")
    _legacy_tree(tmp_path / "derivatives", gone, {"band_power": [".nc"]}, root=up)
    # A plain unmatched output (no '&<Name><suffix>' source part) stays an orphan.
    (tmp_path / "derivatives/sub-02/stray.txt@BandPower.nc").write_text("x")

    plan = _plan(pipe, tmp_path)
    statuses = sorted(u.status for u in plan.units())
    assert statuses == ["orphan", "unresolved"]
    assert any("chained pipeline match no discovered source" in r for r in plan.refusals)
    assert plan.to_dict()["unresolved"] == [
        str(tmp_path / "derivatives/sub-02/y.vhdr&CleanPrep.fif@BandPower.nc")
    ]
    preview = migrate.format_preview(plan)
    assert "1 unresolved" in preview
    assert "unresolved chained outputs" in preview
    out = _refused_cli(pipe, tmp_path, capsys)
    assert "unresolved" in out


def test_chained_dataset_with_zero_discovered_sources_is_refused(tmp_path, capsys):
    """After an upstream rename every chained source can vanish at once: the output tree
    is still scanned, its chained outputs are unresolved and nothing is applied."""
    pipe = _project(tmp_path, datasets=CHAINED_DATASETS)
    (tmp_path / "upstream").mkdir()  # pattern matches nothing
    gone = [str(tmp_path / "upstream/sub-01/x.vhdr@CleanPrep.fif")]
    up = str(tmp_path / "upstream")
    legacy = _legacy_tree(tmp_path / "derivatives", gone, {"band_power": [".nc"]}, root=up)

    plan = _plan(pipe, tmp_path)
    assert [u.status for u in plan.units()] == ["unresolved"]
    assert plan.to_dict()["unresolved"] == [str(legacy[0])]
    assert any("chained pipeline match no discovered source" in r for r in plan.refusals)
    with pytest.raises(MigrationRefused):
        apply_migration(plan)
    out = _refused_cli(pipe, tmp_path, capsys)  # YAML and outputs byte-identical
    assert "unresolved" in out
    assert "naming_version" not in (tmp_path / "pl.yml").read_text()


def test_outputs_next_to_undiscovered_sources_are_reported(tmp_path, capsys):
    """Datasets without derivatives_path: folders with no discovered source are scanned
    too, so a chained output there is unresolved instead of silently left behind."""
    datasets = CHAINED_DATASETS.replace("  derivatives_path: derivatives\n", "")
    pipe = _project(tmp_path, datasets=datasets)
    kept = _sources(tmp_path / "upstream", ["sub-01/x.vhdr@CleanPrep.fif"])
    Path(f"{tmp_path}/upstream/sub-01/x.vhdr&CleanPrep.fif@BandPower.nc").write_text("x")
    stale = tmp_path / "upstream/sub-02/y.vhdr&CleanPrep.fif@BandPower.nc"
    stale.parent.mkdir(parents=True)
    stale.write_text("y")

    plan = _plan(pipe, tmp_path)
    assert sorted(u.status for u in plan.units()) == ["move", "unresolved"]
    assert [os.path.basename(n) for _, n in plan.moves] == ["x.vhdr@CleanPrep.fif@band_power.nc"]
    assert plan.to_dict()["unresolved"] == [str(stale)]
    _refused_cli(pipe, tmp_path, capsys)

    stale.unlink()  # recompute it on purpose
    plan = _plan(pipe, tmp_path)
    assert plan.refusals == []
    apply_migration(plan)
    assert unit_state(f"{kept[0]}@band_power", "band_power").state == "done"
    assert not _plan(pipe, tmp_path).has_work


def test_chained_unit_moved_before_its_manifest_is_completed_on_resume(tmp_path, monkeypatch):
    """Crash after a move but before .done: the moved file has the 0.4.0 name (two '@',
    rejected by the 0.3.x parser); it is recognized as a planned target and the unit
    completes. A further run is a no-op."""
    pipe = _project(tmp_path, datasets=CHAINED_DATASETS)
    sources = _sources(tmp_path / "upstream", ["sub-01/x.vhdr@CleanPrep.fif"])
    up = str(tmp_path / "upstream")
    _legacy_tree(tmp_path / "derivatives", sources, {"band_power": [".nc", ".json"]}, root=up)
    folder = tmp_path / "derivatives/sub-01"
    prefix = folder / "x.vhdr@CleanPrep.fif@band_power"

    plan = _plan(pipe, tmp_path)
    assert len(plan.moves) == 2
    real_replace = os.replace
    calls = {"n": 0}

    def flaky_replace(src, dst):
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("disk went away")
        return real_replace(src, dst)

    monkeypatch.setattr(migrate.os, "replace", flaky_replace)
    with pytest.raises(OSError, match="disk went away"):
        apply_migration(plan)
    monkeypatch.setattr(migrate.os, "replace", real_replace)
    # Half moved: one file at the new name, one at the 0.3.x name, no .done.
    names = sorted(p.name for p in folder.iterdir())
    assert len([n for n in names if n.startswith("x.vhdr@CleanPrep.fif@band_power.")]) == 1
    assert len([n for n in names if n.startswith("x.vhdr&CleanPrep.fif@BandPower.")]) == 1
    assert not Path(f"{prefix}.done").exists()

    resumed = _plan(pipe, tmp_path)
    assert resumed.refusals == []
    assert [t.unrecognized for t in resumed.trees] == [[]]
    assert len(resumed.moves) == 1
    assert len(resumed.manifests) == 1
    apply_migration(resumed)
    state = unit_state(str(prefix), "band_power")
    assert (state.state, state.origin) == ("done", "migrate")
    assert set(state.artifacts) == {".nc", ".json"}
    assert sorted(p.name for p in folder.iterdir()) == [
        "x.vhdr@CleanPrep.fif@band_power.done",
        "x.vhdr@CleanPrep.fif@band_power.json",
        "x.vhdr@CleanPrep.fif@band_power.nc",
    ]
    again = _plan(pipe, tmp_path)
    assert again.refusals == []
    assert not again.has_work


def test_chained_unit_fully_moved_without_manifest_gets_its_manifest(tmp_path, monkeypatch):
    pipe = _project(tmp_path, datasets=CHAINED_DATASETS)
    sources = _sources(tmp_path / "upstream", ["sub-01/x.vhdr@CleanPrep.fif"])
    up = str(tmp_path / "upstream")
    outputs = {"band_power": [".nc"], "Clean": [".error"]}
    _legacy_tree(tmp_path / "derivatives", sources, outputs, root=up)
    plan = _plan(pipe, tmp_path)

    def crash(*args, **kwargs):
        raise OSError("crash before the manifest")

    monkeypatch.setattr(migrate, "write_manifest", crash)
    with pytest.raises(OSError, match="crash before the manifest"):
        apply_migration(plan)
    monkeypatch.undo()
    folder = tmp_path / "derivatives/sub-01"
    # The crash hit the first unit's manifest; the second unit was not reached.
    assert sorted(p.name for p in folder.iterdir()) == [
        "x.vhdr&CleanPrep.fif@Clean.error",
        "x.vhdr@CleanPrep.fif@band_power.nc",
    ]
    resumed = _plan(pipe, tmp_path)
    assert resumed.refusals == []
    assert [os.path.basename(n) for _, n in resumed.moves] == ["x.vhdr@CleanPrep.fif@Clean.error"]
    assert sorted(u.status for u in resumed.units()) == ["identity", "move"]
    assert len(resumed.manifests) == 1
    apply_migration(resumed)
    done = unit_state(str(folder / "x.vhdr@CleanPrep.fif@band_power"), "band_power")
    assert done.state == "done"
    assert unit_state(str(folder / "x.vhdr@CleanPrep.fif@Clean"), "Clean").state == "error"
    assert not _plan(pipe, tmp_path).has_work


def test_upstream_map_matches_renamed_upstream_sources(tmp_path):
    """After the upstream pipeline renamed @CleanPrep to @clean_prep (and the user edited the
    downstream pattern), --upstream-map lets the downstream outputs follow."""
    datasets = CHAINED_DATASETS.replace("*@CleanPrep.fif", "*@clean_prep.fif")
    pipe = _project(tmp_path, datasets=datasets)
    old_sources = [str(tmp_path / "upstream/sub-01/x.vhdr@CleanPrep.fif")]
    _legacy_tree(tmp_path / "derivatives", old_sources, {"band_power": [".nc"]})
    _sources(tmp_path / "upstream", ["sub-01/x.vhdr@clean_prep.fif"])

    without = _plan(pipe, tmp_path)
    assert without.moves == []
    assert len(without.units("unresolved")) == 1  # refused, not silently recomputed
    assert without.refusals

    plan = _plan(pipe, tmp_path, upstream_renames=["CleanPrep=clean_prep"])
    assert plan.refusals == []
    assert [os.path.basename(n) for _, n in plan.moves] == ["x.vhdr@clean_prep.fif@band_power.nc"]
    apply_migration(plan)
    prefix = tmp_path / "derivatives/sub-01/x.vhdr@clean_prep.fif@band_power"
    assert unit_state(str(prefix), "band_power").state == "done"


def test_other_yaml_files_referencing_old_tokens_are_printed(full_project, tmp_path):
    pipe, _ = full_project
    other = tmp_path / "stage2.yml"
    other.write_text("d:\n  file_pattern: derivatives/**/*@BandPower.nc\n")
    plan = _plan(pipe, tmp_path)
    assert any("stage2.yml:2" in p and "@band_power" in p for p in plan.chained_patterns)
    apply_migration(plan)
    assert other.read_text() == "d:\n  file_pattern: derivatives/**/*@BandPower.nc\n"


# ---------------------------------------------------------------------------
# MNE split recordings
# ---------------------------------------------------------------------------


def _split_raw(path: Path):
    mne = pytest.importorskip("mne")
    np = pytest.importorskip("numpy")
    info = mne.create_info(4, 1000.0, "eeg")
    raw = mne.io.RawArray(
        np.random.default_rng(0).standard_normal((4, 200_000)) * 1e-5, info, verbose="error"
    )
    raw.set_annotations(mne.Annotations([1.0], [0.5], ["blink"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    written = raw.save(str(path), split_size="2MB", verbose="error", overwrite=True)
    parts = [str(p) for p in written] if written else [str(path)]
    assert len(parts) > 1
    return mne.io.read_raw_fif(parts[0], preload=True, verbose="error"), parts


def test_mne_split_recording_is_rewritten_and_verified(tmp_path):
    mne = pytest.importorskip("mne")
    np = pytest.importorskip("numpy")
    pipe = _project(tmp_path)
    sources = _sources(tmp_path / "raw", RELS[:2])
    root = find_unique_root(sources, mode="maximal")
    old_main = tmp_path / "derivatives" / f"{RELS[0]}@BandPower.fif"
    original, old_parts = _split_raw(old_main)
    _legacy_write(tmp_path / "derivatives", root, sources[1], "band_power", ".fif", "small")

    plan = _plan(pipe, tmp_path)
    assert [r.old_parts for r in plan.mne_rewrites] == [old_parts]
    result = apply_migration(plan)

    assert all(not os.path.exists(p) for p in old_parts)
    prefix = str(tmp_path / "derivatives" / f"{RELS[0]}@band_power")
    artifacts = manifest_artifacts(prefix, "band_power")
    assert artifacts is not None
    assert len(artifacts[".fif"]) > 1
    assert artifacts[".fif"][0] == prefix + ".fif"
    assert result.rewritten[0]["new_parts"] == artifacts[".fif"]
    reloaded = mne.io.read_raw_fif(prefix + ".fif", preload=True, verbose="error")
    assert np.array_equal(reloaded.get_data(), original.get_data())
    assert list(reloaded.annotations.description) == ["blink"]
    # The plain single-file .fif of the other unit was simply renamed.
    assert Path(str(tmp_path / "derivatives" / f"{RELS[1]}@band_power.fif")).read_text() == "small"


def test_mne_rewrite_failure_leaves_unit_in_place(tmp_path, monkeypatch):
    pytest.importorskip("mne")
    pipe = _project(tmp_path)
    _sources(tmp_path / "raw", RELS[:1])
    old_main = tmp_path / "derivatives" / f"{RELS[0]}@BandPower.fif"
    _, old_parts = _split_raw(old_main)
    monkeypatch.setattr(migrate, "_same_recording", lambda kind, a, b: "data differ")

    result = apply_migration(_plan(pipe, tmp_path))

    assert all(os.path.exists(p) for p in old_parts)
    prefix = str(tmp_path / "derivatives" / f"{RELS[0]}@band_power")
    assert not os.path.exists(prefix + ".fif")
    assert not os.path.exists(prefix + "-1.fif")
    assert unit_state(prefix, "band_power").state == "missing"
    assert len(result.recompute) == 1
    assert "data differ" in result.recompute[0]["reason"]


def test_mne_rewrite_resumes_after_verification(tmp_path):
    """Interrupted after the new files were verified: the next run only deletes the old parts."""
    pytest.importorskip("mne")
    pipe = _project(tmp_path)
    _sources(tmp_path / "raw", RELS[:1])
    old_main = tmp_path / "derivatives" / f"{RELS[0]}@BandPower.fif"
    _, old_parts = _split_raw(old_main)
    prefix = str(tmp_path / "derivatives" / f"{RELS[0]}@band_power")
    plan = _plan(pipe, tmp_path)
    rewrite = plan.mne_rewrites[0]
    new_parts, problem = migrate._rewrite_mne(rewrite)
    assert problem is None
    journal = tmp_path / "derivatives/.neurodags/migrate-journal.json"
    journal.parent.mkdir(parents=True)
    journal.write_text(
        json.dumps(
            {
                "op": "mne-verified",
                "old_parts": old_parts,
                "new": prefix + ".fif",
                "new_parts": new_parts,
            }
        )
        + "\n"
    )

    plan = _plan(pipe, tmp_path)
    assert plan.refusals == []
    assert plan.mne_rewrites[0].finished_parts == new_parts
    apply_migration(plan)
    assert all(not os.path.exists(p) for p in old_parts)
    assert manifest_artifacts(prefix, "band_power") == {".fif": new_parts}


def test_split_recording_with_unchanged_name_is_moved_as_is(tmp_path):
    """Only the folder changes (pilot-era tree, name without '_'): parts reference each other
    by base name, so they are moved without rewriting."""
    mne = pytest.importorskip("mne")
    pipe = _project(tmp_path)
    _sources(tmp_path / "raw", RELS)
    old_main = tmp_path / "derivatives" / f"{os.path.basename(RELS[0])}@Clean.fif"
    original, old_parts = _split_raw(old_main)

    plan = _plan(pipe, tmp_path)
    assert plan.mne_rewrites == []
    assert len(plan.moves) == len(old_parts)
    apply_migration(plan)
    prefix = str(tmp_path / "derivatives" / f"{RELS[0]}@Clean")
    artifacts = manifest_artifacts(prefix, "Clean")
    assert artifacts is not None
    assert len(artifacts[".fif"]) == len(old_parts)
    reloaded = mne.io.read_raw_fif(prefix + ".fif", preload=True, verbose="error")
    assert reloaded.n_times == original.n_times


# ---------------------------------------------------------------------------
# Journal and interrupted runs
# ---------------------------------------------------------------------------


def test_interrupted_apply_resumes(full_project, tmp_path, monkeypatch):
    pipe, _ = full_project
    plan = _plan(pipe, tmp_path)
    real_replace = os.replace
    calls = {"n": 0}

    def flaky_replace(src, dst):
        if os.fspath(dst).endswith(".done"):  # manifest publication, not a move
            return real_replace(src, dst)
        calls["n"] += 1
        if calls["n"] == 4:
            raise OSError("disk went away")
        return real_replace(src, dst)

    monkeypatch.setattr(migrate.os, "replace", flaky_replace)
    with pytest.raises(OSError, match="disk went away"):
        apply_migration(plan)
    monkeypatch.setattr(migrate.os, "replace", real_replace)

    # Configuration untouched (written only after every move succeeded).
    assert (tmp_path / "pl.yml").read_text() == PIPELINE_TEXT
    assert not (tmp_path / "pl.yml.bak-0.3").exists()
    journal = tmp_path / "derivatives/.neurodags/migrate-journal.json"
    assert [e["op"] for e in read_journal(str(journal))].count("move") == 3

    resumed = _plan(pipe, tmp_path)
    assert resumed.refusals == []
    assert len(resumed.moves) == 3
    apply_migration(resumed)
    for rel in RELS:
        for name in ("band_power", "Ep_HiguchiFD", "Clean"):
            prefix = str(tmp_path / "derivatives" / f"{rel}@{name}")
            assert unit_state(prefix, name).state == "done"
    assert "naming_version: 1" in (tmp_path / "pl.yml").read_text()
    assert not _plan(pipe, tmp_path).has_work


@pytest.mark.parametrize("renames", [[], ["band_power=BandPower"]])
def test_interrupted_between_configuration_files_resumes(
    full_project, tmp_path, monkeypatch, renames
):
    """The datasets YAML is published before the pipeline (which declares naming_version);
    a crash in between is completed by running the same command again."""
    pipe, _ = full_project
    plan = _plan(pipe, tmp_path, renames=renames)
    real_write = migrate._write_config
    order = []

    def crash_on_pipeline(change):
        order.append(change.role)
        if change.role != "datasets":
            raise OSError("crash before the pipeline is written")
        real_write(change)

    monkeypatch.setattr(migrate, "_write_config", crash_on_pipeline)
    with pytest.raises(OSError, match="crash before the pipeline"):
        apply_migration(plan)
    monkeypatch.setattr(migrate, "_write_config", real_write)
    assert order == ["datasets", "pipeline"]
    assert "source_root" in (tmp_path / "ds.yml").read_text()
    assert (tmp_path / "pl.yml").read_text() == PIPELINE_TEXT

    resumed = _plan(pipe, tmp_path, renames=renames)
    assert resumed.refusals == []
    assert [c.role for c in resumed.config_changes] == ["pipeline"]
    apply_migration(resumed)
    config = load_pipeline_configuration(str(pipe))
    assert config["naming_version"] == 1
    new_name = "BandPower" if renames else "band_power"
    assert new_name in config["DerivativeDefinitions"]
    for rel in RELS:
        prefix = str(tmp_path / "derivatives" / f"{rel}@{new_name}")
        assert unit_state(prefix, new_name).state == "done"
    again = _plan(pipe, tmp_path, renames=renames)
    assert again.refusals == []
    assert not again.has_work


def test_unit_half_moved_by_an_interrupted_run_is_completed(tmp_path):
    """One artifact already renamed, the other still 0.3.x: one unit, one manifest."""
    pipe = _project(tmp_path)
    sources = _sources(tmp_path / "raw", RELS)
    _legacy_tree(tmp_path / "derivatives", sources, {"band_power": [".nc", ".json"]})
    base = tmp_path / "derivatives" / RELS[0]
    os.replace(f"{base}@BandPower.nc", f"{base}@band_power.nc")

    plan = _plan(pipe, tmp_path)
    assert plan.refusals == []
    apply_migration(plan)
    state = unit_state(f"{base}@band_power", "band_power")
    assert state.state == "done"
    assert set(state.artifacts) == {".nc", ".json"}


def test_mne_rewrite_started_but_not_verified_is_redone(tmp_path):
    pytest.importorskip("mne")
    pipe = _project(tmp_path)
    _sources(tmp_path / "raw", RELS[:1])
    old_main = tmp_path / "derivatives" / f"{RELS[0]}@BandPower.fif"
    _, old_parts = _split_raw(old_main)
    prefix = str(tmp_path / "derivatives" / f"{RELS[0]}@band_power")
    Path(prefix + ".fif").write_text("half written")
    assert _plan(pipe, tmp_path).refusals  # not ours (no journal): refused

    journal = tmp_path / "derivatives/.neurodags/migrate-journal.json"
    journal.parent.mkdir(parents=True)
    entry = {"op": "mne-start", "old_parts": old_parts, "new": prefix + ".fif"}
    journal.write_text(json.dumps(entry) + "\n")
    plan = _plan(pipe, tmp_path)
    assert plan.refusals == []
    apply_migration(plan)
    assert unit_state(prefix, "band_power").state == "done"
    assert not any(os.path.exists(p) for p in old_parts)


# ---------------------------------------------------------------------------
# Refusals (nothing applied)
# ---------------------------------------------------------------------------


def _refused_cli(pipe: Path, tmp_path: Path, capsys, *extra: str) -> str:
    before = _snapshot(tmp_path)
    code = main(["migrate", str(pipe), "--apply", "--legacy-cwd", str(tmp_path), *extra])
    captured = capsys.readouterr()
    assert code == 2
    assert "migration refused" in captured.err
    assert _snapshot(tmp_path) == before
    return captured.out


def test_ambiguous_mapping_is_refused(tmp_path, capsys):
    pipe = _project(tmp_path)
    sources = _sources(tmp_path / "raw", ["sub-01/eeg/rec.txt", "sub-02/eeg/rec.txt"])
    # Written with a root at sub-01/eeg (pilot): 'rec.txt' alone matches both sources.
    _legacy_write(
        tmp_path / "derivatives", os.path.dirname(sources[0]), sources[0], "Clean", ".fif"
    )
    out = _refused_cli(pipe, tmp_path, capsys)
    assert "ambiguous" in out


def test_existing_target_with_different_content_is_refused(full_project, tmp_path, capsys):
    pipe, _ = full_project
    target = tmp_path / "derivatives" / f"{RELS[0]}@band_power.nc"
    target.write_text("something else")
    out = _refused_cli(pipe, tmp_path, capsys)
    assert "different content" in out


def test_identical_existing_target_is_a_duplicate(full_project, tmp_path):
    pipe, _ = full_project
    old = tmp_path / "derivatives" / f"{RELS[0]}@BandPower.nc"
    target = tmp_path / "derivatives" / f"{RELS[0]}@band_power.nc"
    target.write_bytes(old.read_bytes())
    plan = _plan(pipe, tmp_path)
    assert plan.refusals == []
    assert any(str(old) in d["files"] for d in plan.to_dict()["duplicates"])
    apply_migration(plan)
    assert old.exists()
    assert target.exists()


def test_two_old_outputs_for_one_target_are_refused(full_project, tmp_path, capsys):
    """A pilot-era flat output and a full-tree output of the same source, different content."""
    pipe, sources = full_project
    _legacy_write(
        tmp_path / "derivatives", os.path.dirname(sources[0]), sources[0], "Clean", ".fif", "pilot"
    )
    out = _refused_cli(pipe, tmp_path, capsys)
    assert "map to the same new outputs" in out


def test_two_identical_old_outputs_keep_the_deeper_one(full_project, tmp_path):
    pipe, sources = full_project
    full = tmp_path / "derivatives" / f"{RELS[0]}@Clean.fif"
    flat = _legacy_write(
        tmp_path / "derivatives",
        os.path.dirname(sources[0]),
        sources[0],
        "Clean",
        ".fif",
        full.read_text(),
    )
    plan = _plan(pipe, tmp_path)
    assert plan.refusals == []
    apply_migration(plan)
    assert flat.exists()  # duplicate left in place
    assert (tmp_path / "derivatives" / f"{RELS[0]}@Clean.fif").exists()


def test_config_at_naming_version_1_with_unmigrated_tree_is_refused(full_project, tmp_path, capsys):
    pipe, _ = full_project
    text = "naming_version: 1\n" + PIPELINE_TEXT
    pipe.write_text(text)
    (tmp_path / "ds.yml").write_text(
        DATASETS_TEXT.replace("  file_pattern", "  source_root: raw\n  file_pattern")
    )
    out = _refused_cli(pipe, tmp_path, capsys)
    assert "already declares naming_version: 1" in out


def test_second_mount_tree_after_config_migration(tmp_path):
    """Run once per tree: the config is migrated on 'local'; the 'hpc' tree follows later."""
    datasets = """\
study:
  name: Study
  file_pattern:
    local: local/raw/**/*.txt
    hpc: hpc/raw/**/*.txt
  derivatives_path:
    local: local/derivatives
    hpc: hpc/derivatives
"""
    pipe = _project(
        tmp_path,
        pipeline=PIPELINE_TEXT.replace("mount_point: null", "mount_point: local"),
        datasets=datasets,
    )
    for mount in ("local", "hpc"):
        sources = _sources(tmp_path / mount / "raw", RELS)
        _legacy_tree(tmp_path / mount / "derivatives", sources, {"band_power": [".nc"]})

    plan = _plan(pipe, tmp_path)
    assert plan.source_roots["study"] == {
        "local": "local/raw (suggested)",
        "hpc": "hpc/raw (suggested)",
    }
    apply_migration(plan)
    assert yaml.safe_load((tmp_path / "ds.yml").read_text())["study"]["source_root"] == {
        "local": "local/raw",
        "hpc": "hpc/raw",
    }
    hpc = _plan(pipe, tmp_path, mount="hpc")
    assert hpc.mode == "tree-only"
    assert hpc.refusals == []
    assert len(hpc.moves) == 3
    apply_migration(hpc)
    prefix = str(tmp_path / "hpc/derivatives" / f"{RELS[0]}@band_power")
    assert unit_state(prefix, "band_power").state == "done"


def test_unwritable_derivatives_path_is_refused(full_project, tmp_path, capsys):
    if os.geteuid() == 0:
        pytest.skip("root ignores permissions")
    pipe, _ = full_project
    folder = tmp_path / "derivatives" / os.path.dirname(RELS[0])
    folder.chmod(0o555)
    try:
        out = _refused_cli(pipe, tmp_path, capsys)
    finally:
        folder.chmod(0o755)
    assert "not writable" in out


def test_cross_filesystem_moves_are_refused(full_project, tmp_path, monkeypatch):
    pipe, _ = full_project
    real_stat = os.stat
    old_dir = str(tmp_path / "derivatives" / os.path.dirname(RELS[0]))

    def fake_stat(path, *args, **kwargs):
        st = real_stat(path, *args, **kwargs)
        if os.fspath(path).startswith(old_dir + os.sep) and "@BandPower" in os.fspath(path):
            values = list(st)
            values[2] = st.st_dev + 1
            return os.stat_result(values)
        return st

    monkeypatch.setattr(migrate.os, "stat", fake_stat)
    plan = _plan(pipe, tmp_path)
    monkeypatch.setattr(migrate.os, "stat", real_stat)
    assert any("cross filesystems" in r for r in plan.refusals)
    with pytest.raises(MigrationRefused):
        apply_migration(plan)


def test_shared_derivatives_path_needs_subdirs(tmp_path, capsys):
    datasets = DATASETS_TEXT + """\
other:
  name: Other
  file_pattern: raw2/**/*.txt
  derivatives_path: derivatives
"""
    pipe = _project(tmp_path, datasets=datasets)
    _sources(tmp_path / "raw", RELS[:1])
    _sources(tmp_path / "raw2", ["sub-09/x.txt"])
    out = _refused_cli(pipe, tmp_path, capsys)
    assert "derivatives_subdir" in out


def test_shared_derivatives_path_with_subdirs_moves_into_subfolders(tmp_path):
    datasets = (
        DATASETS_TEXT
        + "  derivatives_subdir: a\n"
        + "other:\n  name: Other\n  file_pattern: raw2/**/*.txt\n"
        + "  derivatives_path: derivatives\n  derivatives_subdir: b\n"
    )
    pipe = _project(tmp_path, datasets=datasets)
    a = _sources(tmp_path / "raw", RELS[:2])
    b = _sources(tmp_path / "raw2", ["sub-09/eeg/x.txt", "sub-10/eeg/y.txt"])
    # 0.3.x: both datasets wrote into the same tree, each with its own root.
    _legacy_tree(tmp_path / "derivatives", a, {"band_power": [".nc"]})
    _legacy_tree(tmp_path / "derivatives", b, {"band_power": [".nc"]})

    plan = _plan(pipe, tmp_path)
    assert plan.refusals == []
    apply_migration(plan)
    assert (
        unit_state(str(tmp_path / "derivatives/a" / f"{RELS[0]}@band_power"), "band_power").state
        == "done"
    )
    assert (
        unit_state(str(tmp_path / "derivatives/b/sub-09/eeg/x.txt@band_power"), "band_power").state
        == "done"
    )


def test_invalid_0_4_names_need_map(tmp_path, capsys):
    pipeline = PIPELINE_TEXT.replace("  Clean:\n", "  Clean.v2:\n").replace(
        "  - Clean\n", "  - Clean.v2\n"
    )
    pipe = _project(tmp_path, pipeline=pipeline)
    sources = _sources(tmp_path / "raw", RELS[:1])
    _legacy_tree(tmp_path / "derivatives", sources, {"Clean.v2": [".fif"]})
    out = _refused_cli(pipe, tmp_path, capsys)
    assert "--map" in out

    plan = _plan(pipe, tmp_path, renames=["Clean.v2=clean_v2"])
    assert plan.refusals == []
    apply_migration(plan)
    text = (tmp_path / "pl.yml").read_text()
    assert "  clean_v2:\n" in text
    assert "  - clean_v2\n" in text
    assert (tmp_path / "derivatives" / f"{RELS[0]}@clean_v2.fif").exists()


def test_map_renames_references_and_files(full_project, tmp_path):
    pipe, _ = full_project
    apply_migration(_plan(pipe, tmp_path, renames=["band_power=BandPower_w-2"]))
    config = load_pipeline_configuration(str(pipe))
    assert "BandPower_w-2" in config["DerivativeDefinitions"]
    steps = config["DerivativeDefinitions"]["Ep_HiguchiFD"]["nodes"]
    assert steps[0]["derivative"] == "BandPower_w-2.nc"
    assert (tmp_path / "derivatives" / f"{RELS[0]}@BandPower_w-2.nc").exists()


def test_filesystem_root_suggestion_needs_explicit_source_root(tmp_path, capsys):
    datasets = DATASETS_TEXT.replace("raw/**/*.txt", "/*/never/**/*.txt")
    pipe = _project(tmp_path, datasets=datasets)
    out = _refused_cli(pipe, tmp_path, capsys)
    assert "--source-root study=PATH" in out


def test_explicit_source_root(full_project, tmp_path):
    pipe, _ = full_project
    plan = _plan(pipe, tmp_path, source_roots=["study=raw/sub-01/.."])
    assert plan.source_roots["study"] == {"*": "raw/sub-01/.. (given)"}
    assert plan.refusals == []


# ---------------------------------------------------------------------------
# Configuration rewrite details
# ---------------------------------------------------------------------------


def test_dataset_paths_are_rewritten_relative_to_the_yaml(tmp_path):
    """0.3.x resolved dataset paths against the working directory; 0.4.0 against the YAML."""
    (tmp_path / "config").mkdir()
    pipe = tmp_path / "config" / "pl.yml"
    pipe.write_text(PIPELINE_TEXT)
    (tmp_path / "config" / "ds.yml").write_text(
        DATASETS_TEXT.replace("raw/**", "data/raw/**").replace(
            "derivatives_path: derivatives", "derivatives_path: data/derivatives"
        )
    )
    sources = _sources(tmp_path / "data/raw", RELS)
    _legacy_tree(tmp_path / "data/derivatives", sources, {"band_power": [".nc"]})

    plan = plan_migration(pipe, legacy_cwd=tmp_path)
    assert plan.refusals == []
    apply_migration(plan)
    ds = yaml.safe_load((tmp_path / "config/ds.yml").read_text())["study"]
    assert ds["file_pattern"] == "../data/raw/**/*.txt"
    assert ds["derivatives_path"] == "../data/derivatives"
    assert ds["source_root"] == "../data/raw"
    prefix = str(tmp_path / "data/derivatives" / f"{RELS[0]}@band_power")
    assert unit_state(prefix, "band_power").state == "done"


def test_inline_datasets_and_flow_style_fallback(tmp_path, capsys):
    pipeline = PIPELINE_TEXT.replace(
        "datasets: ds.yml   # the datasets",
        "datasets: {study: {name: S, file_pattern: raw/**/*.txt, derivatives_path: derivatives}}",
    )
    pipe = _project(tmp_path, pipeline=pipeline)
    sources = _sources(tmp_path / "raw", RELS[:1])
    _legacy_tree(tmp_path / "derivatives", sources, {"band_power": [".nc"]})

    plan = _plan(pipe, tmp_path)
    assert [c.comments_preserved for c in plan.config_changes] == [False]
    assert "comments/formatting are lost" in migrate.format_preview(plan)
    apply_migration(plan)
    config = load_pipeline_configuration(str(pipe))
    assert config["datasets"]["study"]["source_root"] == "raw"
    assert (tmp_path / "pl.yml.bak-0.3").read_text() == pipeline


def test_report_contains_plan_and_column_map(full_project, tmp_path):
    pipe, _ = full_project
    report = tmp_path / "plan.json"
    assert main(["migrate", str(pipe), "--legacy-cwd", str(tmp_path), "--report", str(report)]) == 0
    data = json.loads(report.read_text())
    assert len(data["moves"]) == 6
    assert data["manifests"]
    assert data["config"]["files"]
    columns = data["column_map"]["derivatives"]["band_power"]
    assert columns["columns"] == {"band_power.nc": "band_power.nc"}
    assert columns["error_column"] == {"old": "band_power__error", "new": "band_power.error"}
    assert data["column_map"]["coordinates"]["exact"] is False
    assert data["refusals"] == []


def test_legacy_column_to_new():
    names = ["psd", "band_power"]
    assert legacy_column_to_new("band_power__error", names) == "band_power.error"
    assert legacy_column_to_new("psd.nc", names) == "psd.nc"
    assert legacy_column_to_new("psd.nc@ch-C3_band-8~12", names) == "psd.nc@band-8-12_ch-C3"
    assert legacy_column_to_new("psd.nc@cond-a|b", names) == "psd.nc@cond-a%5Fb"
    assert legacy_column_to_new("psd.nc@ref-a$b", names) == "psd.nc@ref-a%40b"
    assert legacy_column_to_new("psd.nc@x-NA", names) == "psd.nc@x-"
    assert legacy_column_to_new("psd.nc", names, {"psd": "Psd_w-2"}) == "Psd_w-2.nc"
    assert legacy_column_to_new("index", names) is None


def test_backup_with_different_content_is_refused(full_project, tmp_path, capsys):
    pipe, _ = full_project
    (tmp_path / "pl.yml.bak-0.3").write_text("other")
    out = _refused_cli(pipe, tmp_path, capsys)
    assert "already exists with different content" in out


def test_derivatives_without_derivatives_path_are_renamed_next_to_sources(tmp_path):
    datasets = DATASETS_TEXT.replace("  derivatives_path: derivatives\n", "")
    pipe = _project(tmp_path, datasets=datasets)
    sources = _sources(tmp_path / "raw", RELS[:2])
    for source in sources:
        Path(f"{source}@BandPower.nc").write_text("x")
    plan = _plan(pipe, tmp_path)
    assert [os.path.basename(n) for _, n in plan.moves] == [
        "sub-01_ses-1_eeg.txt@band_power.nc",
        "sub-02_ses-1_eeg.txt@band_power.nc",
    ]
    apply_migration(plan)
    assert unit_state(f"{sources[0]}@band_power", "band_power").state == "done"
    assert (tmp_path / ".neurodags" / "migrate-journal.json").exists()


# ---------------------------------------------------------------------------
# Regressions (review 114605e, migrate hunt)
# ---------------------------------------------------------------------------


def _mini_project(tmp_path: Path, name: str = "Clean", rels=("sub/x.txt",)) -> Path:
    """One dataset ``raw/**/*.txt`` -> ``out``, one derivative *name* (legacy_cwd tmp_path)."""
    (tmp_path / "out").mkdir(parents=True, exist_ok=True)
    _sources(tmp_path / "raw", rels)
    cfg = {
        "datasets": {"s": {"name": "S", "file_pattern": "raw/**/*.txt", "derivatives_path": "out"}},
        "DerivativeDefinitions": {name: dict(SIMPLE)},
        "DerivativeList": [name],
    }
    pipe = tmp_path / "p.yml"
    pipe.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return pipe


def _split_any(path: Path) -> list[str]:
    """A real MNE split recording at *path* (``.fif`` or ``.fif.gz``); its parts."""
    mne = pytest.importorskip("mne")
    np = pytest.importorskip("numpy")
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = mne.io.RawArray(
        np.random.default_rng(0).standard_normal((4, 200_000)) * 1e-5,
        mne.create_info(4, 1000.0, "eeg"),
        verbose="error",
    )
    parts = [str(p) for p in raw.save(str(path), split_size="2MB", overwrite=True, verbose="error")]
    assert len(parts) > 1
    return parts


def _reload_n_times(path: str) -> int:
    mne = pytest.importorskip("mne")
    return mne.io.read_raw_fif(path, preload=True, verbose="error").n_times


def test_mne_destination_created_after_planning_is_not_overwritten(tmp_path):
    """M1: a target that appears between plan and apply is never overwritten."""
    pipe = _mini_project(tmp_path, "clean_raw")
    parts = _split_any(tmp_path / "out/sub/x.txt@CleanRaw.fif")
    plan = _plan(pipe, tmp_path)
    target = Path(plan.mne_rewrites[0].new_path)
    sentinel = b"INDEPENDENT RECORDING CREATED AFTER PREVIEW"
    target.write_bytes(sentinel)

    with pytest.raises(MigrationRefused, match="appeared since the plan"):
        apply_migration(plan)
    assert target.read_bytes() == sentinel
    assert all(os.path.exists(p) for p in parts)


def test_mne_continuation_created_after_planning_is_not_overwritten(tmp_path):
    """M1: the same for a continuation name ('-1.fif') of the target."""
    pipe = _mini_project(tmp_path, "clean_raw")
    parts = _split_any(tmp_path / "out/sub/x.txt@CleanRaw.fif")
    plan = _plan(pipe, tmp_path)
    other = tmp_path / "out/sub/x.txt@clean_raw-1.fif"
    other.write_bytes(b"not ours")

    with pytest.raises(MigrationRefused, match="appeared since the plan"):
        apply_migration(plan)
    assert other.read_bytes() == b"not ours"
    assert all(os.path.exists(p) for p in parts)


def test_apply_refuses_while_another_apply_holds_the_lock(full_project, tmp_path):
    """M1: concurrent --apply runs on one output tree are serialized by a lock file."""
    pipe, _ = full_project
    plan = _plan(pipe, tmp_path)
    lock = tmp_path / "derivatives/.neurodags" / migrate.LOCK_NAME
    lock.parent.mkdir(parents=True)
    lock.write_text(json.dumps({"pid": os.getpid(), "host": migrate._hostname()}))
    before = _snapshot(tmp_path / "derivatives")

    with pytest.raises(MigrationRefused, match="another 'neurodags migrate --apply'"):
        apply_migration(plan)
    assert _snapshot(tmp_path / "derivatives") == before
    lock.unlink()
    apply_migration(plan)
    assert not lock.exists()  # released


def test_stale_lock_of_a_dead_process_is_taken_over(full_project, tmp_path):
    pipe, _ = full_project
    lock = tmp_path / "derivatives/.neurodags" / migrate.LOCK_NAME
    lock.parent.mkdir(parents=True)
    lock.write_text(json.dumps({"pid": 2**22 + 12345, "host": migrate._hostname()}))
    apply_migration(_plan(pipe, tmp_path))
    assert not lock.exists()


@pytest.mark.parametrize("target", ["B", "b"])
def test_map_onto_another_existing_definition_is_refused(tmp_path, target):
    """M2: --map A=B (or a case alias of B) with B defined would collapse two definitions."""
    pipe = _mini_project(tmp_path, "A", ("x.txt",))
    cfg = yaml.safe_load(pipe.read_text())
    cfg["DerivativeDefinitions"] = {
        n: {"nodes": [{"id": 0, "node": "dummy", "args": {"param1": n}}]} for n in ("A", "B")
    }
    cfg["DerivativeList"] = ["A", "B"]
    pipe.write_text(yaml.safe_dump(cfg, sort_keys=False))
    (tmp_path / "out/x.txt@A.message.txt").write_text("A result")
    text = pipe.read_text()

    plan = _plan(pipe, tmp_path, renames=[f"A={target}"])
    joined = "\n".join(plan.refusals)
    assert "'A'" in joined
    assert "'B'" in joined
    with pytest.raises(MigrationRefused):
        apply_migration(plan)
    assert pipe.read_text() == text
    assert not (tmp_path / f"out/x.txt@{target}.message.txt").exists()


def test_compressed_split_recording_is_rewritten_as_one_artifact(tmp_path):
    """M3: '.fif-N.gz' continuations belong to the '.fif.gz' artifact (MNE rewrite path)."""
    pipe = _mini_project(tmp_path, "clean_raw")
    old = tmp_path / "out/sub/x.txt@CleanRaw.fif.gz"
    parts = _split_any(old)
    assert any(p.endswith(".fif-1.gz") for p in parts)
    n_times = _reload_n_times(str(old))

    plan = _plan(pipe, tmp_path)
    assert plan.refusals == []
    assert plan.moves == []
    assert [r.old_parts for r in plan.mne_rewrites] == [parts]
    apply_migration(plan)

    prefix = str(tmp_path / "out/sub/x.txt@clean_raw")
    state = unit_state(prefix, "clean_raw")
    assert state.state == "done"
    assert list(state.artifacts) == [".fif.gz"]
    assert len(state.artifacts[".fif.gz"]) > 1
    assert _reload_n_times(prefix + ".fif.gz") == n_times
    assert not any(os.path.exists(p) for p in parts)
    assert not _plan(pipe, tmp_path).has_work


def test_compressed_split_rewrite_interrupted_while_deleting_old_parts(tmp_path, monkeypatch):
    """M3: resume of a verified '.fif.gz' rewrite only finishes deleting the old parts."""
    pipe = _mini_project(tmp_path, "clean_raw")
    parts = _split_any(tmp_path / "out/sub/x.txt@CleanRaw.fif.gz")
    real_unlink = os.unlink

    def interrupt(path, *args, **kwargs):
        if os.fspath(path) == parts[0]:
            raise OSError("interrupted while deleting")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(migrate.os, "unlink", interrupt)
    with pytest.raises(OSError, match="interrupted while deleting"):
        apply_migration(_plan(pipe, tmp_path))
    monkeypatch.setattr(migrate.os, "unlink", real_unlink)
    assert os.path.exists(parts[0])

    resumed = _plan(pipe, tmp_path)
    assert resumed.refusals == []
    assert resumed.to_dict()["unrecognized"] == []
    assert [r.finished_parts is not None for r in resumed.mne_rewrites] == [True]
    apply_migration(resumed)
    prefix = str(tmp_path / "out/sub/x.txt@clean_raw")
    state = unit_state(prefix, "clean_raw")
    assert state.state == "done"
    assert len(state.artifacts[".fif.gz"]) > 1
    _reload_n_times(prefix + ".fif.gz")
    assert not any(os.path.exists(p) for p in parts)


def test_compressed_split_recording_with_unchanged_name_is_moved_whole(tmp_path):
    """M3: folder-only move of a '.fif.gz' split keeps every part under one artifact."""
    pipe = _mini_project(tmp_path)
    parts = _split_any(tmp_path / "out/x.txt@Clean.fif.gz")
    plan = _plan(pipe, tmp_path)
    assert plan.refusals == []
    assert plan.to_dict()["unrecognized"] == []
    assert len(plan.moves) == len(parts)
    apply_migration(plan)
    prefix = str(tmp_path / "out/sub/x.txt@Clean")
    state = unit_state(prefix, "Clean")
    assert state.state == "done"
    assert list(state.artifacts) == [".fif.gz"]
    assert len(state.artifacts[".fif.gz"]) == len(parts)
    _reload_n_times(prefix + ".fif.gz")


@pytest.mark.parametrize("moved_before_crash", [1, 2, 3])
def test_interrupted_folder_only_split_move_resumes_with_every_part(
    tmp_path, monkeypatch, moved_before_crash
):
    """M4: interrupted at any part boundary, resume rejoins the parts across both folders."""
    pipe = _mini_project(tmp_path)
    parts = _split_any(tmp_path / "out/x.txt@Clean.fif")
    assert len(parts) > moved_before_crash
    n_times = _reload_n_times(parts[0])
    plan = _plan(pipe, tmp_path)
    real_replace = os.replace
    moved: list[str] = []

    def interrupt(src, dst):
        if os.fspath(src) in parts:
            if len(moved) == moved_before_crash:
                raise OSError("interrupted between parts")
            moved.append(os.fspath(src))
        return real_replace(src, dst)

    monkeypatch.setattr(migrate.os, "replace", interrupt)
    with pytest.raises(OSError, match="interrupted between parts"):
        apply_migration(plan)
    monkeypatch.setattr(migrate.os, "replace", real_replace)

    resumed = _plan(pipe, tmp_path)
    assert resumed.refusals == []
    assert resumed.to_dict()["unrecognized"] == []
    apply_migration(resumed)
    prefix = str(tmp_path / "out/sub/x.txt@Clean")
    state = unit_state(prefix, "Clean")
    assert state.state == "done"
    assert len(state.artifacts[".fif"]) == len(parts)
    assert _reload_n_times(prefix + ".fif") == n_times
    assert not any(os.path.exists(p) for p in parts)
    assert not _plan(pipe, tmp_path).has_work


def test_folder_only_split_move_that_does_not_reload_gets_no_manifest(tmp_path):
    """M4: a moved split recording is reloaded before its .done is written."""
    pipe = _mini_project(tmp_path)
    parts = _split_any(tmp_path / "out/x.txt@Clean.fif")
    Path(parts[-1]).write_bytes(b"truncated")
    result = apply_migration(_plan(pipe, tmp_path))
    prefix = str(tmp_path / "out/sub/x.txt@Clean")
    assert unit_state(prefix, "Clean").state == "incomplete"
    assert len(result.recompute) == 1


def test_unit_with_extra_artifacts_keeps_them_in_its_manifest(tmp_path):
    """M5: the common suffix set decides completeness; the manifest lists every artifact."""
    pipe = _mini_project(tmp_path, "D", ("0.txt", "1.txt", "2.txt"))
    for i in range(3):
        (tmp_path / f"out/{i}.txt@D.nc").write_text("nc")
    (tmp_path / "out/2.txt@D.report.json").write_text("extra result")
    plan = _plan(pipe, tmp_path)
    assert plan.artifact_sets["D"]["known"] == [".nc"]
    apply_migration(plan)
    state = unit_state(str(tmp_path / "out/2.txt@D"), "D")
    assert state.state == "done"
    assert sorted(state.artifacts) == [".nc", ".report.json"]
    assert not state.unlisted
    assert sorted(unit_state(str(tmp_path / "out/0.txt@D"), "D").artifacts) == [".nc"]


def test_pilot_split_recording_with_rename_into_new_folder(tmp_path):
    """M6: rename + folder change: the destination folder is created before the rewrite."""
    pipe = _mini_project(tmp_path, "clean_raw")
    parts = _split_any(tmp_path / "out/x.txt@CleanRaw.fif")
    n_times = _reload_n_times(parts[0])
    plan = _plan(pipe, tmp_path)
    assert plan.refusals == []
    assert not Path(plan.mne_rewrites[0].new_path).parent.exists()
    result = apply_migration(plan)
    assert result.recompute == []
    prefix = str(tmp_path / "out/sub/x.txt@clean_raw")
    assert unit_state(prefix, "clean_raw").state == "done"
    assert _reload_n_times(prefix + ".fif") == n_times
    assert not any(os.path.exists(p) for p in parts)


def test_rerun_after_keeping_an_identical_duplicate_is_a_no_op(tmp_path, monkeypatch):
    """M7: the kept unit's new .done must not make its identical duplicate 'different'."""
    pipe = _mini_project(tmp_path, "clean_raw")
    for rel in ("x.txt@CleanRaw.nc", "sub/x.txt@CleanRaw.nc"):
        (tmp_path / "out" / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / "out" / rel).write_text("same")
    plan = _plan(pipe, tmp_path)
    assert plan.refusals == []
    assert len(plan.units("duplicate")) == 1
    apply_migration(plan)

    again = _plan(pipe, tmp_path)
    assert again.refusals == []
    assert not again.has_work
    monkeypatch.chdir(tmp_path)
    assert main(["migrate", str(pipe), "--apply"]) == 0
    assert (tmp_path / "out/x.txt@CleanRaw.nc").exists()  # duplicate left in place


def test_identical_duplicate_rerun_after_keeper_manifest_before_config(tmp_path, monkeypatch):
    """M7: interrupted after the keeper's .done, before the configuration: resume completes."""
    pipe = _mini_project(tmp_path, "clean_raw")
    for rel in ("x.txt@CleanRaw.nc", "sub/x.txt@CleanRaw.nc"):
        (tmp_path / "out" / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / "out" / rel).write_text("same")

    def crash(change):
        raise OSError("crash before the configuration")

    monkeypatch.setattr(migrate, "_write_config", crash)
    with pytest.raises(OSError, match="crash before the configuration"):
        apply_migration(_plan(pipe, tmp_path))
    monkeypatch.undo()
    assert unit_state(str(tmp_path / "out/sub/x.txt@clean_raw"), "clean_raw").state == "done"

    resumed = _plan(pipe, tmp_path)
    assert resumed.refusals == []
    apply_migration(resumed)
    assert "naming_version: 1" in pipe.read_text()
    assert not _plan(pipe, tmp_path).has_work


def test_different_duplicate_of_a_migrated_keeper_is_still_refused(tmp_path):
    """M7: a duplicate whose bytes differ from the migrated keeper is still refused."""
    pipe = _mini_project(tmp_path, "clean_raw")
    for rel in ("x.txt@CleanRaw.nc", "sub/x.txt@CleanRaw.nc"):
        (tmp_path / "out" / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / "out" / rel).write_text("same")
    apply_migration(_plan(pipe, tmp_path))
    (tmp_path / "out/x.txt@CleanRaw.nc").write_text("different")
    assert any("different content" in r for r in _plan(pipe, tmp_path).refusals)
