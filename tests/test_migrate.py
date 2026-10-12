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


def _legacy_tree(derivs: Path, sources: list[str], outputs: dict[str, list[str]]) -> list[Path]:
    """0.3.x outputs for every source: ``{name: [tails]}`` (root over the given sources)."""
    root = find_unique_root(sources, mode="maximal")
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
    """Sources that are themselves outputs ('@' in the name) were written with '&'."""
    pipe = _project(tmp_path, datasets=CHAINED_DATASETS)
    sources = _sources(
        tmp_path / "upstream", ["sub-01/x.vhdr@CleanPrep.fif", "sub-02/y.vhdr@CleanPrep.fif"]
    )
    outputs = _legacy_tree(tmp_path / "derivatives", sources, {"band_power": [".nc"]})
    assert outputs[0].name == "x.vhdr&CleanPrep.fif@BandPower.nc"

    plan = _plan(pipe, tmp_path)
    assert plan.refusals == []
    assert [os.path.basename(n) for _, n in plan.moves] == [
        "x.vhdr&CleanPrep.fif@band_power.nc",
        "y.vhdr&CleanPrep.fif@band_power.nc",
    ]
    # Patterns referencing 0.3.x output names are printed, never edited.
    assert any("*@CleanPrep.fif" in p for p in plan.chained_patterns)
    apply_migration(plan)
    assert "*@CleanPrep.fif" in (tmp_path / "ds.yml").read_text()
    manifest = json.loads(
        (tmp_path / "derivatives/sub-01/x.vhdr&CleanPrep.fif@band_power.done").read_text()
    )
    assert manifest["source"]["rel"] == "sub-01/x.vhdr@CleanPrep.fif"


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
    assert len(without.units("orphan")) == 1

    plan = _plan(pipe, tmp_path, upstream_renames=["CleanPrep=clean_prep"])
    assert [os.path.basename(n) for _, n in plan.moves] == ["x.vhdr&clean_prep.fif@band_power.nc"]


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
