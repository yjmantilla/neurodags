"""Tests for the top-level neurodags CLI."""

from __future__ import annotations

import json
import subprocess
import sys
import warnings
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from neurodags.cli import main, resolve_cli_datasets_path
from neurodags.iterators import InputEnumeration


def test_importing_cli_keeps_stdout_clean():
    # #15: import-time derivative registration must not print to stdout (it used
    # to emit DEBUG "Registered derivative" lines via unconfigured structlog,
    # which polluted command output and broke `--format json` piping).
    result = subprocess.run(
        [sys.executable, "-c", "import neurodags.cli"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == "", f"unexpected stdout on import: {result.stdout!r}"


def test_tui_subcommand_launches_app(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "datasets.yml").write_text("{}\n")
    with (
        patch("neurodags.cli.NeuroDagsApp", create=True),
        patch("neurodags.cli._cmd_tui") as cmd,
    ):
        cmd.return_value = 0
        assert main(["tui", "pipeline.yml", "--datasets", "datasets.yml"]) == 0
    assert cmd.call_args.args[0].datasets == str(tmp_path / "datasets.yml")


def test_validate_prints_summary(tmp_path, capsys):
    pipe = tmp_path / "pipeline.yml"
    data = tmp_path / "datasets.yml"
    pipe.write_text(
        "naming_version: 1\n"
        "datasets: datasets.yml\n"
        "mount_point: null\n"
        "DerivativeDefinitions:\n"
        "  BasicPrep: {}\n"
        "DerivativeList:\n"
        "  - BasicPrep\n"
    )
    data.write_text(
        "demo:\n"
        "  name: Demo\n"
        f"  source_root: {tmp_path}\n"
        f"  file_pattern: {tmp_path}/**/*.vhdr\n"
        f"  derivatives_path: {tmp_path}/derivatives\n"
    )

    assert main(["validate", str(pipe)]) == 0
    out = capsys.readouterr().out
    assert "datasets: demo" in out
    assert "derivatives_defined (1): BasicPrep" in out
    assert "run_set [DerivativeList] (1): BasicPrep" in out
    assert "computed_with_dependencies (1): BasicPrep" in out
    # the misleading "enabled == defined" label is gone
    assert "derivatives_enabled" not in out


def test_validate_accepts_anchor_and_merge_keys(tmp_path, capsys):
    # #44: a top-level helper key holding an anchor, merged into node args.
    pipe = tmp_path / "pipeline.yml"
    data = tmp_path / "datasets.yml"
    pipe.write_text(
        "naming_version: 1\n"
        "datasets: datasets.yml\n"
        "mount_point: null\n"
        "_defaults: &psd\n"
        "  fmin: 1\n"
        "  fmax: 40\n"
        "DerivativeDefinitions:\n"
        "  PsdA:\n"
        "    nodes:\n"
        "      - {id: 0, derivative: SourceFile}\n"
        "      - {id: 1, node: my_psd, args: {<<: *psd, data: id.0}}\n"
        "  PsdB:\n"
        "    nodes:\n"
        "      - id: 0\n"
        "        derivative: PsdA.nc\n"
        "      - id: 1\n"
        "        node: my_psd\n"
        "        args:\n"
        "          <<: *psd\n"
        "          fmax: 30\n"
        "          data: id.0\n"
        "DerivativeList:\n"
        "  - PsdB\n"
    )
    data.write_text(
        "demo:\n"
        "  name: Demo\n"
        f"  source_root: {tmp_path}\n"
        f"  file_pattern: {tmp_path}/**/*.vhdr\n"
        f"  derivatives_path: {tmp_path}/derivatives\n"
    )

    assert main(["validate", str(pipe)]) == 0
    out = capsys.readouterr().out
    assert "derivatives_defined (2): PsdA, PsdB" in out
    assert "computed_with_dependencies (2): PsdA, PsdB" in out

    from neurodags.loaders import load_configuration

    defs = load_configuration(pipe)["DerivativeDefinitions"]
    assert defs["PsdA"]["nodes"][1]["args"] == {"fmin": 1, "fmax": 40, "data": "id.0"}
    assert defs["PsdB"]["nodes"][1]["args"] == {"fmin": 1, "fmax": 30, "data": "id.0"}


def test_validate_shows_effective_run_set_with_dependencies(tmp_path, capsys):
    pipe = tmp_path / "pipeline.yml"
    data = tmp_path / "datasets.yml"
    pipe.write_text(
        "naming_version: 1\n"
        "datasets: datasets.yml\n"
        "mount_point: null\n"
        "DerivativeDefinitions:\n"
        "  A: {nodes: [{id: 0, derivative: SourceFile}]}\n"
        "  B: {nodes: [{id: 0, derivative: A.nc}]}\n"
        "  C: {for_dataframe: true, nodes: [{id: 0, derivative: B.nc}]}\n"
        "  Extra: {nodes: [{id: 0, derivative: SourceFile}]}\n"
        "DerivativeList:\n"
        "  - A\n"
        "  - B\n"
        "  - C\n"
        "  - Extra\n"
    )
    data.write_text(
        "demo:\n"
        "  name: Demo\n"
        f"  source_root: {tmp_path}\n"
        f"  file_pattern: {tmp_path}/**/*.vhdr\n"
        f"  derivatives_path: {tmp_path}/derivatives\n"
    )

    # Selecting only C must reveal that B and A are pulled in as dependencies,
    # C is a for_dataframe output, and Extra is not computed by this run.
    assert main(["validate", str(pipe), "--derivative", "C"]) == 0
    out = capsys.readouterr().out
    assert "run_set [--derivative] (1): C" in out
    assert "computed_with_dependencies (3): A, B, C" in out
    assert "pulled in as dependencies: A, B" in out
    assert "for_dataframe outputs: C" in out
    assert "not computed by this run (1): Extra" in out


def test_run_uses_derivative_list_when_not_explicit(dummy_pipeline):
    cfg = dummy_pipeline["config"]
    with (
        patch("neurodags.cli._load_pipeline_config", return_value=cfg),
        patch("neurodags.orchestrators.load_pipeline_configuration", return_value=cfg),
        patch("neurodags.orchestrators.iterate_derivative_pipeline") as iterate,
    ):
        assert main(["run", "pipeline.yml"]) == 0
        called = [call.kwargs["derivative"] for call in iterate.call_args_list]
        assert set(called) == set(cfg["DerivativeList"])


def test_run_with_explicit_derivative(dummy_pipeline):
    cfg = dummy_pipeline["config"]
    with (
        patch("neurodags.cli._load_pipeline_config", return_value=cfg),
        patch("neurodags.orchestrators.load_pipeline_configuration", return_value=cfg),
        patch("neurodags.orchestrators.iterate_derivative_pipeline") as iterate,
    ):
        assert main(["run", "pipeline.yml", "--derivative", "BasicPrep"]) == 0
        iterate.assert_called_once()
        assert iterate.call_args.kwargs["derivative"] == "BasicPrep"


def test_dry_run_saves_combined_dataframe(dummy_pipeline):
    cfg = dummy_pipeline["config"]
    fake_df = pd.DataFrame({"file_path": ["a"], "plan": [[]]})
    with (
        patch("neurodags.cli._load_pipeline_config", return_value=cfg),
        patch("neurodags.orchestrators.load_pipeline_configuration", return_value=cfg),
        patch("neurodags.orchestrators.iterate_derivative_pipeline", return_value=fake_df),
        patch("neurodags.cli._save_dataframe") as save_df,
    ):
        save_df.return_value = Path("dry_run_results.csv")
        assert main(["dry-run", "pipeline.yml", "--derivative", "BasicPrep"]) == 0
        save_df.assert_called_once()


def test_dataframe_command_passes_arguments(dummy_pipeline):
    cfg = dummy_pipeline["config"]
    fake_df = pd.DataFrame({"x": [1]})
    with (
        patch("neurodags.cli._load_pipeline_config", return_value=cfg),
        patch("neurodags.cli.build_derivative_dataframe", return_value=fake_df) as build_df,
    ):
        assert (
            main(
                [
                    "dataframe",
                    "pipeline.yml",
                    "--include-derivative",
                    "BandPowerMean",
                    "--format",
                    "long",
                ]
            )
            == 0
        )
        _, kwargs = build_df.call_args
        assert kwargs["include_derivatives"] == ["BandPowerMean"]
        assert kwargs["output_format"] == "long"


def test_dataframe_njobs_passed_to_build_derivative_dataframe(dummy_pipeline):
    cfg = dummy_pipeline["config"]
    fake_df = pd.DataFrame({"x": [1]})
    with (
        patch("neurodags.cli._load_pipeline_config", return_value=cfg),
        patch("neurodags.cli.build_derivative_dataframe", return_value=fake_df) as build_df,
    ):
        assert main(["dataframe", "pipeline.yml", "--n-jobs", "4"]) == 0
        _, kwargs = build_df.call_args
        assert kwargs.get("n_jobs") == 4


def test_dag_pipeline_prints_mermaid(capsys):
    with (
        patch("neurodags.cli._load_pipeline_config", return_value={"DerivativeDefinitions": {}}),
        patch("neurodags.cli.pipeline_to_mermaid", return_value="graph TD\nA-->B"),
    ):
        assert main(["dag", "pipeline.yml"]) == 0
        assert "graph TD" in capsys.readouterr().out


def test_dag_derivative_html_exports():
    cfg = {"DerivativeDefinitions": {"BandPower": {"nodes": []}}}
    with (
        patch("neurodags.cli._load_pipeline_config", return_value=cfg),
        patch("neurodags.cli.derivative_to_html", return_value=Path("bandpower.html")) as export,
    ):
        assert main(["dag", "pipeline.yml", "--derivative", "BandPower", "--html", "out.html"]) == 0
        export.assert_called_once()


def test_dag_layout_elk_passed_to_derivative_html():
    cfg = {"DerivativeDefinitions": {"BandPower": {"nodes": []}}}
    with (
        patch("neurodags.cli._load_pipeline_config", return_value=cfg),
        patch("neurodags.cli.derivative_to_html", return_value=Path("out.html")) as export,
    ):
        main(
            [
                "dag",
                "pipeline.yml",
                "--derivative",
                "BandPower",
                "--html",
                "out.html",
                "--layout",
                "elk",
            ]
        )
        _, kwargs = export.call_args
        assert kwargs.get("layout") == "elk"


def test_dag_layout_elk_passed_to_pipeline_html():
    cfg = {"DerivativeDefinitions": {}}
    with (
        patch("neurodags.cli._load_pipeline_config", return_value=cfg),
        patch("neurodags.cli.pipeline_to_html", return_value=Path("out.html")) as export,
    ):
        main(["dag", "pipeline.yml", "--html", "out.html", "--layout", "elk"])
        _, kwargs = export.call_args
        assert kwargs.get("layout") == "elk"


def test_view_launches_visualization_subprocess():
    with patch("neurodags.cli.subprocess.Popen") as popen:
        assert main(["view", "result.nc"]) == 0
        popen.assert_called_once()


# ---------------------------------------------------------------------------
# count command
# ---------------------------------------------------------------------------


def test_count_prints_number_of_entries(dummy_pipeline, capsys):
    from neurodags.iterators import InputEntry, InputEnumeration

    cfg = dummy_pipeline["config"]
    fake = InputEnumeration(
        entries=[InputEntry(0, "d", "a", "/r/a"), InputEntry(1, "d", "b", "/r/b")]
    )
    with (
        patch("neurodags.cli._load_pipeline_config", return_value=cfg),
        patch("neurodags.cli._cli_enumeration", return_value=fake),
    ):
        assert main(["count-inputs", "pipeline.yml"]) == 0
    assert capsys.readouterr().out.strip() == "2"


def test_count_counts_dataset_entries_not_unique_paths(dummy_pipeline, tmp_path, capsys):
    """The same file under two dataset entries has two indices, so it needs two array tasks."""
    import copy

    import yaml

    cfg = copy.deepcopy(dummy_pipeline["config"])
    entry = cfg["datasets"]["test_dataset"]
    cfg["datasets"]["condition_b"] = {
        **entry,
        "name": "ConditionB",
        "derivatives_path": str(tmp_path / "derivatives_b"),
    }
    pipeline_yml = tmp_path / "pipeline.yml"
    pipeline_yml.write_text(yaml.safe_dump(cfg))

    assert main(["count-inputs", str(pipeline_yml)]) == 0
    # 2 files x 2 dataset entries = indices 0..3
    assert capsys.readouterr().out.strip() == "4"


def test_count_prints_zero_on_empty_result(dummy_pipeline, capsys):

    cfg = dummy_pipeline["config"]
    with (
        patch("neurodags.cli._load_pipeline_config", return_value=cfg),
        patch("neurodags.cli._cli_enumeration", return_value=InputEnumeration(entries=[])),
    ):
        assert main(["count-inputs", "pipeline.yml"]) == 0
    assert capsys.readouterr().out.strip() == "0"


# ---------------------------------------------------------------------------
# slurm-script command
# ---------------------------------------------------------------------------


def _make_slurm_config(derivatives: list[str]) -> dict:
    defs = {d: {"nodes": [{"id": 0, "derivative": "SourceFile"}]} for d in derivatives}
    return {
        "datasets": {},
        "mount_point": None,
        "DerivativeDefinitions": defs,
        "DerivativeList": derivatives,
    }


def test_slurm_script_per_file_contains_derivative_names(capsys):
    cfg = _make_slurm_config(["BasicPrep", "Spectrum"])
    with patch("neurodags.cli._load_pipeline_config", return_value=cfg):
        assert main(["slurm-script", "pipeline.yml", "--pattern", "per-file"]) == 0
    out = capsys.readouterr().out
    assert "BasicPrep" in out
    assert "Spectrum" in out
    assert "SLURM_ARRAY_TASK_ID" in out


def test_slurm_script_flat_contains_n_derivatives(capsys):
    cfg = _make_slurm_config(["A", "B", "C"])
    with patch("neurodags.cli._load_pipeline_config", return_value=cfg):
        assert main(["slurm-script", "pipeline.yml", "--pattern", "flat"]) == 0
    out = capsys.readouterr().out
    assert "N_DERIVATIVES=3" in out
    assert '"A"' in out
    assert '"B"' in out
    assert '"C"' in out


def test_slurm_script_chained_chains_dependencies(capsys):
    cfg = _make_slurm_config(["BasicPrep", "Spectrum", "BandPower"])
    with patch("neurodags.cli._load_pipeline_config", return_value=cfg):
        assert main(["slurm-script", "pipeline.yml", "--pattern", "chained"]) == 0
    out = capsys.readouterr().out
    assert "--dependency=afterok:$JOB1" in out
    assert "--dependency=afterok:$JOB2" in out
    assert "submit_pipeline.sh" in out
    assert "run_one_derivative.sh" in out


def test_slurm_script_chained_writes_two_files(tmp_path):
    cfg = _make_slurm_config(["BasicPrep", "Spectrum"])
    submit_path = tmp_path / "submit.sh"
    with patch("neurodags.cli._load_pipeline_config", return_value=cfg):
        assert (
            main(
                [
                    "slurm-script",
                    "pipeline.yml",
                    "--pattern",
                    "chained",
                    "--output",
                    str(submit_path),
                ]
            )
            == 0
        )
    assert submit_path.exists()
    assert (tmp_path / "run_one_derivative.sh").exists()
    assert "BasicPrep" in submit_path.read_text()


def test_slurm_script_per_file_writes_file(tmp_path):
    cfg = _make_slurm_config(["BasicPrep"])
    out_path = tmp_path / "run_array.sh"
    with patch("neurodags.cli._load_pipeline_config", return_value=cfg):
        assert (
            main(
                ["slurm-script", "pipeline.yml", "--pattern", "per-file", "--output", str(out_path)]
            )
            == 0
        )
    assert out_path.exists()
    assert "SLURM_ARRAY_TASK_ID" in out_path.read_text()


def test_slurm_script_respects_topo_order(capsys):
    # Spectrum depends on BasicPrep — BasicPrep must appear first in pattern 3
    cfg = {
        "datasets": {},
        "mount_point": None,
        "DerivativeDefinitions": {
            "Spectrum": {"nodes": [{"id": 0, "derivative": "BasicPrep.fif"}]},
            "BasicPrep": {"nodes": [{"id": 0, "derivative": "SourceFile"}]},
        },
        "DerivativeList": ["Spectrum", "BasicPrep"],
    }
    with patch("neurodags.cli._load_pipeline_config", return_value=cfg):
        assert main(["slurm-script", "pipeline.yml", "--pattern", "chained"]) == 0
    out = capsys.readouterr().out
    assert out.index("BasicPrep") < out.index("Spectrum")


# ---------------------------------------------------------------------------
# status command
# ---------------------------------------------------------------------------


def _make_plan(*, cached: bool, has_error: bool, error_path: str | None = None) -> list:
    return [
        {
            "id": "final",
            "kind": "derivative_output",
            "name": "BasicPrep",
            "prefix": "/out/file@BasicPrep",
            "cached": cached,
            "paths": ["/out/file@BasicPrep.fif"] if cached else [],
            "has_error_marker": has_error,
            "error_path": error_path,
        }
    ]


def _status_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_status_no_files_exits_0(dummy_pipeline, capsys):
    cfg = dummy_pipeline["config"]
    with (
        patch("neurodags.cli._load_pipeline_config", return_value=cfg),
        patch("neurodags.cli.run_pipeline", return_value=pd.DataFrame()),
    ):
        assert main(["status", "pipeline.yml"]) == 0
    assert "No files" in capsys.readouterr().out


def test_status_format_json_stdout_is_pure_json(dummy_pipeline, capsys):
    # #15: `status --format json` must emit only the JSON document on stdout;
    # framework/dry-run logs go to stderr, so the deliverable is pipeable.
    cfg = dummy_pipeline["config"]
    fake_df = _status_df(
        [
            {
                "derivative": "BasicPrep",
                "file_path": "a.vhdr",
                "plan": _make_plan(cached=True, has_error=False),
            }
        ]
    )
    with (
        patch("neurodags.cli._load_pipeline_config", return_value=cfg),
        patch("neurodags.cli.run_pipeline", return_value=fake_df),
    ):
        rc = main(["status", "pipeline.yml", "--format", "json"])
    out = capsys.readouterr().out
    assert rc == 0
    payload = json.loads(out)  # stdout must parse cleanly as a single JSON document
    assert payload["complete"] is True
    assert "derivatives" in payload


def test_status_prints_summary_table(dummy_pipeline, capsys):
    cfg = dummy_pipeline["config"]
    fake_df = _status_df(
        [
            {
                "derivative": "BasicPrep",
                "file_path": "a.vhdr",
                "plan": _make_plan(cached=True, has_error=False),
            },
            {
                "derivative": "BasicPrep",
                "file_path": "b.vhdr",
                "plan": _make_plan(cached=False, has_error=False),
            },
            {
                "derivative": "BasicPrep",
                "file_path": "c.vhdr",
                "plan": _make_plan(cached=False, has_error=True, error_path="c@BasicPrep.error"),
            },
        ]
    )
    with (
        patch("neurodags.cli._load_pipeline_config", return_value=cfg),
        patch("neurodags.cli.run_pipeline", return_value=fake_df),
    ):
        rc = main(["status", "pipeline.yml"])
    out = capsys.readouterr().out
    assert rc == 1  # has errors
    assert "BasicPrep" in out
    assert "1" in out  # done count
    assert "Total" in out


def test_status_exit_0_when_complete(dummy_pipeline, capsys):
    cfg = dummy_pipeline["config"]
    fake_df = _status_df(
        [
            {
                "derivative": "BasicPrep",
                "file_path": "a.vhdr",
                "plan": _make_plan(cached=True, has_error=False),
            },
            {
                "derivative": "BasicPrep",
                "file_path": "b.vhdr",
                "plan": _make_plan(cached=True, has_error=False),
            },
        ]
    )
    with (
        patch("neurodags.cli._load_pipeline_config", return_value=cfg),
        patch("neurodags.cli.run_pipeline", return_value=fake_df),
    ):
        assert main(["status", "pipeline.yml"]) == 0


def test_status_exit_1_when_missing(dummy_pipeline, capsys):
    cfg = dummy_pipeline["config"]
    fake_df = _status_df(
        [
            {
                "derivative": "BasicPrep",
                "file_path": "a.vhdr",
                "plan": _make_plan(cached=True, has_error=False),
            },
            {
                "derivative": "BasicPrep",
                "file_path": "b.vhdr",
                "plan": _make_plan(cached=False, has_error=False),
            },
        ]
    )
    with (
        patch("neurodags.cli._load_pipeline_config", return_value=cfg),
        patch("neurodags.cli.run_pipeline", return_value=fake_df),
    ):
        assert main(["status", "pipeline.yml"]) == 1


def test_status_list_errors_prints_file_paths(dummy_pipeline, capsys):
    cfg = dummy_pipeline["config"]
    fake_df = _status_df(
        [
            {
                "index": 3,
                "dataset": "ds",
                "derivative": "BasicPrep",
                "file_path": "/data/sub-01.vhdr",
                "plan": _make_plan(
                    cached=False, has_error=True, error_path="/out/sub-01@BasicPrep.error"
                ),
            },
        ]
    )
    with (
        patch("neurodags.cli._load_pipeline_config", return_value=cfg),
        patch("neurodags.cli.run_pipeline", return_value=fake_df),
    ):
        main(["status", "pipeline.yml", "--list-errors"])
    out = capsys.readouterr().out
    assert "[idx 3] /data/sub-01.vhdr" in out
    assert "error file: /out/sub-01@BasicPrep.error" in out


def test_status_list_missing_prints_file_paths(dummy_pipeline, capsys):
    cfg = dummy_pipeline["config"]
    fake_df = _status_df(
        [
            {
                "index": 5,
                "dataset": "ds",
                "derivative": "BasicPrep",
                "file_path": "/data/sub-02.vhdr",
                "plan": _make_plan(cached=False, has_error=False),
            },
        ]
    )
    with (
        patch("neurodags.cli._load_pipeline_config", return_value=cfg),
        patch("neurodags.cli.run_pipeline", return_value=fake_df),
    ):
        main(["status", "pipeline.yml", "--list-missing"])
    out = capsys.readouterr().out
    assert "[idx 5] /data/sub-02.vhdr" in out


def test_status_no_list_errors_hint_shown(dummy_pipeline, capsys):
    cfg = dummy_pipeline["config"]
    fake_df = _status_df(
        [
            {
                "derivative": "BasicPrep",
                "file_path": "x.vhdr",
                "plan": _make_plan(cached=False, has_error=True),
            },
        ]
    )
    with (
        patch("neurodags.cli._load_pipeline_config", return_value=cfg),
        patch("neurodags.cli.run_pipeline", return_value=fake_df),
    ):
        main(["status", "pipeline.yml"])
    out = capsys.readouterr().out
    assert "--list-errors" in out


def test_status_format_json(dummy_pipeline, capsys):
    import json

    cfg = dummy_pipeline["config"]
    fake_df = _status_df(
        [
            {
                "derivative": "BasicPrep",
                "file_path": "a.vhdr",
                "plan": _make_plan(cached=True, has_error=False),
            },
            {
                "derivative": "BasicPrep",
                "file_path": "b.vhdr",
                "plan": _make_plan(cached=False, has_error=False),
            },
        ]
    )
    with (
        patch("neurodags.cli._load_pipeline_config", return_value=cfg),
        patch("neurodags.cli.run_pipeline", return_value=fake_df),
    ):
        rc = main(["status", "pipeline.yml", "--format", "json"])
    out = capsys.readouterr().out
    data = json.loads(out)
    assert rc == 1  # missing → incomplete
    assert data["complete"] is False
    assert data["grand_total"]["missing"] == 1
    assert data["grand_total"]["done"] == 1
    assert "BasicPrep" in data["derivatives"]


def test_status_njobs_passed_to_run_pipeline(dummy_pipeline, capsys):
    cfg = dummy_pipeline["config"]
    fake_df = _status_df(
        [
            {
                "derivative": "BasicPrep",
                "file_path": "a.vhdr",
                "plan": _make_plan(cached=True, has_error=False),
            },
        ]
    )
    with (
        patch("neurodags.cli._load_pipeline_config", return_value=cfg),
        patch("neurodags.cli.run_pipeline", return_value=fake_df) as mock_run,
    ):
        main(["status", "pipeline.yml", "--n-jobs", "4"])
        _, kwargs = mock_run.call_args
        assert kwargs.get("n_jobs") == 4


def _indexed_status_df() -> pd.DataFrame:
    """Two derivatives, four files, mixed statuses, deliberately out of index order.

    Row labels (0..n) are unrelated to the global ``index`` column, as in the real
    dry-run dataframe (``pd.concat(..., ignore_index=True)``). Indices are numpy int64.
    """
    rows = [
        # BasicPrep: 9 errored, 4 missing, 1 missing, 2 done
        (
            "BasicPrep",
            9,
            "/d/i.vhdr",
            _make_plan(cached=False, has_error=True, error_path="/o/i.e"),
        ),
        ("BasicPrep", 4, "/d/e.vhdr", _make_plan(cached=False, has_error=False)),
        ("BasicPrep", 1, "/d/b.vhdr", _make_plan(cached=False, has_error=False)),
        ("BasicPrep", 2, "/d/c.vhdr", _make_plan(cached=True, has_error=False)),
        # Spectrum: 4 errored, 1 done
        ("Spectrum", 4, "/d/e.vhdr", _make_plan(cached=False, has_error=True, error_path="/o/e.e")),
        ("Spectrum", 1, "/d/b.vhdr", _make_plan(cached=True, has_error=False)),
    ]
    df = pd.DataFrame(
        [
            {"index": idx, "dataset": "ds", "derivative": d, "file_path": fp, "plan": plan}
            for d, idx, fp, plan in rows
        ]
    )
    df["index"] = df["index"].astype("int64")
    return df


def _run_status_with(dummy_pipeline, fake_df, argv):
    cfg = dummy_pipeline["config"]
    with (
        patch("neurodags.cli._load_pipeline_config", return_value=cfg),
        patch("neurodags.cli.run_pipeline", return_value=fake_df),
    ):
        return main(["status", "pipeline.yml", *argv])


def test_status_list_missing_shows_index_sorted(dummy_pipeline, capsys):
    rc = _run_status_with(dummy_pipeline, _indexed_status_df(), ["--list-missing"])
    out = capsys.readouterr().out
    assert rc == 1
    section = out[out.index("Missing files:") :]
    assert "  [BasicPrep]" in section
    lines = [ln.strip() for ln in section.splitlines() if ln.strip().startswith("[idx")]
    assert lines == ["[idx 1] /d/b.vhdr", "[idx 4] /d/e.vhdr"]
    # summary table unchanged
    assert "Total" in out


def test_status_list_errors_shows_index_and_error_file(dummy_pipeline, capsys):
    _run_status_with(dummy_pipeline, _indexed_status_df(), ["--list-errors"])
    out = capsys.readouterr().out
    section = out[out.index("Errored files:") :]
    assert section.index("[BasicPrep]") < section.index("[idx 9] /d/i.vhdr")
    assert section.index("[Spectrum]") < section.index("[idx 4] /d/e.vhdr")
    assert "error file: /o/i.e" in section
    assert "error file: /o/e.e" in section


def test_status_json_adds_index_lists_keeps_existing_keys(dummy_pipeline, capsys):
    rc = _run_status_with(dummy_pipeline, _indexed_status_df(), ["--format", "json"])
    data = json.loads(capsys.readouterr().out)  # also guards numpy int64 serialisation
    assert rc == 1
    assert set(data) == {
        "config",
        "n_files",
        "derivatives",
        "grand_total",
        "complete",
        "missing_indices",
        "incomplete_indices",
        "errored_indices",
        "unlisted_files",
        "inputs",
        "missing_source_indices",
        "not_in_list",
        "planning_error_indices",
    }
    assert data["missing_indices"] == [1, 4]
    assert data["errored_indices"] == [4, 9]
    assert all(type(i) is int for i in data["missing_indices"] + data["errored_indices"])
    for counts in data["derivatives"].values():
        assert set(counts) == {
            "total",
            "done",
            "skipped",
            "incomplete",
            "missing",
            "errored",
            "missing-source",
            "planning-error",
            "migrated",
        }
    assert data["derivatives"]["BasicPrep"] == {
        "total": 4,
        "done": 1,
        "skipped": 0,
        "incomplete": 0,
        "missing": 2,
        "errored": 1,
        "missing-source": 0,
        "planning-error": 0,
        "migrated": 0,
    }
    assert data["n_files"] == 4
    assert "files" not in data


def test_status_json_include_files(dummy_pipeline, capsys):
    _run_status_with(dummy_pipeline, _indexed_status_df(), ["--format", "json", "--include-files"])
    data = json.loads(capsys.readouterr().out)
    files = data["files"]
    assert len(files) == 6
    assert [(f["index"], f["derivative"]) for f in files] == [
        (1, "BasicPrep"),
        (1, "Spectrum"),
        (2, "BasicPrep"),
        (4, "BasicPrep"),
        (4, "Spectrum"),
        (9, "BasicPrep"),
    ]
    assert set(files[0]) == {
        "index",
        "dataset",
        "file_path",
        "derivative",
        "status",
        "label",
        "error_path",
        "unlisted",
        "planning_error",
    }
    rec = next(f for f in files if f["index"] == 4 and f["derivative"] == "Spectrum")
    assert rec == {
        "index": 4,
        "dataset": "ds",
        "file_path": "/d/e.vhdr",
        "derivative": "Spectrum",
        "status": "errored",
        "label": "errored",
        "error_path": "/o/e.e",
        "unlisted": [],
        "planning_error": None,
    }
    assert files[0]["error_path"] is None


def test_status_include_files_requires_json(dummy_pipeline, capsys):
    with (
        patch("neurodags.cli.run_pipeline") as mock_run,
        pytest.raises(SystemExit) as exc,
    ):
        main(["status", "pipeline.yml", "--include-files"])
    assert exc.value.code == 2
    mock_run.assert_not_called()
    assert "--include-files requires --format json" in capsys.readouterr().err


def test_status_format_indices_union_and_narrowing(dummy_pipeline, capsys):
    cases = [
        ([], "1 4 9"),
        (["--list-missing"], "1 4"),
        (["--list-errors"], "4 9"),
        (["--list-missing", "--list-errors"], "1 4 9"),
    ]
    for extra, expected in cases:
        rc = _run_status_with(dummy_pipeline, _indexed_status_df(), ["--format", "indices", *extra])
        out = capsys.readouterr().out
        assert out == expected + "\n", extra
        assert rc == 1


def test_status_format_indices_complete_prints_empty_line_exit_0(dummy_pipeline, capsys):
    df = pd.DataFrame(
        [
            {
                "index": 0,
                "dataset": "ds",
                "derivative": "BasicPrep",
                "file_path": "a.vhdr",
                "plan": _make_plan(cached=True, has_error=False),
            }
        ]
    )
    rc = _run_status_with(dummy_pipeline, df, ["--format", "indices"])
    assert rc == 0
    assert capsys.readouterr().out == "\n"


def test_status_empty_result_json_is_valid(dummy_pipeline, capsys):
    rc = _run_status_with(dummy_pipeline, pd.DataFrame(), ["--format", "json", "--include-files"])
    data = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert data["n_files"] == 0
    assert data["derivatives"] == {}
    assert data["grand_total"] == {
        "total": 0,
        "done": 0,
        "skipped": 0,
        "incomplete": 0,
        "missing": 0,
        "errored": 0,
        "missing-source": 0,
        "planning-error": 0,
        "migrated": 0,
    }
    assert data["complete"] is True
    assert data["missing_indices"] == []
    assert data["errored_indices"] == []
    assert data["files"] == []


def test_status_empty_result_indices_prints_empty_line(dummy_pipeline, capsys):
    rc = _run_status_with(dummy_pipeline, None, ["--format", "indices"])
    assert rc == 0
    assert capsys.readouterr().out == "\n"


def test_status_rows_without_index_column_do_not_crash(dummy_pipeline, capsys):
    df = _status_df(
        [
            {
                "derivative": "BasicPrep",
                "file_path": "/d/x.vhdr",
                "plan": _make_plan(cached=False, has_error=False),
            }
        ]
    )
    rc = _run_status_with(dummy_pipeline, df, ["--list-missing"])
    assert rc == 1
    assert "[idx ?] /d/x.vhdr" in capsys.readouterr().out

    _run_status_with(dummy_pipeline, df, ["--format", "json", "--include-files"])
    data = json.loads(capsys.readouterr().out)
    assert data["missing_indices"] == []
    assert data["files"][0]["index"] is None
    assert data["files"][0]["dataset"] is None


def test_status_index_round_trips_to_only_index(dummy_pipeline, tmp_path, capsys):
    """The index printed by `status` is the one `--only-index` consumes (#29)."""
    import yaml

    from neurodags.orchestrators import iterate_derivative_pipeline, run_pipeline

    cfg = dummy_pipeline["config"]
    pipeline_yml = tmp_path / "pipeline.yml"
    pipeline_yml.write_text(yaml.safe_dump(cfg))

    # Compute BasicPrep for file index 1 only.
    iterate_derivative_pipeline(cfg, "BasicPrep", only_index=1, raise_on_error=True)
    capsys.readouterr()

    status_argv = ["status", str(pipeline_yml), "--derivative", "BasicPrep"]
    rc = main([*status_argv, "--format", "indices"])
    assert rc == 1
    assert capsys.readouterr().out == "0\n"

    main([*status_argv, "--format", "json", "--include-files"])
    files = json.loads(capsys.readouterr().out)["files"]
    by_index = {f["index"]: f for f in files}
    assert by_index[0]["status"] == "missing"
    assert by_index[1]["status"] == "done"
    missing_path = by_index[0]["file_path"]

    # The text list shows the same index next to the same path.
    main([*status_argv, "--list-missing"])
    assert f"[idx 0] {missing_path}" in capsys.readouterr().out

    # Feeding that index to only_index targets exactly that file ...
    plan = run_pipeline(cfg, derivatives=["BasicPrep"], dry_run=True, only_index=[0])
    assert list(plan["file_path"]) == [missing_path]
    assert list(plan["index"]) == [0]

    # ... and running it completes the derivative.
    iterate_derivative_pipeline(cfg, "BasicPrep", only_index=0, raise_on_error=True)
    capsys.readouterr()
    assert main([*status_argv, "--format", "indices"]) == 0
    assert capsys.readouterr().out == "\n"


def test_cmd_run_passes_path_string_to_run_pipeline(dummy_pipeline):
    """_cmd_run must pass args.config (path str) not the loaded dict to run_pipeline.

    If the dict is passed instead, new_definitions and datasets relative paths
    in the YAML are resolved against cwd rather than the pipeline file's location.
    """
    cfg = dummy_pipeline["config"]
    with (
        patch("neurodags.cli._load_pipeline_config", return_value=cfg),
        patch("neurodags.cli.run_pipeline") as mock_run,
    ):
        mock_run.return_value = None
        main(["run", "pipeline.yml", "--derivative", "BasicPrep"])
        called_config = mock_run.call_args.kwargs.get(
            "pipeline_configuration",
            mock_run.call_args.args[0] if mock_run.call_args.args else None,
        )
        assert (
            called_config == "pipeline.yml"
        ), "run_pipeline must receive the path string, not the parsed dict"


# ---------------------------------------------------------------------------
# --log-level / --log-file global flags
# ---------------------------------------------------------------------------


def test_log_level_flag_passed_to_configure_logging(dummy_pipeline, capsys):
    cfg = dummy_pipeline["config"]
    with (
        patch("neurodags.cli._load_pipeline_config", return_value=cfg),
        patch("neurodags.cli.run_pipeline", return_value=pd.DataFrame()),
        patch("neurodags.cli.configure_logging") as mock_log,
        patch("neurodags.cli._cli_enumeration", return_value=InputEnumeration(entries=[])),
    ):
        assert main(["--log-level", "WARNING", "count-inputs", "pipeline.yml"]) == 0
        mock_log.assert_called_once()
        _, kwargs = mock_log.call_args
        assert kwargs.get("level") == "WARNING"


def test_log_file_flag_passed_to_configure_logging(dummy_pipeline, tmp_path):
    cfg = dummy_pipeline["config"]
    log_path = str(tmp_path / "run.jsonl")
    with (
        patch("neurodags.cli._load_pipeline_config", return_value=cfg),
        patch("neurodags.cli.run_pipeline", return_value=pd.DataFrame()),
        patch("neurodags.cli.configure_logging") as mock_log,
        patch("neurodags.cli._cli_enumeration", return_value=InputEnumeration(entries=[])),
    ):
        assert main(["--log-file", log_path, "count-inputs", "pipeline.yml"]) == 0
        _, kwargs = mock_log.call_args
        assert kwargs.get("log_file") == log_path


# ---------------------------------------------------------------------------
# -d/--datasets path resolution (#43)
# ---------------------------------------------------------------------------


def _write_nested_pipeline(root: Path) -> tuple[Path, Path]:
    """``root/p/x/pipeline.yml`` + ``root/p/x/alt.yml`` (dataset name ``alt``)."""
    pdir = root / "p" / "x"
    pdir.mkdir(parents=True)
    pipe = pdir / "pipeline.yml"
    pipe.write_text(
        "naming_version: 1\n"
        "datasets: alt.yml\n"
        "mount_point: null\n"
        "DerivativeDefinitions:\n"
        "  BasicPrep: {}\n"
        "DerivativeList:\n"
        "  - BasicPrep\n"
    )
    alt = pdir / "alt.yml"
    alt.write_text(
        "alt:\n"
        "  name: Alt\n"
        f"  source_root: {root}\n"
        f"  file_pattern: {root}/**/*.vhdr\n"
        f"  derivatives_path: {root}/derivatives\n"
    )
    return pipe, alt


def test_datasets_path_is_cwd_relative(tmp_path, monkeypatch, capsys):
    _write_nested_pipeline(tmp_path)
    monkeypatch.chdir(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)  # the cwd-relative form must not warn
        rc = main(["validate", "p/x/pipeline.yml", "-d", "p/x/alt.yml"])
    assert rc == 0
    assert "datasets: alt" in capsys.readouterr().out


def test_datasets_path_expands_env_vars(tmp_path, monkeypatch, capsys):
    _write_nested_pipeline(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ND_TEST_DIR", str(tmp_path / "p"))
    assert main(["validate", "p/x/pipeline.yml", "-d", "$ND_TEST_DIR/x/alt.yml"]) == 0
    assert "datasets: alt" in capsys.readouterr().out


def test_datasets_path_yaml_relative_fallback_removed(tmp_path, monkeypatch, capsys):
    # 0.4.0: the YAML-relative -d lookup deprecated in 0.3.5 is gone; -d is cwd-relative.
    _write_nested_pipeline(tmp_path)
    monkeypatch.chdir(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)  # no deprecation path left
        rc = main(["validate", "p/x/pipeline.yml", "-d", "alt.yml"])
    err = capsys.readouterr().err
    assert rc == 2
    assert "Traceback" not in err
    assert str(tmp_path / "alt.yml") in err
    assert str(tmp_path / "p" / "x" / "alt.yml") not in err


def test_resolve_cli_datasets_path_is_cwd_relative_only(tmp_path, monkeypatch):
    from neurodags.cli import DatasetsPathError

    _, alt = _write_nested_pipeline(tmp_path)
    monkeypatch.chdir(tmp_path)
    assert resolve_cli_datasets_path("p/x/alt.yml") == str(alt)
    with pytest.raises(DatasetsPathError, match="not found"):
        resolve_cli_datasets_path("alt.yml")


def test_datasets_path_fallback_removed_subprocess_is_one_clean_line(tmp_path):
    _write_nested_pipeline(tmp_path)
    result = subprocess.run(
        [sys.executable, "-m", "neurodags.cli", "validate", "p/x/pipeline.yml", "-d", "alt.yml"],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert result.returncode == 2
    lines = result.stderr.splitlines()
    assert len(lines) == 1
    assert lines[0].startswith("neurodags: error: datasets file 'alt.yml' not found")


def test_datasets_path_cwd_wins_over_yaml_relative(tmp_path, monkeypatch):
    _write_nested_pipeline(tmp_path)
    (tmp_path / "alt.yml").write_text("{}\n")  # same name, in the cwd
    monkeypatch.chdir(tmp_path)
    with patch("neurodags.cli._cmd_validate", return_value=0) as cmd:
        assert main(["validate", "p/x/pipeline.yml", "-d", "alt.yml"]) == 0
    assert cmd.call_args.args[0].datasets == str(tmp_path / "alt.yml")


def test_datasets_path_missing_is_clean_error(tmp_path, monkeypatch, capsys):
    _write_nested_pipeline(tmp_path)
    monkeypatch.chdir(tmp_path)
    rc = main(["validate", "p/x/pipeline.yml", "-d", "nope.yml"])
    err = capsys.readouterr().err
    assert rc == 2
    assert "Traceback" not in err
    assert str(tmp_path / "nope.yml") in err
    assert str(tmp_path / "p" / "x" / "nope.yml") not in err  # no YAML-relative lookup


def test_datasets_path_unset_env_var_is_clean_error(tmp_path, monkeypatch, capsys):
    _write_nested_pipeline(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ND_TEST_UNSET", raising=False)
    rc = main(["status", "p/x/pipeline.yml", "-d", "$ND_TEST_UNSET/alt.yml", "--exit-zero"])
    assert rc == 2
    assert "ND_TEST_UNSET" in capsys.readouterr().err


def test_datasets_path_missing_subprocess_has_no_traceback(tmp_path):
    _write_nested_pipeline(tmp_path)
    result = subprocess.run(
        [sys.executable, "-m", "neurodags.cli", "run", "p/x/pipeline.yml", "-d", "nope.yml"],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert result.returncode == 2
    assert "Traceback" not in result.stderr
    assert "not found" in result.stderr


@pytest.mark.parametrize(
    "argv",
    [
        ["run"],
        ["dry-run", "--output", "out.csv"],
        ["count-inputs"],
        ["status"],
        ["dataframe", "--output", "out.csv"],
    ],
)
def test_datasets_path_resolved_for_every_subcommand(tmp_path, monkeypatch, argv):
    _, alt = _write_nested_pipeline(tmp_path)
    monkeypatch.chdir(tmp_path)

    seen: list = []

    def fake_enumeration(args):
        seen.append(args.datasets)
        return InputEnumeration(entries=[])

    with (
        patch("neurodags.cli.run_pipeline", return_value=pd.DataFrame()) as rp,
        patch("neurodags.cli.build_derivative_dataframe", return_value=pd.DataFrame()) as bd,
        patch("neurodags.cli._cli_enumeration", side_effect=fake_enumeration),
    ):
        main([argv[0], "p/x/pipeline.yml", "-d", "p/x/alt.yml", *argv[1:]])
    if argv[0] == "count-inputs":  # counts the enumeration, no dry run
        assert seen == [str(alt)]
        return
    called = bd if argv[0] == "dataframe" else rp
    assert called.call_args.kwargs["datasets_configuration"] == str(alt)


# ---------------------------------------------------------------------------
# status --exit-zero (#45)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fmt", ["table", "json", "indices"])
def test_status_exit_zero_with_missing(dummy_pipeline, capsys, fmt):
    df = _indexed_status_df()
    assert _run_status_with(dummy_pipeline, df, ["--format", fmt]) == 1  # default unchanged
    assert _run_status_with(dummy_pipeline, df, ["--format", fmt, "--exit-zero"]) == 0
    if fmt == "indices":
        assert capsys.readouterr().out.splitlines()[-1] == "1 4 9"


def test_status_exit_zero_bad_config_still_fails(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "neurodags.cli", "status", "missing.yml", "--exit-zero"],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert result.returncode != 0
    assert result.stdout == ""
