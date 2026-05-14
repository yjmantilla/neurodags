"""Tests for the top-level neurodags CLI."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd

from neurodags.cli import main


def test_tui_subcommand_launches_app():
    with (
        patch("neurodags.cli.NeuroDagsApp", create=True),
        patch("neurodags.cli._cmd_tui") as cmd,
    ):
        cmd.return_value = 0
        assert main(["tui", "pipeline.yml", "--datasets", "datasets.yml"]) == 0


def test_validate_prints_summary(tmp_path, capsys):
    pipe = tmp_path / "pipeline.yml"
    data = tmp_path / "datasets.yml"
    pipe.write_text(
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
        f"  file_pattern: {tmp_path}/**/*.vhdr\n"
        f"  derivatives_path: {tmp_path}/derivatives\n"
    )

    assert main(["validate", str(pipe)]) == 0
    out = capsys.readouterr().out
    assert "datasets: demo" in out
    assert "derivatives_enabled: BasicPrep" in out


def test_run_uses_derivative_list_when_not_explicit(dummy_pipeline):
    cfg = dummy_pipeline["config"]
    with (
        patch("neurodags.cli._load_pipeline_config", return_value=cfg),
        patch("neurodags.orchestrators.iterate_derivative_pipeline") as iterate,
    ):
        assert main(["run", "pipeline.yml"]) == 0
        called = [call.kwargs["derivative"] for call in iterate.call_args_list]
        assert set(called) == set(cfg["DerivativeList"])


def test_run_with_explicit_derivative(dummy_pipeline):
    cfg = dummy_pipeline["config"]
    with (
        patch("neurodags.cli._load_pipeline_config", return_value=cfg),
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


def test_view_launches_visualization_subprocess():
    with patch("neurodags.cli.subprocess.Popen") as popen:
        assert main(["view", "result.nc"]) == 0
        popen.assert_called_once()
