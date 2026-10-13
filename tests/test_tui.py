"""Tests for the Textual TUI (neurodags.tui)."""

from __future__ import annotations

import asyncio
import io
import sys
from unittest.mock import patch

import pandas as pd
import pytest

pytest.importorskip("textual", reason="textual not installed; skip TUI tests")

from textual.widgets import DataTable, Input, Select, Static, TabbedContent, TextArea

from neurodags.tui import (
    NeuroDagsApp,
    _InspectableStatic,
    _parse_int,
    _run_pipeline_sync,
    _status_sync,
    main,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FAKE_CONFIG: dict = {
    "Datasets": {"ds1": {"file_pattern": "*.vhdr"}},
    "DerivativeDefinitions": {"StepA": {}, "StepB": {}},
}


def _run(coro):  # type: ignore[return]
    return asyncio.run(coro)


async def _switch_tab(pilot, tab_id: str) -> None:
    pilot.app.query_one(TabbedContent).active = tab_id
    await pilot.pause()
    await pilot.pause()


# ---------------------------------------------------------------------------
# Pure-function unit tests
# ---------------------------------------------------------------------------


class TestParseInt:
    def test_valid_positive(self):
        assert _parse_int("3") == 3

    def test_valid_negative(self):
        assert _parse_int("-1") == -1

    def test_blank(self):
        assert _parse_int("") is None

    def test_non_numeric(self):
        assert _parse_int("abc") is None

    def test_whitespace_trimmed(self):
        assert _parse_int("  5  ") == 5

    def test_zero(self):
        assert _parse_int("0") == 0


class TestRunPipelineSync:
    def test_captures_stdout(self):
        def fake_run(*_a, **_kw):
            print("hello from pipeline")

        buf = io.StringIO()
        with patch("neurodags.orchestrators.run_pipeline", fake_run):
            _run_pipeline_sync({}, None, None, None, None, False, buf)
        assert "hello from pipeline" in buf.getvalue()

    def test_captures_stderr(self):
        def fake_run(*_a, **_kw):
            import sys

            print("err line", file=sys.stderr)

        buf = io.StringIO()
        with patch("neurodags.orchestrators.run_pipeline", fake_run):
            _run_pipeline_sync({}, None, None, None, None, False, buf)
        assert "err line" in buf.getvalue()

    def test_propagates_exception(self):
        def bad_run(*_a, **_kw):
            raise ValueError("boom")

        buf = io.StringIO()
        with patch("neurodags.orchestrators.run_pipeline", bad_run):
            with pytest.raises(ValueError, match="boom"):
                _run_pipeline_sync({}, None, None, None, None, False, buf)

    def test_supports_datasets_config_argument(self):
        captured = {}

        def fake_run(*_a, **_kw):
            captured.update(_kw)

        with patch("neurodags.orchestrators.run_pipeline", fake_run):
            _run_pipeline_sync({}, ["D"], "datasets.yml", 3, 2, True, io.StringIO())
        assert captured["datasets_configuration"] == "datasets.yml"
        assert captured["max_files_per_dataset"] == 3
        assert captured["n_jobs"] == 2
        assert captured["skip_errors"] is True


class TestInspectableStatic:
    def test_render_returns_original_content(self):
        widget = _InspectableStatic("summary text")
        assert widget.render() == "summary text"


# ---------------------------------------------------------------------------
# App startup
# ---------------------------------------------------------------------------


class TestAppStartup:
    def test_starts_without_config(self):
        async def _():
            async with NeuroDagsApp().run_test() as pilot:
                assert pilot.app.query_one("#config-path-input", Input) is not None

        _run(_())

    def test_starts_with_config_path_preloads(self, tmp_path):
        yaml = tmp_path / "pipe.yaml"
        yaml.write_text("naming_version: 1\nDatasets: {}\nDerivativeDefinitions: {StepA: {}}\n")

        async def _():
            async with NeuroDagsApp(config_path=str(yaml)).run_test() as pilot:
                inp = pilot.app.query_one("#config-path-input", Input)
                assert str(yaml) in inp.value
                assert pilot.app._config is not None

        _run(_())

    def test_on_mount_skips_load_when_no_path(self):
        async def _():
            with patch.object(NeuroDagsApp, "_load_config") as m:
                async with NeuroDagsApp().run_test():
                    m.assert_not_called()

        _run(_())

    def test_on_mount_calls_load_when_path_given(self, tmp_path):
        yaml = tmp_path / "p.yaml"
        yaml.write_text("Datasets: {}\nDerivativeDefinitions: {}\n")

        async def _():
            with patch.object(NeuroDagsApp, "_load_config", wraps=None) as m:
                async with NeuroDagsApp(config_path=str(yaml)).run_test():
                    m.assert_called_once_with(str(yaml))

        _run(_())

    def test_starts_with_datasets_path_preloads(self, tmp_path):
        yaml = tmp_path / "pipe.yaml"
        data_yaml = tmp_path / "datasets.yaml"
        yaml.write_text("naming_version: 1\nDatasets: {}\nDerivativeDefinitions: {StepA: {}}\n")
        data_yaml.write_text("Datasets: {}\n")

        async def _():
            async with NeuroDagsApp(
                config_path=str(yaml), datasets_path=str(data_yaml)
            ).run_test() as pilot:
                inp = pilot.app.query_one("#datasets-path-input", Input)
                assert str(data_yaml) in inp.value

        _run(_())


# ---------------------------------------------------------------------------
# Config tab
# ---------------------------------------------------------------------------


class TestConfigTab:
    def test_load_config_with_datasets_override(self, tmp_path):
        pipe_yaml = tmp_path / "p.yaml"
        pipe_yaml.write_text("naming_version: 1\nDerivativeDefinitions:\n  StepA: {}\n")
        data_yaml = tmp_path / "d.yaml"
        # A real datasets YAML (the summary uses the loader execution uses).
        data_yaml.write_text("ds1:\n  name: ds1\n  source_root: .\n  file_pattern: '*.vhdr'\n")

        async def _():
            async with NeuroDagsApp().run_test() as pilot:
                app = pilot.app
                app.query_one("#config-path-input", Input).value = str(pipe_yaml)
                app.query_one("#datasets-path-input", Input).value = str(data_yaml)
                await pilot.click("#btn-load-config")
                await pilot.pause()
                assert app._config is not None
                assert app._datasets_path == str(data_yaml)
                summary = str(app.query_one("#config-summary", Static).render())
                assert "ds1" in summary

        _run(_())

    def test_load_config_bad_path_shows_error(self):
        async def _():
            async with NeuroDagsApp().run_test() as pilot:
                app = pilot.app
                app.query_one("#config-path-input", Input).value = "/nonexistent/file.yaml"
                await pilot.click("#btn-load-config")
                await pilot.pause()
                status = str(app.query_one("#config-status", Static).render())
                assert "Error" in status or app._config is None

        _run(_())

    def test_load_config_empty_path_does_nothing(self):
        async def _():
            async with NeuroDagsApp().run_test() as pilot:
                app = pilot.app
                app.query_one("#config-path-input", Input).value = ""
                with patch.object(app, "_load_config") as m:
                    await pilot.click("#btn-load-config")
                    await pilot.pause()
                    m.assert_not_called()

        _run(_())

    def test_sync_derivative_selects_populates_both(self):
        async def _():
            async with NeuroDagsApp().run_test() as pilot:
                app = pilot.app
                app._config = FAKE_CONFIG
                app._derivatives = ["StepA", "StepB"]
                app._sync_derivative_selects()
                sel = app.query_one("#dryrun-derivative", Select)
                opts = [v for _, v in sel._options]  # type: ignore[attr-defined]
                assert "StepA" in opts
                assert "StepB" in opts

        _run(_())


# ---------------------------------------------------------------------------
# DAG tab
# ---------------------------------------------------------------------------


class TestDagTab:
    def test_refresh_without_config_shows_fallback(self):
        async def _():
            async with NeuroDagsApp().run_test() as pilot:
                await _switch_tab(pilot, "tab-dag")
                await pilot.click("#btn-dag-refresh")
                await pilot.pause()
                text = pilot.app.query_one("#dag-mermaid", TextArea).text
                assert "Load config" in text

        _run(_())

    def test_refresh_with_config_shows_mermaid(self):
        async def _():
            async with NeuroDagsApp().run_test() as pilot:
                app = pilot.app
                app._config = FAKE_CONFIG
                await _switch_tab(pilot, "tab-dag")
                with patch(
                    "neurodags.mermaid.pipeline_to_mermaid", return_value="graph LR\n  A-->B"
                ):
                    await pilot.click("#btn-dag-refresh")
                    await pilot.pause()
                text = app.query_one("#dag-mermaid", TextArea).text
                assert text != "Load config first."

        _run(_())

    def test_open_html_without_config_notifies(self):
        async def _():
            async with NeuroDagsApp().run_test() as pilot:
                app = pilot.app
                # config is None → should call notify with error
                with patch.object(app, "notify") as m:
                    app._open_dag_html()
                    m.assert_called_once()
                    assert "error" in str(m.call_args)

        _run(_())

    def test_open_html_with_config_calls_browser(self, tmp_path):
        async def _():
            async with NeuroDagsApp().run_test() as pilot:
                app = pilot.app
                app._config = FAKE_CONFIG
                app._config_path = str(tmp_path / "p.yaml")
                fake_path = tmp_path / "pipeline_dag_tui.html"

                with (
                    patch("neurodags.mermaid.pipeline_to_html", return_value=fake_path) as mph,
                    patch("neurodags.tui.webbrowser.open") as mwb,
                ):
                    app._open_dag_html()
                    mph.assert_called_once()
                    mwb.assert_called_once()

        _run(_())

    def test_refresh_dag_handles_exception(self):
        async def _():
            async with NeuroDagsApp().run_test() as pilot:
                app = pilot.app
                app._config = FAKE_CONFIG
                with patch(
                    "neurodags.mermaid.pipeline_to_mermaid", side_effect=RuntimeError("bad")
                ):
                    app._refresh_dag()
                    text = app.query_one("#dag-mermaid", TextArea).text
                    assert "Error" in text

        _run(_())


# ---------------------------------------------------------------------------
# Dry Run tab
# ---------------------------------------------------------------------------


class TestDryRunTab:
    def test_dry_run_without_config_notifies_error(self):
        async def _():
            async with NeuroDagsApp().run_test() as pilot:
                app = pilot.app
                with patch.object(app, "notify") as m:
                    await app._run_dry_run()
                    m.assert_called_once()
                    assert "error" in str(m.call_args)

        _run(_())

    def test_dry_run_blank_derivative_runs_all(self):
        async def _():
            async with NeuroDagsApp().run_test() as pilot:
                app = pilot.app
                app._config = FAKE_CONFIG
                app._derivatives = ["StepA"]
                # blank select = run all derivatives (no warning, no error)
                with patch("neurodags.orchestrators.run_pipeline", return_value=None):
                    with patch.object(app, "notify") as m:
                        await app._run_dry_run()
                        notified = str(m.call_args)
                        assert "error" not in notified

        _run(_())

    def test_dry_run_populates_table(self):
        fake_df = pd.DataFrame({"file": ["a.vhdr", "b.vhdr"], "status": ["planned", "planned"]})

        async def _():
            async with NeuroDagsApp().run_test() as pilot:
                app = pilot.app
                app._config = FAKE_CONFIG
                app._derivatives = ["StepA"]
                app._sync_derivative_selects()
                # pick first option
                sel = app.query_one("#dryrun-derivative", Select)
                sel.value = "StepA"
                await pilot.pause()

                with patch(
                    "neurodags.orchestrators.iterate_derivative_pipeline", return_value=fake_df
                ):
                    await app._run_dry_run()

                table = app.query_one("#dryrun-table", DataTable)
                assert table.row_count == 2

        _run(_())

    def test_dry_run_empty_result_warns(self):
        async def _():
            async with NeuroDagsApp().run_test() as pilot:
                app = pilot.app
                app._config = FAKE_CONFIG
                app._derivatives = ["StepA"]
                app._sync_derivative_selects()
                sel = app.query_one("#dryrun-derivative", Select)
                sel.value = "StepA"
                await pilot.pause()

                with (
                    patch("neurodags.orchestrators.iterate_derivative_pipeline", return_value=None),
                    patch.object(app, "notify") as m,
                ):
                    await app._run_dry_run()
                    assert "warning" in str(m.call_args)

        _run(_())

    def test_dry_run_exception_notifies_error(self):
        async def _():
            async with NeuroDagsApp().run_test() as pilot:
                app = pilot.app
                app._config = FAKE_CONFIG
                app._derivatives = ["StepA"]
                app._sync_derivative_selects()
                sel = app.query_one("#dryrun-derivative", Select)
                sel.value = "StepA"
                await pilot.pause()

                with (
                    patch(
                        "neurodags.orchestrators.iterate_derivative_pipeline",
                        side_effect=RuntimeError("fail"),
                    ),
                    patch.object(app, "notify") as m,
                ):
                    await app._run_dry_run()
                    assert "error" in str(m.call_args)

        _run(_())


# ---------------------------------------------------------------------------
# Run Pipeline tab
# ---------------------------------------------------------------------------


class TestRunPipelineTab:
    def test_run_without_config_notifies_error(self):
        async def _():
            async with NeuroDagsApp().run_test() as pilot:
                app = pilot.app
                with patch.object(app, "notify") as m:
                    await app._run_pipeline()
                    m.assert_called_once()
                    assert "error" in str(m.call_args)

        _run(_())

    def test_run_blank_derivative_runs_all(self):
        async def _():
            async with NeuroDagsApp().run_test() as pilot:
                app = pilot.app
                app._config = FAKE_CONFIG
                # blank select = run all derivatives (no warning, no error)
                with patch("neurodags.tui._run_pipeline_sync"):
                    with patch.object(app, "notify") as m:
                        await app._run_pipeline()
                        notified = str(m.call_args)
                        assert "error" not in notified

        _run(_())

    def test_run_success_writes_log(self):
        async def _():
            async with NeuroDagsApp().run_test() as pilot:
                app = pilot.app
                app._config = FAKE_CONFIG
                app._derivatives = ["StepA"]
                app._sync_derivative_selects()
                sel = app.query_one("#run-derivative", Select)
                sel.value = "StepA"
                await pilot.pause()

                def fake_func(*_a, **_kw):
                    print("running step")

                with patch("neurodags.tui._run_pipeline_sync") as m:
                    m.side_effect = lambda cfg, derivs, ds, mf, nj, se, buf: fake_func()
                    await app._run_pipeline()

        _run(_())

    def test_run_exception_notifies_error(self):
        async def _():
            async with NeuroDagsApp().run_test() as pilot:
                app = pilot.app
                app._config = FAKE_CONFIG
                app._derivatives = ["StepA"]
                app._sync_derivative_selects()
                sel = app.query_one("#run-derivative", Select)
                sel.value = "StepA"
                await pilot.pause()

                with (
                    patch("neurodags.tui._run_pipeline_sync", side_effect=RuntimeError("crash")),
                    patch.object(app, "notify") as m,
                ):
                    await app._run_pipeline()
                    assert "error" in str(m.call_args)

        _run(_())


# ---------------------------------------------------------------------------
# DataFrame tab
# ---------------------------------------------------------------------------


class TestDataFrameTab:
    def test_assemble_without_config_notifies_error(self):
        async def _():
            async with NeuroDagsApp().run_test() as pilot:
                app = pilot.app
                with patch.object(app, "notify") as m:
                    await app._assemble_dataframe()
                    m.assert_called_once()
                    assert "error" in str(m.call_args)

        _run(_())

    def test_assemble_populates_table(self):
        fake_df = pd.DataFrame({"subject": ["s1", "s2"], "val": [1.0, 2.0]})

        async def _():
            async with NeuroDagsApp().run_test() as pilot:
                app = pilot.app
                app._config = FAKE_CONFIG

                with patch(
                    "neurodags.orchestrators.build_derivative_dataframe", return_value=fake_df
                ):
                    await app._assemble_dataframe()

                table = app.query_one("#df-table", DataTable)
                assert table.row_count == 2

        _run(_())

    def test_assemble_empty_df_warns(self):
        async def _():
            async with NeuroDagsApp().run_test() as pilot:
                app = pilot.app
                app._config = FAKE_CONFIG

                with (
                    patch(
                        "neurodags.orchestrators.build_derivative_dataframe",
                        return_value=pd.DataFrame(),
                    ),
                    patch.object(app, "notify") as m,
                ):
                    await app._assemble_dataframe()
                    assert "warning" in str(m.call_args)

        _run(_())

    def test_assemble_exception_notifies_error(self):
        async def _():
            async with NeuroDagsApp().run_test() as pilot:
                app = pilot.app
                app._config = FAKE_CONFIG

                with (
                    patch(
                        "neurodags.orchestrators.build_derivative_dataframe",
                        side_effect=RuntimeError("bad"),
                    ),
                    patch.object(app, "notify") as m,
                ):
                    await app._assemble_dataframe()
                    assert "error" in str(m.call_args)

        _run(_())

    def test_assemble_passes_format_and_include(self):
        fake_df = pd.DataFrame({"a": [1]})

        async def _():
            async with NeuroDagsApp().run_test() as pilot:
                app = pilot.app
                app._config = FAKE_CONFIG
                app.query_one("#df-derivatives", Input).value = "StepA, StepB"
                app.query_one("#df-format", Select).value = "long"
                await pilot.pause()

                with patch(
                    "neurodags.orchestrators.build_derivative_dataframe", return_value=fake_df
                ) as m:
                    await app._assemble_dataframe()
                    _, kwargs = m.call_args
                    assert kwargs.get("include_derivatives") == ["StepA", "StepB"]
                    assert kwargs.get("output_format") == "long"

        _run(_())


# ---------------------------------------------------------------------------
# NC Viewer tab
# ---------------------------------------------------------------------------


class TestNcTab:
    def test_launch_empty_path_warns(self):
        async def _():
            async with NeuroDagsApp().run_test() as pilot:
                app = pilot.app
                with patch.object(app, "notify") as m:
                    app._launch_nc_viewer()
                    m.assert_called_once()
                    assert "warning" in str(m.call_args)

        _run(_())

    def test_launch_missing_file_notifies_error(self, tmp_path):
        async def _():
            async with NeuroDagsApp().run_test() as pilot:
                app = pilot.app
                app.query_one("#nc-path", Input).value = str(tmp_path / "no.nc")
                with patch.object(app, "notify") as m:
                    app._launch_nc_viewer()
                    m.assert_called_once()
                    assert "error" in str(m.call_args)

        _run(_())

    def test_launch_existing_file_starts_subprocess(self, tmp_path):
        nc = tmp_path / "test.nc"
        nc.write_bytes(b"fake")

        async def _():
            async with NeuroDagsApp().run_test() as pilot:
                app = pilot.app
                app.query_one("#nc-path", Input).value = str(nc)

                with patch("neurodags.tui.subprocess.Popen") as mp:
                    app._launch_nc_viewer()
                    mp.assert_called_once()
                    args = mp.call_args[0][0]
                    assert args[:3] == [sys.executable, "-m", "neurodags.visualization"]
                    assert str(nc) == args[3]

                status = str(app.query_one("#nc-status", Static).render())
                assert "Launched" in status or "127.0.0.1" in status

        _run(_())

    def test_launch_popen_exception_notifies_error(self, tmp_path):
        nc = tmp_path / "test.nc"
        nc.write_bytes(b"fake")

        async def _():
            async with NeuroDagsApp().run_test() as pilot:
                app = pilot.app
                app.query_one("#nc-path", Input).value = str(nc)

                with (
                    patch("neurodags.tui.subprocess.Popen", side_effect=OSError("no exec")),
                    patch.object(app, "notify") as m,
                ):
                    app._launch_nc_viewer()
                    assert "error" in str(m.call_args)

        _run(_())


# ---------------------------------------------------------------------------
# Status tab
# ---------------------------------------------------------------------------


def _make_status_plan(*, cached: bool, has_error: bool, error_path: str | None = None) -> list:
    return [
        {
            "id": "final",
            "kind": "derivative_output",
            "name": "StepA",
            "prefix": "/out/file@StepA",
            "cached": cached,
            "paths": ["/out/file@StepA.fif"] if cached else [],
            "has_error_marker": has_error,
            "error_path": error_path,
        }
    ]


class TestStatusTab:
    def test_status_without_config_notifies_error(self):
        async def _():
            async with NeuroDagsApp().run_test() as pilot:
                app = pilot.app
                with patch.object(app, "notify") as m:
                    await app._run_status()
                    m.assert_called_once()
                    assert "error" in str(m.call_args)

        _run(_())

    def test_status_populates_table(self):
        rows = [
            {
                "derivative": "StepA",
                "file_path": "a.vhdr",
                "plan": _make_status_plan(cached=True, has_error=False),
            },
            {
                "derivative": "StepA",
                "file_path": "b.vhdr",
                "plan": _make_status_plan(cached=False, has_error=False),
            },
        ]
        fake_df = pd.DataFrame(rows)

        async def _():
            async with NeuroDagsApp().run_test() as pilot:
                app = pilot.app
                app._config = FAKE_CONFIG
                app._derivatives = ["StepA"]

                with patch("neurodags.orchestrators.run_pipeline", return_value=fake_df):
                    await app._run_status()

                table = app.query_one("#status-table", DataTable)
                # one data row + grand-total row
                assert table.row_count == 2

        _run(_())

    def test_status_shows_errored_files(self):
        rows = [
            {
                "derivative": "StepA",
                "file_path": "/data/bad.vhdr",
                "plan": _make_status_plan(
                    cached=False, has_error=True, error_path="/out/bad@StepA.error"
                ),
            },
        ]
        fake_df = pd.DataFrame(rows)

        async def _():
            async with NeuroDagsApp().run_test() as pilot:
                app = pilot.app
                app._config = FAKE_CONFIG
                app._derivatives = ["StepA"]

                with patch("neurodags.orchestrators.run_pipeline", return_value=fake_df):
                    await app._run_status()

                errors = str(app.query_one("#status-errors", Static).render())
                assert "/data/bad.vhdr" in errors
                assert "/out/bad@StepA.error" in errors

        _run(_())

    def test_status_no_files_warns(self):
        async def _():
            async with NeuroDagsApp().run_test() as pilot:
                app = pilot.app
                app._config = FAKE_CONFIG

                with (
                    patch("neurodags.orchestrators.run_pipeline", return_value=pd.DataFrame()),
                    patch.object(app, "notify") as m,
                ):
                    await app._run_status()
                    assert "warning" in str(m.call_args)

        _run(_())

    def test_status_exception_notifies_error(self):
        async def _():
            async with NeuroDagsApp().run_test() as pilot:
                app = pilot.app
                app._config = FAKE_CONFIG

                with (
                    patch("neurodags.orchestrators.run_pipeline", side_effect=RuntimeError("boom")),
                    patch.object(app, "notify") as m,
                ):
                    await app._run_status()
                    assert "error" in str(m.call_args)

        _run(_())


class TestStatusSync:
    def test_returns_empty_on_no_files(self):
        with patch("neurodags.orchestrators.run_pipeline", return_value=pd.DataFrame()):
            rows, grand, errored = _status_sync("p.yml", None, None)
        assert rows == []
        assert grand == {"total": 0, "done": 0, "incomplete": 0, "missing": 0, "errored": 0}
        assert errored == []

    def test_classifies_done_missing_errored(self):
        fake_df = pd.DataFrame(
            [
                {
                    "derivative": "StepA",
                    "file_path": "a.vhdr",
                    "plan": _make_status_plan(cached=True, has_error=False),
                },
                {
                    "derivative": "StepA",
                    "file_path": "b.vhdr",
                    "plan": _make_status_plan(cached=False, has_error=False),
                },
                {
                    "derivative": "StepA",
                    "file_path": "c.vhdr",
                    "plan": _make_status_plan(cached=False, has_error=True, error_path="c.error"),
                },
            ]
        )
        with patch("neurodags.orchestrators.run_pipeline", return_value=fake_df):
            _rows, grand, errored = _status_sync("p.yml", None, None)
        assert grand == {"total": 3, "done": 1, "incomplete": 0, "missing": 1, "errored": 1}
        assert len(errored) == 1
        assert errored[0] == ("StepA", "c.vhdr", "c.error")

    def test_derivative_filter_passed_through(self):
        with patch("neurodags.orchestrators.run_pipeline", return_value=pd.DataFrame()) as mock_run:
            _status_sync("p.yml", None, ["StepA"])
            _, kwargs = mock_run.call_args
            assert kwargs.get("derivatives") == ["StepA"]


# ---------------------------------------------------------------------------
# main() CLI entry point
# ---------------------------------------------------------------------------


class TestMain:
    def test_main_no_args(self):
        with (
            patch.object(sys, "argv", ["neurodags-tui"]),
            patch("neurodags.tui.NeuroDagsApp.run") as m,
        ):
            main()
            m.assert_called_once()

    def test_main_with_config_arg(self, tmp_path):
        yaml = tmp_path / "p.yaml"
        yaml.write_text("naming_version: 1\nDatasets: {}\n")

        with (
            patch.object(sys, "argv", ["neurodags-tui", str(yaml)]),
            patch("neurodags.tui.NeuroDagsApp.run") as m,
            patch("neurodags.tui.NeuroDagsApp.__init__", return_value=None) as mi,
        ):
            main()
            mi.assert_called_once_with(config_path=str(yaml))
            m.assert_called_once()

    def test_main_with_datasets_arg(self, tmp_path):
        yaml = tmp_path / "p.yaml"
        datasets = tmp_path / "d.yaml"
        yaml.write_text("naming_version: 1\nDatasets: {}\n")
        datasets.write_text("Datasets: {}\n")

        with (
            patch.object(sys, "argv", ["neurodags-tui", str(yaml), "--datasets", str(datasets)]),
            patch("neurodags.tui.NeuroDagsApp.run") as m,
            patch("neurodags.tui.NeuroDagsApp.__init__", return_value=None) as mi,
        ):
            main()
            mi.assert_called_once_with(config_path=str(yaml), datasets_path=str(datasets))
            m.assert_called_once()


# ---------------------------------------------------------------------------
# End to end: what the Config tab shows is what runs; derivative filter syntax
# ---------------------------------------------------------------------------


def _two_dataset_files(p, names=("shown", "actual")):
    import yaml as _yaml

    (p / "cfg").mkdir()
    for name in names:
        (p / name).mkdir()
        (p / name / "x.txt").write_text(name)

    def ds(name):
        return {
            name: {
                "name": name,
                "source_root": str(p / name),
                "file_pattern": str(p / name / "*.txt"),
                "derivatives_path": str(p / (name + "-out")),
            }
        }

    defs = {
        n: {
            "for_dataframe": True,
            "nodes": [
                {"id": 0, "derivative": "SourceFile"},
                {"id": 1, "node": "dummy", "args": {"param1": "id.0"}},
            ],
        }
        for n in ("A", "B", "C")
    }
    # Same relative name in two places: cwd/datasets.yml and cfg/datasets.yml.
    (p / "datasets.yml").write_text(_yaml.safe_dump(ds(names[0])))
    (p / "cfg" / "datasets.yml").write_text(_yaml.safe_dump(ds(names[1])))
    (p / "cfg" / "pipeline.yml").write_text(
        _yaml.safe_dump(
            {
                "naming_version": 1,
                "DerivativeDefinitions": defs,
                "DerivativeList": ["A", "B", "C"],
            }
        )
    )


def _artifacts(p):
    return sorted(str(f.relative_to(p)) for f in p.rglob("*.message.txt"))


class TestConfigTabExecutesWhatItShows:
    def test_relative_datasets_override_is_cwd_relative_for_display_and_run(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        _two_dataset_files(tmp_path)

        async def _():
            async with NeuroDagsApp().run_test() as pilot:
                app = pilot.app
                app.query_one("#config-path-input", Input).value = "cfg/pipeline.yml"
                app.query_one("#datasets-path-input", Input).value = "datasets.yml"
                app._handle_load_config()
                summary = str(app.query_one("#config-summary", Static).render())
                assert "Datasets: shown" in summary
                assert app._datasets_path == str(tmp_path / "datasets.yml")
                await app._run_pipeline()
                assert _artifacts(tmp_path) == [
                    "shown-out/x.txt@A.message.txt",
                    "shown-out/x.txt@B.message.txt",
                    "shown-out/x.txt@C.message.txt",
                ]

        _run(_())

    def test_missing_relative_datasets_override_is_an_error(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _two_dataset_files(tmp_path)
        (tmp_path / "datasets.yml").unlink()  # only cfg/datasets.yml exists now

        async def _():
            async with NeuroDagsApp().run_test() as pilot:
                app = pilot.app
                app.query_one("#config-path-input", Input).value = "cfg/pipeline.yml"
                app.query_one("#datasets-path-input", Input).value = "datasets.yml"
                app._handle_load_config()
                status = str(app.query_one("#config-status", Static).render())
                assert "Error" in status
                assert "not found" in status
                assert app._config is None
                assert app._datasets_path is None

        _run(_())

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("A B", {"A", "B"}),
            ("A,B", {"A", "B"}),
            (" A ,  B ", {"A", "B"}),
            ("A\tB", {"A", "B"}),
            ("B", {"B"}),
            ("", {"A", "B", "C"}),  # blank: every for_dataframe derivative
            ("  ", {"A", "B", "C"}),
        ],
    )
    def test_dataframe_filter_selects_exactly_the_named_derivatives(
        self, tmp_path, monkeypatch, value, expected
    ):
        # C is computed and eligible too: ignoring the filter would include it.
        monkeypatch.chdir(tmp_path)
        _two_dataset_files(tmp_path)

        async def _():
            async with NeuroDagsApp().run_test() as pilot:
                app = pilot.app
                app.query_one("#config-path-input", Input).value = "cfg/pipeline.yml"
                app.query_one("#datasets-path-input", Input).value = "datasets.yml"
                app._handle_load_config()
                await app._run_pipeline()
                app.query_one("#df-derivatives", Input).value = value
                await app._assemble_dataframe()
                assert app._df is not None
                columns = {c for c in app._df.columns if c.endswith(".message.txt")}
                assert columns == {f"{d}.message.txt" for d in expected}

        _run(_())
