"""Textual TUI for NeuroDAGs pipeline management.

Usage:
    neurodags-tui [config.yaml]
    python -m neurodags.tui [config.yaml]

Requires the ``[tui]`` extra:
    pip install neurodags[tui]
"""

from __future__ import annotations

import asyncio
import io
import subprocess
import sys
import webbrowser
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, ClassVar

from rich.text import Text

try:
    from textual.app import App, ComposeResult
    from textual.containers import Horizontal, Vertical
    from textual.markup import escape as escape_markup
    from textual.widgets import (
        Button,
        Checkbox,
        DataTable,
        Footer,
        Header,
        Input,
        Label,
        Log,
        Select,
        Static,
        TabbedContent,
        TabPane,
        TextArea,
    )
except ImportError as exc:
    raise ImportError("TUI requires the [tui] extra: pip install neurodags[tui]") from exc


_BLANK = (
    Select.BLANK
)  # kept for format fallback comparison; use select.is_blank for no-selection checks


class _InspectableStatic(Static):
    """Static widget that renders its original content for easier inspection in tests."""

    def __init__(
        self,
        content: Any = "",
        *,
        expand: bool = False,
        shrink: bool = False,
        markup: bool = True,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
        disabled: bool = False,
    ) -> None:
        """Initialize the inspectable static widget."""
        super().__init__(
            content=content,
            expand=expand,
            shrink=shrink,
            markup=markup,
            name=name,
            id=id,
            classes=classes,
            disabled=disabled,
        )

    def render(self) -> Any:
        return self.content


class _ConfigTab(Vertical):
    """Configuration tab."""

    DEFAULT_CSS = """
    _ConfigTab { height: auto; padding: 1; }
    _ConfigTab .row { height: auto; margin-bottom: 1; }
    _ConfigTab Input { width: 3fr; }
    _ConfigTab Static { color: $text-muted; }
    _ConfigTab #config-status { color: $success; }
    """
    """Internal CSS."""

    def compose(self) -> ComposeResult:
        """Compose child widgets."""
        yield Label("Pipeline YAML path")
        with Horizontal(classes="row"):
            yield Input(placeholder="/path/to/pipeline.yaml", id="config-path-input")
            yield Button("Load", id="btn-load-config", variant="primary")
        yield Label("Datasets YAML path (optional override)")
        with Horizontal(classes="row"):
            yield Input(placeholder="/path/to/datasets.yaml", id="datasets-path-input")
        yield Static("", id="config-status")
        yield _InspectableStatic("", id="config-summary")


class _DagTab(Vertical):
    """DAG visualization tab."""

    DEFAULT_CSS = """
    _DagTab { height: 1fr; padding: 1; }
    _DagTab .row { height: auto; margin-bottom: 1; }
    _DagTab TextArea { height: 1fr; }
    """
    """Internal CSS."""

    def compose(self) -> ComposeResult:
        """Compose child widgets."""
        with Horizontal(classes="row"):
            yield Button("Refresh", id="btn-dag-refresh", variant="primary")
            yield Button("Open HTML in browser", id="btn-dag-html")
        yield TextArea("Load config first.", id="dag-mermaid", read_only=True)


class _DryRunTab(Vertical):
    """Dry run tab."""

    DEFAULT_CSS = """
    _DryRunTab { height: 1fr; padding: 1; }
    _DryRunTab .row { height: auto; margin-bottom: 1; }
    _DryRunTab Select { width: 2fr; }
    _DryRunTab Input { width: 1fr; }
    _DryRunTab DataTable { height: 1fr; }
    """
    """Internal CSS."""

    def compose(self) -> ComposeResult:
        """Compose child widgets."""
        with Horizontal(classes="row"):
            yield Select([], id="dryrun-derivative", prompt="Select derivative")
            yield Input(placeholder="max files/dataset (blank=all)", id="dryrun-max-files")
            yield Checkbox("Skip errored files", id="dryrun-skip-errors")
            yield Button("Run Dry Run", id="btn-dryrun", variant="primary")
        yield DataTable(id="dryrun-table")


class _RunTab(Vertical):
    """Pipeline execution tab."""

    DEFAULT_CSS = """
    _RunTab { height: 1fr; padding: 1; }
    _RunTab .row { height: auto; margin-bottom: 1; }
    _RunTab Select { width: 2fr; }
    _RunTab Input { width: 1fr; }
    _RunTab Log { height: 1fr; }
    """
    """Internal CSS."""

    def compose(self) -> ComposeResult:
        """Compose child widgets."""
        with Horizontal(classes="row"):
            yield Select([], id="run-derivative", prompt="Select derivative")
            yield Input(placeholder="max files/dataset (blank=all)", id="run-max-files")
            yield Input(placeholder="n_jobs (blank=1)", id="run-njobs")
            yield Checkbox("Skip errored files", id="run-skip-errors")
            yield Button("Run", id="btn-run", variant="success")
        yield Log(id="run-log", highlight=True)


class _DataFrameTab(Vertical):
    """DataFrame assembly tab."""

    DEFAULT_CSS = """
    _DataFrameTab { height: 1fr; padding: 1; }
    _DataFrameTab .row { height: auto; margin-bottom: 1; }
    _DataFrameTab Input { width: 2fr; }
    _DataFrameTab Select { width: 1fr; }
    _DataFrameTab DataTable { height: 1fr; }
    """
    """Internal CSS."""

    def compose(self) -> ComposeResult:
        """Compose child widgets."""
        with Horizontal(classes="row"):
            yield Input(
                placeholder="derivatives: comma-separated (blank=all)",
                id="df-derivatives",
            )
            yield Select(
                [("wide", "wide"), ("long", "long")],
                id="df-format",
                value="wide",
            )
            yield Input(placeholder="max files/dataset", id="df-max-files")
            yield Button("Assemble", id="btn-df-assemble", variant="primary")
        yield DataTable(id="df-table")


class _NcTab(Vertical):
    """NC viewer tab."""

    DEFAULT_CSS = """
    _NcTab { height: auto; padding: 1; }
    _NcTab .row { height: auto; margin-bottom: 1; }
    _NcTab Input { width: 3fr; }
    """
    """Internal CSS."""

    def compose(self) -> ComposeResult:
        """Compose child widgets."""
        yield Label(".fif or .nc file path")
        with Horizontal(classes="row"):
            yield Input(placeholder="/path/to/file.nc", id="nc-path")
            yield Button("Launch Dash Explorer", id="btn-nc-launch", variant="primary")
        yield Static("", id="nc-status")


class NeuroDagsApp(App):
    """NeuroDAGs TUI - pipeline dry-run, execution, DataFrame assembly, and NC viewer."""

    CSS = """
    TabbedContent { height: 1fr; }
    TabPane { height: 1fr; }
    """
    """Internal CSS."""

    BINDINGS: ClassVar[list[tuple[str, str, str]]] = [("q", "quit", "Quit")]

    def __init__(self, config_path: str | None = None, datasets_path: str | None = None) -> None:
        """Initialize the TUI with optional config and datasets paths."""
        super().__init__()
        self._config_path: str | None = config_path
        self._datasets_path: str | None = datasets_path
        self._config: dict[str, Any] | None = None
        self._derivatives: list[str] = []

    def compose(self) -> ComposeResult:
        """Compose the main application layout."""
        yield Header()
        with TabbedContent():
            with TabPane("Config", id="tab-config"):
                yield _ConfigTab()
            with TabPane("DAG", id="tab-dag"):
                yield _DagTab()
            with TabPane("Dry Run", id="tab-dryrun"):
                yield _DryRunTab()
            with TabPane("Run Pipeline", id="tab-run"):
                yield _RunTab()
            with TabPane("DataFrame", id="tab-dataframe"):
                yield _DataFrameTab()
            with TabPane("NC Viewer", id="tab-nc"):
                yield _NcTab()
        yield Footer()

    def on_mount(self) -> None:
        if self._config_path:
            self.query_one("#config-path-input", Input).value = self._config_path
        if self._datasets_path:
            self.query_one("#datasets-path-input", Input).value = self._datasets_path

        if self._config_path:
            if self._datasets_path:
                self._load_config(self._config_path, self._datasets_path)
            else:
                self._load_config(self._config_path)

    # ------------------------------------------------------------------
    # Event routing
    # ------------------------------------------------------------------

    def on_button_pressed(self, event: Button.Pressed) -> None:
        handlers: dict[str, Any] = {
            "btn-load-config": self._handle_load_config,
            "btn-dag-refresh": self._refresh_dag,
            "btn-dag-html": self._open_dag_html,
            "btn-dryrun": lambda: self.run_worker(self._run_dry_run(), exclusive=True),
            "btn-run": lambda: self.run_worker(self._run_pipeline(), exclusive=True),
            "btn-df-assemble": lambda: self.run_worker(self._assemble_dataframe(), exclusive=True),
            "btn-nc-launch": self._launch_nc_viewer,
        }
        handler = handlers.get(event.button.id or "")
        if handler:
            handler()

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------

    def _handle_load_config(self) -> None:
        path = self.query_one("#config-path-input", Input).value.strip()
        ds_path = self.query_one("#datasets-path-input", Input).value.strip() or None
        if path:
            if ds_path:
                self._load_config(path, ds_path)
            else:
                self._load_config(path)

    def _load_config(self, path: str, datasets_path: str | None = None) -> None:
        from neurodags.loaders import load_configuration

        status = self.query_one("#config-status", Static)
        summary = self.query_one("#config-summary", Static)
        try:
            self._config = load_configuration(path)
            self._config_path = path
            self._datasets_path = datasets_path
            self._derivatives = list(self._config.get("DerivativeDefinitions", {}).keys())
            if datasets_path:
                datasets_config = load_configuration(datasets_path)
            else:
                datasets_val = self._config.get("Datasets") or self._config.get("datasets")
                if isinstance(datasets_val, str):
                    ds_path = Path(datasets_val)
                    if not ds_path.is_absolute():
                        ds_path = Path(path).resolve().parent / ds_path
                    datasets_config = load_configuration(str(ds_path))
                else:
                    datasets_config = self._config
            datasets_dict = (
                datasets_config.get("Datasets")
                or datasets_config.get("datasets")
                or datasets_config
            )
            datasets = list((datasets_dict or {}).keys())

            status_lines = [f"[green]Loaded:[/green] {escape_markup(path)}"]
            if datasets_path:
                status_lines.append(f"[green]Datasets:[/green] {escape_markup(datasets_path)}")
            status.update("\n".join(status_lines))

            summary.update(
                Text(
                    f"Datasets: {', '.join(datasets) or 'none'}\n"
                    f"Derivatives: {', '.join(self._derivatives) or 'none'}"
                )
            )
            self._sync_derivative_selects()
        except Exception as exc:
            status.update(f"[red]Error:[/red] {escape_markup(str(exc))}")

    def _sync_derivative_selects(self) -> None:
        options = [(d, d) for d in self._derivatives]
        all_options = [("All (DerivativeList)", "__all__"), *options]
        for widget_id in ("#dryrun-derivative", "#run-derivative"):
            try:
                sel = self.query_one(widget_id, Select)
                sel.set_options(all_options)
                sel.value = "__all__"
            except Exception:
                pass

    # ------------------------------------------------------------------
    # DAG
    # ------------------------------------------------------------------

    def _refresh_dag(self) -> None:
        from neurodags.mermaid import pipeline_to_mermaid

        ta = self.query_one("#dag-mermaid", TextArea)
        if self._config is None:
            ta.load_text("Load config first.")
            return
        try:
            ta.load_text(pipeline_to_mermaid(self._config))
        except Exception as exc:
            ta.load_text(f"Error: {exc}")

    def _open_dag_html(self) -> None:
        from neurodags.mermaid import pipeline_to_html

        if self._config is None:
            self.notify("Load config first.", severity="error")
            return
        try:
            out = Path(self._config_path or ".").parent / "pipeline_dag_tui.html"
            html_path = pipeline_to_html(self._config, output_path=out, auto_open=False)
            webbrowser.open(html_path.resolve().as_uri())
        except Exception as exc:
            self.notify(f"Error: {exc}", severity="error")

    # ------------------------------------------------------------------
    # Dry run
    # ------------------------------------------------------------------

    async def _run_dry_run(self) -> None:
        from neurodags.orchestrators import run_pipeline

        if self._config is None:
            self.notify("Load config first.", severity="error")
            return
        dryrun_sel = self.query_one("#dryrun-derivative", Select)
        val = dryrun_sel.value
        derivatives: list[str] | None = (
            None if (dryrun_sel.is_blank() or val == "__all__") else [str(val)]
        )
        max_files = _parse_int(self.query_one("#dryrun-max-files", Input).value)
        skip_errors = self.query_one("#dryrun-skip-errors", Checkbox).value

        table = self.query_one("#dryrun-table", DataTable)
        table.clear(columns=True)
        self.notify("Running dry run…")

        try:
            result = await asyncio.to_thread(
                run_pipeline,
                self._config_path,
                datasets_configuration=self._datasets_path,
                derivatives=derivatives,
                max_files_per_dataset=max_files,
                dry_run=True,
                skip_errors=skip_errors,
            )
        except Exception as exc:
            self.notify(f"Dry run error: {exc}", severity="error")
            return

        if result is None or not hasattr(result, "columns") or len(result) == 0:
            self.notify("Dry run returned no data.", severity="warning")
            return

        table.add_columns(*[str(c) for c in result.columns])
        for _, row in result.iterrows():
            table.add_row(*[str(v) for v in row])
        self.notify(f"Dry run complete: {len(result)} rows.")

    # ------------------------------------------------------------------
    # Run pipeline
    # ------------------------------------------------------------------

    async def _run_pipeline(self) -> None:
        if self._config is None:
            self.notify("Load config first.", severity="error")
            return
        run_sel = self.query_one("#run-derivative", Select)
        val = run_sel.value
        derivatives: list[str] | None = (
            None if (run_sel.is_blank() or val == "__all__") else [str(val)]
        )
        max_files = _parse_int(self.query_one("#run-max-files", Input).value)
        n_jobs = _parse_int(self.query_one("#run-njobs", Input).value)
        skip_errors = self.query_one("#run-skip-errors", Checkbox).value

        log_widget = self.query_one("#run-log", Log)
        log_widget.clear()
        label = "all (DerivativeList)" if derivatives is None else derivatives[0]
        log_widget.write_line(f"Running: {label}")

        writer = _LiveLogWriter(self, log_widget)
        try:
            await asyncio.to_thread(
                _run_pipeline_sync,
                self._config_path,
                derivatives,
                self._datasets_path,
                max_files,
                n_jobs,
                skip_errors,
                writer,
            )
        except Exception as exc:
            log_widget.write_line(f"Error: {exc}")
            self.notify(f"Pipeline error: {exc}", severity="error")
            return

        writer.flush()
        log_widget.write_line("Pipeline complete.")
        self.notify("Pipeline complete!")

    # ------------------------------------------------------------------
    # DataFrame assembly
    # ------------------------------------------------------------------

    async def _assemble_dataframe(self) -> None:
        from neurodags.orchestrators import build_derivative_dataframe

        if self._config is None:
            self.notify("Load config first.", severity="error")
            return
        derivs_raw = self.query_one("#df-derivatives", Input).value.strip()
        include = [d.strip() for d in derivs_raw.split(",") if d.strip()] or None
        fmt_sel = self.query_one("#df-format", Select)
        fmt: str = str(fmt_sel.value) if not fmt_sel.is_blank() else "wide"
        max_files = _parse_int(self.query_one("#df-max-files", Input).value)

        table = self.query_one("#df-table", DataTable)
        table.clear(columns=True)
        self.notify("Assembling DataFrame…")

        try:
            df = await asyncio.to_thread(
                build_derivative_dataframe,
                self._config_path,
                datasets_configuration=self._datasets_path,
                include_derivatives=include,
                max_files_per_dataset=max_files,
                output_format=fmt,
            )
        except Exception as exc:
            self.notify(f"Error: {exc}", severity="error")
            return

        if df is None or len(df) == 0:
            self.notify("Empty DataFrame.", severity="warning")
            return

        table.add_columns(*[str(c) for c in df.columns])
        for _, row in df.iterrows():
            table.add_row(*[str(v) for v in row])
        self.notify(f"DataFrame: {df.shape[0]} rows x {df.shape[1]} cols")

    # ------------------------------------------------------------------
    # NC viewer
    # ------------------------------------------------------------------

    def _launch_nc_viewer(self) -> None:
        nc_path = self.query_one("#nc-path", Input).value.strip()
        if not nc_path:
            self.notify("Enter .nc or .fif file path.", severity="warning")
            return
        if not Path(nc_path).exists():
            self.notify(f"File not found: {nc_path}", severity="error")
            return
        try:
            subprocess.Popen(
                [sys.executable, "-m", "neurodags.visualization", nc_path],
                start_new_session=True,
            )
            self.query_one("#nc-status", Static).update(
                f"[green]Launched Dash explorer for {nc_path}[/green]\n"
                "Open browser at http://127.0.0.1:8050"
            )
            self.notify("Dash explorer launched.")
        except Exception as exc:
            self.notify(f"Launch error: {exc}", severity="error")


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


class _LiveLogWriter(io.TextIOBase):
    """Forwards writes to a Textual Log widget from a background thread."""

    def __init__(self, app: App, log_widget: Log) -> None:
        self._app = app
        self._log = log_widget
        self._buf = ""

    def write(self, s: str) -> int:
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self._app.call_from_thread(self._log.write_line, line)
        return len(s)

    def flush(self) -> None:
        if self._buf:
            self._app.call_from_thread(self._log.write_line, self._buf)
            self._buf = ""


def _parse_int(s: str) -> int | None:
    s = s.strip()
    try:
        return int(s)
    except ValueError:
        return None


def _run_pipeline_sync(
    config: str | None,
    derivatives: list[str] | None,
    datasets_config: str | dict | None,
    max_files: int | None,
    n_jobs: int | None,
    skip_errors: bool,
    buf: io.TextIOBase,
) -> None:
    from neurodags.orchestrators import run_pipeline

    with redirect_stdout(buf), redirect_stderr(buf):
        run_pipeline(
            config,
            datasets_configuration=datasets_config,
            derivatives=derivatives,
            max_files_per_dataset=max_files,
            n_jobs=n_jobs,
            skip_errors=skip_errors,
        )


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="NeuroDAGs TUI")
    parser.add_argument("config", nargs="?", help="Path to pipeline YAML config")
    parser.add_argument("-d", "--datasets", help="Optional path to datasets YAML config override")
    args = parser.parse_args()
    if args.datasets:
        NeuroDagsApp(config_path=args.config, datasets_path=args.datasets).run()
    else:
        NeuroDagsApp(config_path=args.config).run()


if __name__ == "__main__":
    main()
