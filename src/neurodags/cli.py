"""Top-level CLI for NeuroDAGs."""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from neurodags.datasets import get_datasets_and_mount_point_from_pipeline_configuration
from neurodags.loaders import load_configuration
from neurodags.mermaid import (
    derivative_to_html,
    derivative_to_mermaid,
    pipeline_to_html,
    pipeline_to_mermaid,
)
from neurodags.orchestrators import (
    build_derivative_dataframe,
    run_pipeline,
)


def _save_dataframe(df: pd.DataFrame, output_path: str | None, default_name: str) -> Path:
    path = Path(output_path) if output_path else Path(default_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        try:
            df.to_parquet(path, index=False)
        except (ImportError, ValueError):
            csv_fallback = path.with_suffix(".csv")
            df.to_csv(csv_fallback, index=False)
            return csv_fallback
    else:
        df.to_csv(path, index=False)
    return path


def _load_pipeline_config(config_path: str) -> dict:
    return load_configuration(config_path)


def _resolve_derivatives(config: dict, derivatives: list[str] | None) -> list[str]:
    if derivatives:
        return derivatives
    derivative_list = config.get("DerivativeList", [])
    if not derivative_list:
        raise ValueError("No derivatives specified. Pass --derivative or define DerivativeList.")
    return list(derivative_list)


def _add_common_execution_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("config", help="Path to the pipeline YAML configuration.")
    parser.add_argument(
        "-d",
        "--datasets",
        default=None,
        help="Optional path to a datasets YAML override.",
    )
    parser.add_argument(
        "--derivative",
        action="append",
        dest="derivatives",
        default=None,
        help="Derivative to execute. Repeat the flag to run more than one.",
    )
    parser.add_argument(
        "--max-files-per-dataset",
        type=int,
        default=None,
        help="Maximum number of files to process per dataset.",
    )
    parser.add_argument(
        "--only-index",
        type=int,
        nargs="*",
        default=None,
        help="Restrict processing to these file indices.",
    )
    parser.add_argument(
        "--raise-on-error",
        action="store_true",
        help="Re-raise processing errors instead of continuing.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help="Number of parallel workers to use.",
    )
    parser.add_argument(
        "--joblib-backend",
        default=None,
        help="Joblib backend (for example: loky, threading).",
    )
    parser.add_argument(
        "--joblib-prefer",
        default=None,
        help="Joblib preference hint (for example: processes, threads).",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="NeuroDAGs command line interface")
    subparsers = parser.add_subparsers(dest="command", required=True)

    tui_parser = subparsers.add_parser("tui", help="Launch the Textual TUI.")
    tui_parser.add_argument("config", nargs="?", help="Path to the pipeline YAML configuration.")
    tui_parser.add_argument(
        "-d",
        "--datasets",
        default=None,
        help="Optional path to a datasets YAML override.",
    )

    validate_parser = subparsers.add_parser("validate", help="Load a config and print a summary.")
    validate_parser.add_argument("config", help="Path to the pipeline YAML configuration.")
    validate_parser.add_argument(
        "-d",
        "--datasets",
        default=None,
        help="Optional path to a datasets YAML override.",
    )

    run_parser = subparsers.add_parser("run", help="Execute one or more derivatives.")
    _add_common_execution_args(run_parser)

    dry_run_parser = subparsers.add_parser("dry-run", help="Dry-run one or more derivatives.")
    _add_common_execution_args(dry_run_parser)
    dry_run_parser.add_argument(
        "--output",
        default=None,
        help="Optional CSV or Parquet output path for the combined dry-run dataframe.",
    )

    dataframe_parser = subparsers.add_parser(
        "dataframe", help="Build a dataframe from for_dataframe derivatives."
    )
    dataframe_parser.add_argument("config", help="Path to the pipeline YAML configuration.")
    dataframe_parser.add_argument(
        "-d",
        "--datasets",
        default=None,
        help="Optional path to a datasets YAML override.",
    )
    dataframe_parser.add_argument(
        "--include-derivative",
        action="append",
        dest="include_derivatives",
        default=None,
        help="Derivative to include in dataframe assembly. Repeat the flag to add more.",
    )
    dataframe_parser.add_argument(
        "--max-files-per-dataset",
        type=int,
        default=None,
        help="Maximum number of files to process per dataset.",
    )
    dataframe_parser.add_argument(
        "--only-index",
        type=int,
        nargs="*",
        default=None,
        help="Restrict processing to these file indices.",
    )
    dataframe_parser.add_argument(
        "--format",
        choices=("wide", "long"),
        default="wide",
        dest="output_format",
        help="Output dataframe layout.",
    )
    dataframe_parser.add_argument(
        "--preserve-complex-values",
        action="store_true",
        help="Keep nested derivative artifacts instead of flattening them.",
    )
    dataframe_parser.add_argument(
        "--raise-on-error",
        action="store_true",
        help="Re-raise dataframe collection errors.",
    )
    dataframe_parser.add_argument(
        "--output",
        default=None,
        help="Optional CSV or Parquet output path.",
    )

    dag_parser = subparsers.add_parser("dag", help="Print or export Mermaid DAG diagrams.")
    dag_parser.add_argument("config", help="Path to the pipeline YAML configuration.")
    dag_parser.add_argument(
        "--derivative",
        default=None,
        help="When set, render the node-level DAG for a single derivative.",
    )
    dag_parser.add_argument(
        "--html",
        default=None,
        help="Optional output HTML path. If omitted, the Mermaid text is printed.",
    )
    dag_parser.add_argument(
        "--title",
        default=None,
        help="Optional HTML title override for pipeline DAG output.",
    )
    dag_parser.add_argument(
        "--open",
        action="store_true",
        dest="auto_open",
        help="Open the generated HTML in the default browser.",
    )

    view_parser = subparsers.add_parser("view", help="Launch the Dash NC/FIF explorer.")
    view_parser.add_argument("path", help="Path to a .nc or .fif file.")

    return parser


def _cmd_tui(args: argparse.Namespace) -> int:
    from neurodags.tui import NeuroDagsApp

    NeuroDagsApp(config_path=args.config, datasets_path=args.datasets).run()
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    config = _load_pipeline_config(args.config)
    datasets, mount_point = get_datasets_and_mount_point_from_pipeline_configuration(
        args.config, datasets_input=args.datasets
    )
    derivatives = list((config.get("DerivativeDefinitions") or {}).keys())
    enabled = list(config.get("DerivativeList") or [])
    print(f"config: {args.config}")
    print(f"datasets: {', '.join(datasets.keys()) or 'none'}")
    print(f"mount_point: {mount_point}")
    print(f"derivatives_defined: {', '.join(derivatives) or 'none'}")
    print(f"derivatives_enabled: {', '.join(enabled) or 'none'}")
    return 0


def _cmd_run(args: argparse.Namespace, *, dry_run: bool) -> int:
    config = _load_pipeline_config(args.config)
    derivatives = _resolve_derivatives(config, args.derivatives)

    result = run_pipeline(
        pipeline_configuration=config,
        datasets_configuration=args.datasets,
        derivatives=derivatives,
        max_files_per_dataset=args.max_files_per_dataset,
        dry_run=dry_run,
        only_index=args.only_index,
        raise_on_error=args.raise_on_error,
        n_jobs=args.n_jobs,
        joblib_backend=args.joblib_backend,
        joblib_prefer=args.joblib_prefer,
    )

    if dry_run and result is not None:
        output_path = _save_dataframe(result, args.output, "dry_run_results.csv")
        print(f"saved dry run results to {output_path}")
    return 0


def _cmd_dataframe(args: argparse.Namespace) -> int:
    config = _load_pipeline_config(args.config)
    df = build_derivative_dataframe(
        pipeline_configuration=config,
        datasets_configuration=args.datasets,
        include_derivatives=args.include_derivatives,
        max_files_per_dataset=args.max_files_per_dataset,
        only_index=args.only_index,
        output_format=args.output_format,
        preserve_complex_values=args.preserve_complex_values,
        raise_on_error=args.raise_on_error,
    )
    if args.output:
        output_path = _save_dataframe(df, args.output, "derivative_dataframe.csv")
        print(f"saved dataframe to {output_path}")
    else:
        print(df.to_string(index=False))
    return 0


def _cmd_dag(args: argparse.Namespace) -> int:
    config = _load_pipeline_config(args.config)
    if args.derivative:
        derivative_def = (config.get("DerivativeDefinitions") or {}).get(args.derivative)
        if derivative_def is None:
            raise KeyError(f"Unknown derivative '{args.derivative}'.")
        if args.html:
            html_path = derivative_to_html(
                derivative_def,
                args.derivative,
                output_path=args.html,
                auto_open=args.auto_open,
            )
            print(html_path)
        else:
            print(derivative_to_mermaid(derivative_def, args.derivative))
        return 0

    if args.html:
        html_path = pipeline_to_html(
            config,
            output_path=args.html,
            title=args.title or "Pipeline DAG",
            auto_open=args.auto_open,
        )
        print(html_path)
    else:
        print(pipeline_to_mermaid(config))
    return 0


def _cmd_view(args: argparse.Namespace) -> int:
    subprocess.Popen(
        [sys.executable, "-m", "neurodags.visualization", args.path],
        start_new_session=True,
    )
    print("Dash explorer launched at http://127.0.0.1:8050")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command == "tui":
        return _cmd_tui(args)
    if args.command == "validate":
        return _cmd_validate(args)
    if args.command == "run":
        return _cmd_run(args, dry_run=False)
    if args.command == "dry-run":
        return _cmd_run(args, dry_run=True)
    if args.command == "dataframe":
        return _cmd_dataframe(args)
    if args.command == "dag":
        return _cmd_dag(args)
    if args.command == "view":
        return _cmd_view(args)

    parser.error(f"Unknown command '{args.command}'")
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
