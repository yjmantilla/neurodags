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
    _derivative_topo_order,
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
        "--skip-errors",
        action="store_true",
        help="Skip files that have a prior .error marker from a previous failed run.",
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


def _slurm_pattern1(config_path: str, derivatives: list[str]) -> str:
    deriv_names = ", ".join(derivatives)
    return f"""\
#!/bin/bash
# Pattern 1: one array task per file — all derivatives run in dependency order per task.
# Derivatives: {deriv_names}
#
# Submit:
#   N=$(neurodags count {config_path})
#   sbatch --array=0-$((N - 1)) run_array.sh
#
#SBATCH --job-name=neurodags
#SBATCH --time=02:00:00    # TODO: adjust per file
#SBATCH --mem=16G          # TODO: adjust
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/neurodags_%A_%a.out
#SBATCH --error=logs/neurodags_%A_%a.err

source activate myenv  # TODO: activate your environment

python - <<'EOF'
import os
from neurodags.loaders import load_configuration
from neurodags.orchestrators import run_pipeline

config = load_configuration("{config_path}")
run_pipeline(
    config,
    only_index=int(os.environ["SLURM_ARRAY_TASK_ID"]),
    raise_on_error=True,
    skip_errors=True,
)
EOF
"""


def _slurm_pattern2(config_path: str, derivatives: list[str]) -> str:
    n = len(derivatives)
    deriv_arr = "(" + " ".join(f'"{d}"' for d in derivatives) + ")"
    return f"""\
#!/bin/bash
# Pattern 2: one array task per file x derivative (maximum parallelism).
# WARNING: use only when derivatives are independent (no inter-derivative dependencies).
#
# Submit:
#   N=$(neurodags count {config_path})
#   TOTAL=$(( N * {n} ))
#   sbatch --array=0-$((TOTAL - 1)) run_array_per_deriv.sh
#
#SBATCH --job-name=neurodags
#SBATCH --time=01:00:00    # TODO: adjust per derivative
#SBATCH --mem=8G           # TODO: adjust
#SBATCH --output=logs/neurodags_%A_%a.out
#SBATCH --error=logs/neurodags_%A_%a.err

DERIVATIVES={deriv_arr}
N_DERIVATIVES={n}

FILE_INDEX=$(( SLURM_ARRAY_TASK_ID / N_DERIVATIVES ))
DERIV_INDEX=$(( SLURM_ARRAY_TASK_ID % N_DERIVATIVES ))
DERIVATIVE=${{DERIVATIVES[$DERIV_INDEX]}}

source activate myenv  # TODO: activate your environment

python - <<EOF
import os
from neurodags.loaders import load_configuration
from neurodags.orchestrators import run_pipeline

config = load_configuration("{config_path}")
run_pipeline(
    config,
    derivatives=["$DERIVATIVE"],
    only_index=$FILE_INDEX,
    raise_on_error=True,
)
EOF
"""


def _slurm_pattern3(config_path: str, derivatives: list[str]) -> tuple[str, str]:
    """Return (submit_script, worker_script) for chained per-derivative arrays."""
    submit_lines = [
        "#!/bin/bash",
        "# Pattern 3: per-derivative sequential arrays chained via --dependency=afterok.",
        "# All files run in parallel within each derivative; derivatives run in order.",
        "#",
        "# Usage: bash submit_pipeline.sh",
        "",
        f"N=$(neurodags count {config_path})",
        'ARRAY="0-$((N - 1))"',
        "",
    ]

    prev_var: str | None = None
    for i, deriv in enumerate(derivatives):
        var = f"JOB{i + 1}"
        dep = f"--dependency=afterok:${prev_var} " if prev_var else ""
        is_last = i == len(derivatives) - 1
        if is_last:
            submit_lines.append(
                f"sbatch --array=$ARRAY {dep}" f"--export=DERIVATIVE={deriv} run_one_derivative.sh"
            )
        else:
            submit_lines.append(
                f"{var}=$(sbatch --parsable --array=$ARRAY {dep}"
                f"--export=DERIVATIVE={deriv} run_one_derivative.sh)"
            )
        prev_var = var

    submit = "\n".join(submit_lines) + "\n"

    worker = f"""\
#!/bin/bash
# Worker script used by submit_pipeline.sh (Pattern 3).
# DERIVATIVE is injected via --export from the submit script.
#
#SBATCH --job-name=neurodags
#SBATCH --time=01:00:00    # TODO: adjust per derivative
#SBATCH --mem=8G           # TODO: adjust
#SBATCH --output=logs/neurodags_%x_%A_%a.out
#SBATCH --error=logs/neurodags_%x_%A_%a.err

source activate myenv  # TODO: activate your environment

python - <<EOF
import os
from neurodags.loaders import load_configuration
from neurodags.orchestrators import run_pipeline

config = load_configuration("{config_path}")
run_pipeline(
    config,
    derivatives=["$DERIVATIVE"],
    only_index=int(os.environ["SLURM_ARRAY_TASK_ID"]),
    raise_on_error=True,
    skip_errors=True,
)
EOF
"""
    return submit, worker


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

    count_parser = subparsers.add_parser(
        "count", help="Print the number of unique files the pipeline will process."
    )
    count_parser.add_argument("config", help="Path to the pipeline YAML configuration.")
    count_parser.add_argument(
        "-d",
        "--datasets",
        default=None,
        help="Optional path to a datasets YAML override.",
    )
    count_parser.add_argument(
        "--derivative",
        action="append",
        dest="derivatives",
        default=None,
        help="Derivative to use for counting. Repeat to add more. Defaults to first in DerivativeList.",
    )

    slurm_parser = subparsers.add_parser(
        "slurm-script",
        help="Emit a SLURM array job template populated with this pipeline's derivatives.",
    )
    slurm_parser.add_argument("config", help="Path to the pipeline YAML configuration.")
    slurm_parser.add_argument(
        "-d",
        "--datasets",
        default=None,
        help="Optional path to a datasets YAML override.",
    )
    slurm_parser.add_argument(
        "--derivative",
        action="append",
        dest="derivatives",
        default=None,
        help="Derivative to include. Repeat to add more. Defaults to DerivativeList.",
    )
    slurm_parser.add_argument(
        "--pattern",
        choices=("per-file", "flat", "chained"),
        default="chained",
        help=(
            "Submission pattern: "
            "per-file=all derivatives per file task, "
            "flat=file x derivative flat array (max parallelism, independent derivatives only), "
            "chained=per-derivative sequential arrays linked via --dependency=afterok (default)."
        ),
    )
    slurm_parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output path for the generated script. "
            "For chained pattern, a second file (run_one_derivative.sh) is written alongside. "
            "Omit to print to stdout."
        ),
    )

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
        skip_errors=args.skip_errors,
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


def _cmd_count(args: argparse.Namespace) -> int:
    config = _load_pipeline_config(args.config)
    derivatives = args.derivatives or (list(config.get("DerivativeList") or [])[:1])
    if not derivatives:
        raise ValueError("No derivatives found. Pass --derivative or define DerivativeList.")
    result = run_pipeline(
        pipeline_configuration=config,
        datasets_configuration=args.datasets,
        derivatives=derivatives,
        dry_run=True,
    )
    count = 0 if result is None or result.empty else int(result["file_path"].nunique())
    print(count)
    return 0


def _cmd_slurm_script(args: argparse.Namespace) -> int:
    config = _load_pipeline_config(args.config)
    derivatives = _resolve_derivatives(config, args.derivatives)
    ordered = _derivative_topo_order(config, derivatives)

    if args.pattern == "per-file":
        script = _slurm_pattern1(args.config, ordered)
        if args.output:
            Path(args.output).write_text(script)
            print(f"wrote {args.output}")
        else:
            print(script)
    elif args.pattern == "flat":
        script = _slurm_pattern2(args.config, ordered)
        if args.output:
            Path(args.output).write_text(script)
            print(f"wrote {args.output}")
        else:
            print(script)
    else:
        submit, worker = _slurm_pattern3(args.config, ordered)
        if args.output:
            submit_path = Path(args.output)
            worker_path = submit_path.parent / "run_one_derivative.sh"
            submit_path.write_text(submit)
            worker_path.write_text(worker)
            print(f"wrote {submit_path}")
            print(f"wrote {worker_path}")
        else:
            print(f"# === FILE: submit_pipeline.sh ===\n{submit}")
            print(f"# === FILE: run_one_derivative.sh ===\n{worker}")
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
    if args.command == "count":
        return _cmd_count(args)
    if args.command == "slurm-script":
        return _cmd_slurm_script(args)

    parser.error(f"Unknown command '{args.command}'")
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
