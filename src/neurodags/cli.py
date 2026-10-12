"""Top-level CLI for NeuroDAGs."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from neurodags.datasets import get_datasets_and_mount_point_from_pipeline_configuration
from neurodags.iterators import (
    InputEnumeration,
    InputsError,
    InputsFileError,
    discover_inputs,
    enumerate_inputs,
    format_inputs,
    write_inputs_file,
)
from neurodags.loaders import ConfigVersionError, load_pipeline_configuration
from neurodags.loggers import configure_logging
from neurodags.mermaid import (
    derivative_to_html,
    derivative_to_mermaid,
    pipeline_to_html,
    pipeline_to_mermaid,
)
from neurodags.naming import NamingError
from neurodags.orchestrators import (
    _derivative_topo_order,
    build_derivative_dataframe,
    run_pipeline,
)
from neurodags.utils import (
    DatasetConfigError,
    check_derivative_names,
    expand_path,
    inspect_dataset_paths,
    resolve_dataset_paths,
)

# Help text shared by every subcommand that takes -d/--datasets (see #43).
DATASETS_HELP = (
    "Optional path to a datasets YAML that overrides the pipeline's 'datasets' key. "
    "Resolved relative to the current working directory (~ and $VARS are expanded). "
    "Relative paths inside that YAML are resolved against the YAML's own directory."
)

# Help text shared by every subcommand that takes --inputs (spec §8.6).
INPUTS_HELP = (
    "Frozen input list written by 'neurodags inputs' (path relative to the current working "
    "directory). It replaces discovery: indices come from the list, files added on disk "
    "later are ignored, and listed files that no longer exist are reported as "
    "missing-source. Cannot be combined with --max-files-per-dataset."
)


class DatasetsPathError(ValueError):
    """A ``-d/--datasets`` path given on the command line could not be resolved."""


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
    # Refuses a configuration without naming_version: 1 (ConfigVersionError).
    return load_pipeline_configuration(config_path)


def resolve_cli_datasets_path(datasets: str | None) -> str | None:
    """Resolve a CLI ``-d/--datasets`` path to an absolute path.

    CLI paths are relative to where the command is run (the current working
    directory); paths written inside the pipeline YAML (its ``datasets:`` key) stay
    relative to the YAML and are not handled here. ``~`` and ``$VARS`` are expanded as
    for the ``datasets:`` key. The YAML-relative fallback deprecated in 0.3.5 was
    removed in 0.4.0.

    Parameters
    ----------
    datasets : str or None
        The value passed to ``-d/--datasets``. ``None`` is returned unchanged.

    Returns
    -------
    str or None
        The absolute path of the datasets YAML, or ``None``.

    Raises
    ------
    DatasetsPathError
        If a referenced environment variable is unset/empty, or the file does not
        exist relative to the working directory.
    """
    if datasets is None:
        return None
    try:
        expanded = expand_path(datasets, field="--datasets")
    except ValueError as exc:
        raise DatasetsPathError(str(exc)) from exc

    cwd_path = Path(os.path.abspath(expanded))
    if cwd_path.is_file():
        return str(cwd_path)

    raise DatasetsPathError(
        f"datasets file '{datasets}' not found (tried {cwd_path}). "
        "CLI paths are relative to the current working directory."
    )


def resolve_cli_inputs_path(inputs: str | None) -> str | None:
    """Resolve a CLI ``--inputs`` path to an absolute path (cwd-relative, ``~``/``$VARS``).

    Raises
    ------
    InputsFileError
        If a referenced environment variable is unset/empty or the file does not exist.
    """
    if inputs is None:
        return None
    try:
        expanded = expand_path(inputs, field="--inputs")
    except ValueError as exc:
        raise InputsFileError(str(exc)) from exc
    path = Path(os.path.abspath(expanded))
    if not path.is_file():
        raise InputsFileError(
            f"input list '{inputs}' not found (tried {path}). "
            "CLI paths are relative to the current working directory."
        )
    return str(path)


def _cli_enumeration(args: argparse.Namespace) -> InputEnumeration:
    """Enumerate the inputs once for a CLI invocation (discovery or ``--inputs``)."""
    datasets, mount_point = get_datasets_and_mount_point_from_pipeline_configuration(
        args.config, datasets_input=args.datasets
    )
    resolve_dataset_paths(datasets, mount_point)
    return enumerate_inputs(datasets, mount_point, inputs=getattr(args, "inputs", None))


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
        help=DATASETS_HELP,
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
        help="Restrict processing to these file indices (unknown or negative indices are an "
        "error).",
    )
    parser.add_argument("--inputs", default=None, metavar="FILE", help=INPUTS_HELP)
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


def _slurm_per_file(config_path: str, derivatives: list[str]) -> str:
    deriv_names = ", ".join(derivatives)
    return f"""\
#!/bin/bash
# per-file: one array task per file — all derivatives run in dependency order per task.
# Derivatives: {deriv_names}
#
# Submit (freeze the input list once; every task reads the same list, see the HPC guide):
#   neurodags inputs {config_path} -o inputs.tsv
#   N=$(neurodags count-inputs {config_path} --inputs inputs.tsv)
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
from neurodags.loaders import load_pipeline_configuration
from neurodags.orchestrators import run_pipeline

config_path = os.path.abspath("{config_path}")
config = load_pipeline_configuration(config_path)
run_pipeline(
    config,
    base_dir=os.path.dirname(config_path),
    inputs=os.path.abspath("inputs.tsv"),  # frozen list (neurodags inputs ... -o inputs.tsv)
    only_index=int(os.environ["SLURM_ARRAY_TASK_ID"]),
    raise_on_error=True,
    skip_errors=True,
)
EOF
"""


def _slurm_flat(config_path: str, derivatives: list[str]) -> str:
    n = len(derivatives)
    deriv_arr = "(" + " ".join(f'"{d}"' for d in derivatives) + ")"
    return f"""\
#!/bin/bash
# flat: one array task per file x derivative (maximum parallelism).
# WARNING: use only when derivatives are independent (no inter-derivative dependencies).
#
# Submit (freeze the input list once; every task reads the same list, see the HPC guide):
#   neurodags inputs {config_path} -o inputs.tsv
#   N=$(neurodags count-inputs {config_path} --inputs inputs.tsv)
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
from neurodags.loaders import load_pipeline_configuration
from neurodags.orchestrators import run_pipeline

config_path = os.path.abspath("{config_path}")
config = load_pipeline_configuration(config_path)
run_pipeline(
    config,
    base_dir=os.path.dirname(config_path),
    inputs=os.path.abspath("inputs.tsv"),  # frozen list (neurodags inputs ... -o inputs.tsv)
    derivatives=["$DERIVATIVE"],
    only_index=$FILE_INDEX,
    raise_on_error=True,
)
EOF
"""


def _slurm_chained(config_path: str, derivatives: list[str]) -> tuple[str, str]:
    """Return (submit_script, worker_script) for chained per-derivative arrays."""
    submit_lines = [
        "#!/bin/bash",
        "# chained: per-derivative sequential arrays chained via --dependency=afterok.",
        "# All files run in parallel within each derivative; derivatives run in order.",
        "#",
        "# Usage: bash submit_pipeline.sh",
        "",
        "# Freeze the input list once; every task reads the same list (see the HPC guide).",
        f"neurodags inputs {config_path} -o inputs.tsv",
        f"N=$(neurodags count-inputs {config_path} --inputs inputs.tsv)",
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
# Worker script used by submit_pipeline.sh (chained pattern).
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
from neurodags.loaders import load_pipeline_configuration
from neurodags.orchestrators import run_pipeline

config_path = os.path.abspath("{config_path}")
config = load_pipeline_configuration(config_path)
run_pipeline(
    config,
    base_dir=os.path.dirname(config_path),
    inputs=os.path.abspath("inputs.tsv"),  # frozen list (neurodags inputs ... -o inputs.tsv)
    derivatives=["$DERIVATIVE"],
    only_index=int(os.environ["SLURM_ARRAY_TASK_ID"]),
    raise_on_error=True,
    skip_errors=True,
)
EOF
"""
    return submit, worker


def _status_classify(plan: list) -> str:
    """Return 'done', 'skipped', 'missing', 'errored', 'no-save', or 'unknown' from a dry-run plan list.

    'done' means at least one artifact file matching the derivative prefix exists on disk.
    It does NOT verify that all artifact files are present — a derivative that produces
    multiple artifacts (multiple keys in NodeResult.artifacts) will show as 'done' even
    if only one of those files was written (e.g. after a partial failure).
    'skipped' means a .skip marker exists — the derivative was intentionally not computed
    for this source file (e.g. a condition not present in a subject's recording).
    """
    if not isinstance(plan, list):
        return "unknown"
    final = next((s for s in plan if s.get("kind") == "derivative_output"), None)
    if final is None:
        return "unknown"
    if final.get("has_skip_marker"):
        return "skipped"
    if final.get("has_error_marker"):
        return "errored"
    if not final.get("will_save", True):
        return "no-save"
    if final.get("cached"):
        return "done"
    return "missing"


def _status_error_path(plan: list) -> str | None:
    if not isinstance(plan, list):
        return None
    final = next((s for s in plan if s.get("kind") == "derivative_output"), None)
    return final.get("error_path") if final else None


def _status_scalar(value: object) -> object:
    """Return ``None`` for missing/NaN values, else the value unchanged."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _status_rows(result: pd.DataFrame) -> list[dict]:
    """Project a dry-run dataframe onto one status record per (source file, derivative).

    Each record carries the global file ``index`` (the same index ``--only-index`` and the
    SLURM array templates consume), cast to a plain ``int`` so it is JSON serialisable. A
    missing ``index`` column is tolerated (``index`` is then ``None``) — this is a reporting
    path and must not fail.
    """
    rows = []
    for _, row in result.iterrows():
        plan = row.get("plan", [])
        idx = _status_scalar(row.get("index"))
        dataset = _status_scalar(row.get("dataset"))
        # Listed in a frozen input list (--inputs) but gone from disk (spec §8.6).
        missing_source = _status_scalar(row.get("source_status")) == "missing-source"
        rows.append(
            {
                "index": None if idx is None else int(idx),
                "dataset": None if dataset is None else str(dataset),
                "derivative": row.get("derivative", "unknown"),
                "file_path": row.get("file_path", ""),
                "status": "missing-source" if missing_source else _status_classify(plan),
                "error_path": _status_error_path(plan),
            }
        )
    return rows


def _status_index_key(row: dict) -> tuple[bool, int]:
    idx = row["index"]
    return (idx is None, idx if idx is not None else 0)


def _status_indices(rows: list[dict], statuses: tuple[str, ...]) -> list[int]:
    """Sorted, de-duplicated file indices whose status is one of *statuses*."""
    return sorted({r["index"] for r in rows if r["status"] in statuses and r["index"] is not None})


def _cmd_status(args: argparse.Namespace) -> int:
    import json

    config = _load_pipeline_config(args.config)
    derivatives = _resolve_derivatives(config, args.derivatives)

    # With --inputs, enumerate here (once, shared with the dry run) to also report the
    # files on disk that are not in the list; otherwise run_pipeline enumerates once.
    enumeration = _cli_enumeration(args) if getattr(args, "inputs", None) else None
    result = run_pipeline(
        pipeline_configuration=args.config,
        datasets_configuration=args.datasets,
        derivatives=derivatives,
        dry_run=True,
        n_jobs=args.n_jobs,
        enumeration=enumeration,
    )

    fmt = getattr(args, "format", "table")
    include_files = getattr(args, "include_files", False)
    rows = [] if result is None or result.empty else _status_rows(result)

    grand = {
        "total": 0,
        "done": 0,
        "skipped": 0,
        "missing": 0,
        "errored": 0,
        "missing-source": 0,
    }
    per_deriv: dict[str, dict[str, int]] = {}
    all_derivs = sorted({r["derivative"] for r in rows})
    total_files = len({r["file_path"] for r in rows})

    for deriv in all_derivs:
        sub = [r for r in rows if r["derivative"] == deriv]
        counts = {k: sum(1 for r in sub if r["status"] == k) for k in grand if k != "total"}
        per_deriv[deriv] = {"total": len(sub), **counts}
        for k, v in per_deriv[deriv].items():
            grand[k] += v

    complete = grand["missing"] == 0 and grand["errored"] == 0 and grand["missing-source"] == 0
    exit_code = 0 if complete or getattr(args, "exit_zero", False) else 1
    missing_source_indices = _status_indices(rows, ("missing-source",))
    n_not_in_list = enumeration.n_not_in_list if enumeration is not None else 0
    inputs_file = enumeration.inputs_file if enumeration is not None else None

    if fmt == "indices":
        if args.list_missing and not args.list_errors:
            wanted: tuple[str, ...] = ("missing",)
        elif args.list_errors and not args.list_missing:
            wanted = ("errored",)
        else:
            wanted = ("missing", "errored")
        print(" ".join(str(i) for i in _status_indices(rows, wanted)))
        return exit_code

    if fmt == "json":
        payload: dict = {
            "config": args.config,
            "n_files": total_files,
            "derivatives": per_deriv,
            "grand_total": grand,
            "complete": complete,
            "missing_indices": _status_indices(rows, ("missing",)),
            "errored_indices": _status_indices(rows, ("errored",)),
            "inputs": inputs_file,
            "missing_source_indices": missing_source_indices,
            "not_in_list": n_not_in_list,
        }
        if include_files:
            payload["files"] = [
                {
                    "index": r["index"],
                    "dataset": r["dataset"],
                    "file_path": r["file_path"],
                    "derivative": r["derivative"],
                    "status": r["status"],
                    "error_path": r["error_path"],
                }
                for r in sorted(rows, key=lambda r: (*_status_index_key(r), r["derivative"]))
            ]
        print(json.dumps(payload, indent=2))
        return exit_code

    def _print_input_list_report() -> None:
        if n_not_in_list:
            print(
                f"\n{n_not_in_list} file(s) on disk are not in the input list "
                f"{inputs_file} and were ignored."
            )
        if missing_source_indices:
            print(
                f"\n{len(missing_source_indices)} listed input(s) no longer exist on disk "
                f"(missing-source), not processed: indices "
                f"{' '.join(map(str, missing_source_indices))}"
            )

    if not rows:
        print("No files found.")
        _print_input_list_report()
        return 0

    # --- table output ---
    col_w = max(24, max(len(d) for d in all_derivs) + 2)
    header = (
        f"{'Derivative':<{col_w}} {'total':>6}  {'done':>6}  {'skipped':>8}"
        f"  {'missing':>8}  {'errored':>8}"
    )
    sep = "─" * len(header)

    print(f"config: {args.config}")
    if inputs_file:
        print(f"inputs: {inputs_file}")
    print(f"files:  {total_files}")
    print()
    print(header)
    print(sep)

    for deriv in all_derivs:
        c = per_deriv[deriv]
        print(
            f"{deriv:<{col_w}} {c['total']:>6}  {c['done']:>6}  {c['skipped']:>8}"
            f"  {c['missing']:>8}  {c['errored']:>8}"
        )

    print(sep)
    print(
        f"{'Total':<{col_w}} {grand['total']:>6}  {grand['done']:>6}  {grand['skipped']:>8}"
        f"  {grand['missing']:>8}  {grand['errored']:>8}"
    )
    print(
        "\nNote: 'total' counts (derivative x source-file) pairs, not output files. "
        "'done' means ≥1 artifact file exists — a derivative that writes multiple files "
        "per source file may show 'done' even after a partial write. "
        "'skipped' means a .skip marker exists — the derivative was intentionally not "
        "computed for that source file (e.g. condition absent from the recording)."
    )
    _print_input_list_report()

    if grand["errored"]:
        print(f"\n{grand['errored']} error(s) found.", end="")
        if not args.list_errors:
            print("  Run with --list-errors for details.", end="")
        print()

    if grand["skipped"]:
        print(
            f"\n{grand['skipped']} derivative(s) skipped (not applicable for those source files)."
        )

    if grand["missing"] and not complete:
        print(f"{grand['missing']} derivative(s) missing.", end="")
        if not args.list_missing:
            print("  Run with --list-missing for details.", end="")
        print()

    if not complete:
        print("Use --format indices to get the file indices for --only-index resubmission.")

    def _print_list(title: str, status: str, show_error_path: bool) -> None:
        selected = [r for r in rows if r["status"] == status]
        if not selected:
            return
        print(f"\n{title}")
        for deriv in all_derivs:
            sub = sorted((r for r in selected if r["derivative"] == deriv), key=_status_index_key)
            if not sub:
                continue
            print(f"  [{deriv}]")
            for r in sub:
                tag = f"[idx {r['index'] if r['index'] is not None else '?'}]"
                print(f"    {tag} {r['file_path']}")
                if show_error_path and r["error_path"]:
                    print(f"      error file: {r['error_path']}")

    if args.list_errors:
        _print_list("Errored files:", "errored", show_error_path=True)

    if args.list_missing:
        _print_list("Missing files:", "missing", show_error_path=False)

    return exit_code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="NeuroDAGs command line interface")
    parser.add_argument(
        "--log-level",
        default=None,
        metavar="LEVEL",
        help="Set log verbosity (DEBUG, INFO, WARNING, ERROR). Default: INFO or $LOG_LEVEL.",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        metavar="PATH",
        help=(
            "Write all log events to PATH in JSONL format (one JSON object per line). "
            "Load later with: pd.read_json(PATH, lines=True)"
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    tui_parser = subparsers.add_parser("tui", help="Launch the Textual TUI.")
    tui_parser.add_argument("config", nargs="?", help="Path to the pipeline YAML configuration.")
    tui_parser.add_argument(
        "-d",
        "--datasets",
        default=None,
        help=DATASETS_HELP,
    )

    validate_parser = subparsers.add_parser("validate", help="Load a config and print a summary.")
    validate_parser.add_argument("config", help="Path to the pipeline YAML configuration.")
    validate_parser.add_argument(
        "-d",
        "--datasets",
        default=None,
        help=DATASETS_HELP,
    )
    validate_parser.add_argument(
        "--derivative",
        action="append",
        dest="derivatives",
        default=None,
        help="Preview the effective run set for this derivative selection (repeat the flag "
        "to add more). Defaults to the config's DerivativeList.",
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
        help=DATASETS_HELP,
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
        help="Restrict processing to these file indices (unknown or negative indices are an "
        "error).",
    )
    dataframe_parser.add_argument("--inputs", default=None, metavar="FILE", help=INPUTS_HELP)
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
        "--n-jobs",
        type=int,
        default=None,
        help="Number of parallel workers for file-level collection. -1 uses all cores.",
    )
    dataframe_parser.add_argument(
        "--output",
        default="output_neurodags_dataframe.csv",
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
    dag_parser.add_argument(
        "--layout",
        choices=("dagre", "elk"),
        default="elk",
        help=(
            "Layout engine for HTML output: 'elk' (default, orthogonal routing — requires CDN "
            "access) or 'dagre' (step-curve edges, no CDN dependency)."
        ),
    )

    view_parser = subparsers.add_parser("view", help="Launch the Dash NC/FIF explorer.")
    view_parser.add_argument("path", help="Path to a .nc or .fif file.")

    inputs_parser = subparsers.add_parser(
        "inputs",
        help="Write the frozen input list (index, dataset, path below source_root).",
    )
    inputs_parser.add_argument("config", help="Path to the pipeline YAML configuration.")
    inputs_parser.add_argument(
        "-d",
        "--datasets",
        default=None,
        help=DATASETS_HELP,
    )
    inputs_parser.add_argument(
        "-o",
        "--output",
        default=None,
        metavar="FILE",
        help="Output path (relative to the current working directory). Omit to print to stdout.",
    )

    count_parser = subparsers.add_parser(
        "count-inputs",
        help="Print the number of input entries (the size of the --only-index range).",
    )
    count_parser.add_argument("config", help="Path to the pipeline YAML configuration.")
    count_parser.add_argument(
        "-d",
        "--datasets",
        default=None,
        help=DATASETS_HELP,
    )
    count_parser.add_argument(
        "--derivative",
        action="append",
        dest="derivatives",
        default=None,
        help="Accepted for compatibility; the count does not depend on the derivative.",
    )
    count_parser.add_argument("--inputs", default=None, metavar="FILE", help=INPUTS_HELP)

    status_parser = subparsers.add_parser(
        "status",
        help="Show how many files are done, missing, or errored per derivative.",
    )
    status_parser.add_argument("config", help="Path to the pipeline YAML configuration.")
    status_parser.add_argument(
        "-d",
        "--datasets",
        default=None,
        help=DATASETS_HELP,
    )
    status_parser.add_argument(
        "--derivative",
        action="append",
        dest="derivatives",
        default=None,
        help="Derivative to check. Repeat to add more. Defaults to all in DerivativeList.",
    )
    status_parser.add_argument("--inputs", default=None, metavar="FILE", help=INPUTS_HELP)
    status_parser.add_argument(
        "--list-errors",
        action="store_true",
        help=(
            "Print the file index and path of errored files below the summary table. "
            "With --format indices, restrict the output to errored indices."
        ),
    )
    status_parser.add_argument(
        "--list-missing",
        action="store_true",
        help=(
            "Print the file index and path of missing (not yet computed) files below the "
            "summary table. With --format indices, restrict the output to missing indices."
        ),
    )
    status_parser.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help="Number of parallel workers for the underlying dry-run. Same semantics as 'run'.",
    )
    status_parser.add_argument(
        "--format",
        choices=("table", "json", "indices"),
        default="table",
        help=(
            "Output format: 'table' (default), 'json', or 'indices' (only the sorted, "
            "space-separated file indices that are missing or errored — narrow with "
            "--list-missing / --list-errors — ready for --only-index or sbatch --array). "
            "The exit code is 1 whenever anything is missing or errored, in every format "
            "(unless --exit-zero)."
        ),
    )
    status_parser.add_argument(
        "--exit-zero",
        action="store_true",
        help=(
            "Exit 0 whenever the report was produced, even if files are missing or errored "
            "(for report jobs chained with --dependency=afterok). Real failures such as a "
            "missing or invalid config or an unset environment variable still exit non-zero."
        ),
    )
    status_parser.add_argument(
        "--include-files",
        action="store_true",
        help=(
            "With --format json, add a 'files' list with one record per "
            "(source file, derivative) pair: index, dataset, file_path, derivative, "
            "status, error_path."
        ),
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
        help=DATASETS_HELP,
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
        default="per-file",
        help=(
            "Submission pattern: "
            "per-file=all derivatives per file task (default), "
            "flat=file x derivative flat array (max parallelism, independent derivatives only), "
            "chained=per-derivative sequential arrays linked via --dependency=afterok."
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

    if getattr(args, "config", None):
        _load_pipeline_config(args.config)  # refuse an unmigrated config before the UI starts

    NeuroDagsApp(config_path=args.config, datasets_path=args.datasets).run()
    return 0


def _dependency_closure(defs: dict, selection: list[str]) -> list[str]:
    """Return *selection* plus every derivative it transitively references.

    A ``run`` recomputes each selected derivative's ``derivative:`` step
    references (its intermediates) even though they are not listed at the top
    level, so the closure is what a run actually computes. Only references that
    are themselves defined derivatives are followed; ``SourceFile`` is ignored.
    """
    result: set[str] = set()

    def visit(name: str) -> None:
        if name in result:
            return
        result.add(name)
        for step in (defs.get(name) or {}).get("nodes", []) or []:
            ref = step.get("derivative", "")
            if ref and ref != "SourceFile":
                base = ref.split(".", 1)[0]
                if base in defs and base != name:
                    visit(base)

    for name in selection:
        visit(name)
    return sorted(result)


def _print_dataset_report(
    datasets: dict, resolved_paths: dict, mount_point: object, problems: list[str]
) -> None:
    """Print the resolved path fields per active dataset (stdout) and every problem (stderr)."""
    for ds_name, cfg in datasets.items():
        if getattr(cfg, "skip", False):
            print(f"  {ds_name}: skipped")
            continue
        fields = resolved_paths.get(ds_name, {})
        print(f"  {ds_name}:")
        print(f"    mount: {mount_point}")
        print(f"    source_root: {fields.get('source_root')}")
        print(f"    file_pattern: {fields.get('file_pattern')}")
        if fields.get("exclude_pattern"):
            print(f"    exclude_pattern: {fields['exclude_pattern']}")
        print(f"    derivatives_path: {fields.get('derivatives_path')}")
        if fields.get("derivatives_subdir"):
            print(f"    derivatives_subdir: {fields['derivatives_subdir']}")
            print(f"    outputs under: {fields.get('derivatives_base')}")
    for problem in problems:
        # Errors (with source_root suggestions) go to stderr, one per line.
        print(f"error: {problem}", file=sys.stderr)


def _cmd_validate(args: argparse.Namespace) -> int:
    config = _load_pipeline_config(args.config)
    try:
        datasets, mount_point = get_datasets_and_mount_point_from_pipeline_configuration(
            args.config, datasets_input=args.datasets
        )
        resolved_paths, problems = inspect_dataset_paths(datasets, mount_point)
        file_names = check_derivative_names(config.get("DerivativeDefinitions") or {})
    except (ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    defs = config.get("DerivativeDefinitions") or {}
    renamed = {name: fname for name, fname in file_names.items() if name != fname}
    defined = list(defs.keys())
    derivative_list = list(config.get("DerivativeList") or [])

    selected = list(args.derivatives) if getattr(args, "derivatives", None) else derivative_list
    selection_source = "--derivative" if getattr(args, "derivatives", None) else "DerivativeList"

    computed = _dependency_closure(defs, selected)
    computed_set = set(computed)
    selected_set = set(selected)
    deps_only = [d for d in computed if d not in selected_set]
    for_dataframe = [d for d in computed if (defs.get(d) or {}).get("for_dataframe")]
    not_computed = [d for d in defined if d not in computed_set]

    print(f"config: {args.config}")
    print(f"datasets: {', '.join(datasets.keys()) or 'none'}")
    print(f"mount_point: {mount_point}")
    _print_dataset_report(datasets, resolved_paths, mount_point, problems)
    print(f"derivatives_defined ({len(defined)}): {', '.join(defined) or 'none'}")
    if renamed:
        print(f"on-disk names differing from YAML names ({len(renamed)}):")
        for name, fname in renamed.items():
            print(f"  {name} -> files named @{fname}")
    print(f"run_set [{selection_source}] ({len(selected)}): {', '.join(selected) or 'none'}")
    print(f"computed_with_dependencies ({len(computed)}): {', '.join(computed) or 'none'}")
    if deps_only:
        print(f"  pulled in as dependencies: {', '.join(deps_only)}")
    if for_dataframe:
        print(f"  for_dataframe outputs: {', '.join(for_dataframe)}")
    if not_computed:
        print(f"not computed by this run ({len(not_computed)}): {', '.join(not_computed)}")
    return 1 if problems else 0


def _cmd_run(args: argparse.Namespace, *, dry_run: bool) -> int:
    config = _load_pipeline_config(args.config)
    derivatives = _resolve_derivatives(config, args.derivatives)

    result = run_pipeline(
        pipeline_configuration=args.config,
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
        inputs=args.inputs,
    )

    if dry_run and result is not None:
        output_path = _save_dataframe(result, args.output, "dry_run_results.csv")
        print(f"saved dry run results to {output_path}")
    return 0


def _cmd_dataframe(args: argparse.Namespace) -> int:
    df = build_derivative_dataframe(
        pipeline_configuration=args.config,
        datasets_configuration=args.datasets,
        include_derivatives=args.include_derivatives,
        max_files_per_dataset=args.max_files_per_dataset,
        only_index=args.only_index,
        output_format=args.output_format,
        preserve_complex_values=args.preserve_complex_values,
        raise_on_error=args.raise_on_error,
        n_jobs=args.n_jobs,
        inputs=args.inputs,
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
                layout=args.layout,
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
            layout=args.layout,
        )
        print(html_path)
    else:
        print(pipeline_to_mermaid(config))
    return 0


def _cmd_count(args: argparse.Namespace) -> int:
    _load_pipeline_config(args.config)
    # Number of index entries, the unit --only-index selects (the same file listed under
    # two dataset entries has two indices). With --inputs: the number of listed entries,
    # missing-source ones included, so the array covers every index of the list.
    print(len(_cli_enumeration(args).entries))
    return 0


def _cmd_inputs(args: argparse.Namespace) -> int:
    _load_pipeline_config(args.config)
    datasets, mount_point = get_datasets_and_mount_point_from_pipeline_configuration(
        args.config, datasets_input=args.datasets
    )
    resolve_dataset_paths(datasets, mount_point)
    enumeration = discover_inputs(datasets, mount_point)
    if args.output:
        path = write_inputs_file(enumeration, args.output)
        print(f"wrote {len(enumeration.entries)} input(s) to {path}")
    else:
        sys.stdout.write(format_inputs(enumeration.entries))
    return 0


def _cmd_slurm_script(args: argparse.Namespace) -> int:
    config = _load_pipeline_config(args.config)
    derivatives = _resolve_derivatives(config, args.derivatives)
    ordered = _derivative_topo_order(config, derivatives)

    if args.pattern == "per-file":
        script = _slurm_per_file(args.config, ordered)
        if args.output:
            Path(args.output).write_text(script)
            print(f"wrote {args.output}")
        else:
            print(script)
    elif args.pattern == "flat":
        script = _slurm_flat(args.config, ordered)
        if args.output:
            Path(args.output).write_text(script)
            print(f"wrote {args.output}")
        else:
            print(script)
    else:
        submit, worker = _slurm_chained(args.config, ordered)
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

    if args.command == "status" and getattr(args, "include_files", False) and args.format != "json":
        parser.error("--include-files requires --format json")

    configure_logging(
        level=args.log_level,
        log_file=args.log_file,
    )

    if getattr(args, "datasets", None) is not None:
        try:
            args.datasets = resolve_cli_datasets_path(args.datasets)
        except DatasetsPathError as exc:
            print(f"neurodags: error: {exc}", file=sys.stderr)
            return 2

    try:
        if getattr(args, "inputs", None) is not None:
            args.inputs = resolve_cli_inputs_path(args.inputs)
        return _dispatch(parser, args)
    except (ConfigVersionError, DatasetConfigError, InputsError, NamingError) as exc:
        # Configuration errors are user errors: one clean message, no traceback. NamingError
        # includes dataframe column collisions (naming.ColumnCollisionError).
        print(f"neurodags: error: {exc}", file=sys.stderr)
        return 2


def _dispatch(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
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
    if args.command == "count-inputs":
        return _cmd_count(args)
    if args.command == "inputs":
        return _cmd_inputs(args)
    if args.command == "status":
        return _cmd_status(args)
    if args.command == "slurm-script":
        return _cmd_slurm_script(args)

    parser.error(f"Unknown command '{args.command}'")
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
