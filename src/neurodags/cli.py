"""Top-level CLI for NeuroDAGs."""

from __future__ import annotations

import argparse
import os
import shlex
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
    output_tree_warnings,
    write_inputs_file,
)
from neurodags.loaders import (
    ConfigVersionError,
    check_pipeline_names,
    load_pipeline_configuration,
)
from neurodags.loggers import configure_logging
from neurodags.mermaid import (
    derivative_to_html,
    derivative_to_mermaid,
    pipeline_to_html,
    pipeline_to_mermaid,
)
from neurodags.migrate import MigrationError
from neurodags.naming import NamingError, validate_derivative_name
from neurodags.orchestrators import (
    _derivative_topo_order,
    build_derivative_dataframe,
    run_pipeline,
)
from neurodags.utils import (
    DatasetConfigError,
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
        for name in derivatives:  # --derivative values follow the name grammar (spec 2.2)
            validate_derivative_name(name)
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


def _slurm_paths(config_path: str, datasets: str | None) -> tuple[str, str, str, str]:
    """Shell pieces shared by the SLURM templates.

    Paths never reach the generated Python source: the shell sets ``PIPELINE`` (and
    ``DATASETS``) from :func:`shlex.quote`-d literals and passes them to ``python -`` as
    arguments, and every Python body is a quoted heredoc (``<<'EOF'``) that the shell
    leaves untouched. Returns ``(quoted pipeline, " -d <quoted datasets>" or "", the
    variable assignments, the ``"$PIPELINE" ["$DATASETS"]`` arguments)``.
    """
    q_config = shlex.quote(config_path)
    d_flag = f" -d {shlex.quote(datasets)}" if datasets else ""
    assign = f"PIPELINE={q_config}\n"
    args = '"$PIPELINE"'
    if datasets:
        assign += f"DATASETS={shlex.quote(datasets)}  # -d given to slurm-script\n"
        args += ' "$DATASETS"'
    return q_config, d_flag, assign, args


def _slurm_run_call(datasets: str | None, first_arg: int, extra: str) -> str:
    """The Python body of a worker: ``sys.argv[first_arg]`` is the pipeline path."""
    d_kwarg = (
        f"\n    datasets_configuration=sys.argv[{first_arg + 1}],  # -d given to slurm-script"
        if datasets
        else ""
    )
    return f"""\
import os
import sys
from neurodags.loaders import load_pipeline_configuration
from neurodags.orchestrators import run_pipeline

config_path = os.path.abspath(sys.argv[{first_arg}])
config = load_pipeline_configuration(config_path)
run_pipeline(
    config,
    base_dir=os.path.dirname(config_path),{d_kwarg}
    inputs=os.path.abspath("inputs.tsv"),  # frozen list (neurodags inputs ... -o inputs.tsv)
{extra}    raise_on_error=True,
"""


def _slurm_per_file(config_path: str, derivatives: list[str], datasets: str | None = None) -> str:
    deriv_names = ", ".join(derivatives)
    q_config, d_flag, assign, args = _slurm_paths(config_path, datasets)
    body = _slurm_run_call(
        datasets,
        1,
        f"    derivatives={list(derivatives)!r},\n"
        '    only_index=int(os.environ["SLURM_ARRAY_TASK_ID"]),\n',
    )
    return f"""\
#!/bin/bash
# per-file: one array task per file — all derivatives run in dependency order per task.
# Derivatives: {deriv_names}
#
# Submit (freeze the input list once; every task reads the same list, see the HPC guide):
#   neurodags inputs {q_config}{d_flag} -o inputs.tsv
#   N=$(neurodags count-inputs {q_config}{d_flag} --inputs inputs.tsv)
#   sbatch --array=0-$((N - 1)) run_array.sh
#
#SBATCH --job-name=neurodags
#SBATCH --time=02:00:00    # TODO: adjust per file
#SBATCH --mem=16G          # TODO: adjust
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/neurodags_%A_%a.out
#SBATCH --error=logs/neurodags_%A_%a.err

source activate myenv  # TODO: activate your environment

{assign}
python - {args} <<'EOF'
{body}    skip_errors=True,
)
EOF
"""


def _slurm_flat(config_path: str, derivatives: list[str], datasets: str | None = None) -> str:
    n = len(derivatives)
    q_config, d_flag, assign, args = _slurm_paths(config_path, datasets)
    deriv_arr = "(" + " ".join(f'"{d}"' for d in derivatives) + ")"
    # argv: 1 = derivative, 2 = file index, 3 = pipeline, 4 = datasets (with -d).
    body = _slurm_run_call(
        datasets,
        3,
        "    derivatives=[sys.argv[1]],\n    only_index=int(sys.argv[2]),\n",
    )
    return f"""\
#!/bin/bash
# flat: one array task per file x derivative (maximum parallelism).
# WARNING: use only when derivatives are independent (no inter-derivative dependencies).
#
# Submit (freeze the input list once; every task reads the same list, see the HPC guide):
#   neurodags inputs {q_config}{d_flag} -o inputs.tsv
#   N=$(neurodags count-inputs {q_config}{d_flag} --inputs inputs.tsv)
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

{assign}
python - "$DERIVATIVE" "$FILE_INDEX" {args} <<'EOF'
{body})
EOF
"""


def _slurm_chained(
    config_path: str, derivatives: list[str], datasets: str | None = None
) -> tuple[str, str]:
    """Return (submit_script, worker_script) for chained per-derivative arrays."""
    q_config, d_flag, assign, args = _slurm_paths(config_path, datasets)
    submit_lines = [
        "#!/bin/bash",
        "# chained: per-derivative sequential arrays chained via --dependency=afterok.",
        "# All files run in parallel within each derivative; derivatives run in order.",
        "#",
        "# Usage: bash submit_pipeline.sh",
        "",
        "# Freeze the input list once; every task reads the same list (see the HPC guide).",
        f"neurodags inputs {q_config}{d_flag} -o inputs.tsv",
        f"N=$(neurodags count-inputs {q_config}{d_flag} --inputs inputs.tsv)",
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

    body = _slurm_run_call(
        datasets,
        1,
        '    derivatives=[os.environ["DERIVATIVE"]],\n'
        '    only_index=int(os.environ["SLURM_ARRAY_TASK_ID"]),\n',
    )
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

{assign}
python - {args} <<'EOF'
{body}    skip_errors=True,
)
EOF
"""
    return submit, worker


_STATE_TO_STATUS = {
    "done": "done",
    "skipped": "skipped",
    "error": "errored",
    "incomplete": "incomplete",
    "missing": "missing",
}


def _status_final(plan: list) -> dict | None:
    if not isinstance(plan, list):
        return None
    return next((s for s in plan if s.get("kind") == "derivative_output"), None)


def _status_classify(plan: list) -> str:
    """Return the status of one unit from its dry-run plan.

    One of 'done', 'skipped', 'errored', 'incomplete', 'missing', 'no-save' or 'unknown':

    - 'done': the ``.done`` manifest is valid (it parses, matches the derivative and
      naming version, and every file it lists exists);
    - 'skipped': a ``.skip`` marker exists (intentionally not computed for this source);
    - 'errored': a ``.error`` marker exists;
    - 'incomplete': output files exist without a valid ``.done`` (a partial or
      interrupted write, or 0.3.x outputs not yet migrated); recomputed by the next run;
    - 'missing': nothing exists for the unit.
    """
    final = _status_final(plan)
    if final is None:
        return "unknown"
    if not final.get("will_save", True):
        return "no-save"
    state = final.get("state")
    if state in _STATE_TO_STATUS:
        return _STATE_TO_STATUS[state]
    # Plans without a state (older payloads): marker-based fallback.
    if final.get("has_skip_marker"):
        return "skipped"
    if final.get("cached"):
        return "done"
    if final.get("has_error_marker"):
        return "errored"
    return "missing"


def _status_origin(plan: list) -> str | None:
    """``origin`` of a done unit's manifest (``"migrate"`` for ``done (migrated)``)."""
    final = _status_final(plan)
    return final.get("origin") if final else None


def _status_unlisted(plan: list) -> list[str]:
    final = _status_final(plan)
    return list(final.get("unlisted") or []) if final else []


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
        # The dry run could not plan this unit (its error is kept on the row).
        planning_error = _status_scalar(row.get("planning_error"))
        if missing_source:
            status = "missing-source"
        elif planning_error is not None:
            status = "planning-error"
        else:
            status = _status_classify(plan)
        migrated = status == "done" and _status_origin(plan) == "migrate"
        rows.append(
            {
                "index": None if idx is None else int(idx),
                "dataset": None if dataset is None else str(dataset),
                "derivative": row.get("derivative", "unknown"),
                "file_path": row.get("file_path", ""),
                "status": status,
                # "done (migrated)": the .done was written by `neurodags migrate` (spec 6.2).
                "label": "done (migrated)" if migrated else status,
                "migrated": migrated,
                "error_path": _status_error_path(plan),
                "unlisted": [] if missing_source else _status_unlisted(plan),
                "planning_error": None if planning_error is None else str(planning_error),
            }
        )
    return rows


def _planning_error_message(rows: list[dict]) -> str | None:
    """One message naming the units a dry run could not plan (``None`` when there are none)."""
    failed = sorted(
        (r for r in rows if r["status"] == "planning-error"),
        key=lambda r: (*_status_index_key(r), r["derivative"]),
    )
    if not failed:
        return None
    lines = [
        f"{len(failed)} unit(s) could not be planned, so their status is unknown and the "
        "inspection is not complete (fix the configuration; see the log for tracebacks):"
    ]
    for r in failed:
        idx = r["index"] if r["index"] is not None else "?"
        lines.append(f"  [{r['derivative']}] [idx {idx}] {r['file_path']}: {r['planning_error']}")
    return "\n".join(lines)


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
        "incomplete": 0,
        "missing": 0,
        "errored": 0,
        "missing-source": 0,
        "planning-error": 0,
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
        # Subset of "done": units whose .done was written by `neurodags migrate`.
        per_deriv[deriv]["migrated"] = sum(1 for r in sub if r["migrated"])
    n_migrated = sum(1 for r in rows if r["migrated"])

    complete = (
        grand["missing"] == 0
        and grand["incomplete"] == 0
        and grand["errored"] == 0
        and grand["missing-source"] == 0
        and grand["planning-error"] == 0
    )
    planning_error_message = _planning_error_message(rows)
    n_unlisted = sum(len(r["unlisted"]) for r in rows)
    exit_code = 0 if complete or getattr(args, "exit_zero", False) else 1
    missing_source_indices = _status_indices(rows, ("missing-source",))
    n_not_in_list = enumeration.n_not_in_list if enumeration is not None else 0
    inputs_file = enumeration.inputs_file if enumeration is not None else None

    if fmt == "indices":
        # Incomplete units are recomputed like missing ones, so they count as missing here.
        if args.list_missing and not args.list_errors:
            wanted: tuple[str, ...] = ("missing", "incomplete")
        elif args.list_errors and not args.list_missing:
            wanted = ("errored",)
        else:
            wanted = ("missing", "incomplete", "errored")
        # An index with a planning error in any selected derivative is left out, even when
        # another derivative is missing there: rerunning it would fail the same way until
        # the configuration is fixed (select that other derivative alone to get it).
        unplannable = set(_status_indices(rows, ("planning-error",)))
        print(" ".join(str(i) for i in _status_indices(rows, wanted) if i not in unplannable))
        if planning_error_message:
            print(f"neurodags: error: {planning_error_message}", file=sys.stderr)
        return exit_code

    if fmt == "json":
        payload: dict = {
            "config": args.config,
            "n_files": total_files,
            "derivatives": per_deriv,
            "grand_total": {**grand, "migrated": n_migrated},
            "complete": complete,
            "missing_indices": _status_indices(rows, ("missing",)),
            "incomplete_indices": _status_indices(rows, ("incomplete",)),
            "errored_indices": _status_indices(rows, ("errored",)),
            "unlisted_files": n_unlisted,
            "inputs": inputs_file,
            "missing_source_indices": missing_source_indices,
            "not_in_list": n_not_in_list,
            "planning_error_indices": _status_indices(rows, ("planning-error",)),
        }
        if include_files:
            payload["files"] = [
                {
                    "index": r["index"],
                    "dataset": r["dataset"],
                    "file_path": r["file_path"],
                    "derivative": r["derivative"],
                    "status": r["status"],
                    "label": r["label"],
                    "error_path": r["error_path"],
                    "unlisted": r["unlisted"],
                    "planning_error": r["planning_error"],
                }
                for r in sorted(rows, key=lambda r: (*_status_index_key(r), r["derivative"]))
            ]
        print(json.dumps(payload, indent=2))
        if planning_error_message:
            print(f"neurodags: error: {planning_error_message}", file=sys.stderr)
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
    # "(migrated)": how many of the done units are "done (migrated)" (shown only then).
    mig_w = 12 if n_migrated else 0

    def _mig(count: int) -> str:
        return f"  {count:>{mig_w - 2}}" if mig_w else ""

    header = (
        f"{'Derivative':<{col_w}} {'total':>6}  {'done':>6}"
        + (f"  {'(migrated)':>{mig_w - 2}}" if mig_w else "")
        + f"  {'skipped':>8}  {'incomplete':>10}  {'missing':>8}  {'errored':>8}"
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
            f"{deriv:<{col_w}} {c['total']:>6}  {c['done']:>6}{_mig(c['migrated'])}"
            f"  {c['skipped']:>8}  {c['incomplete']:>10}  {c['missing']:>8}  {c['errored']:>8}"
        )

    print(sep)
    print(
        f"{'Total':<{col_w}} {grand['total']:>6}  {grand['done']:>6}{_mig(n_migrated)}"
        f"  {grand['skipped']:>8}  {grand['incomplete']:>10}  {grand['missing']:>8}"
        f"  {grand['errored']:>8}"
    )
    print(
        "\nNote: 'total' counts (derivative x source-file) pairs, not output files. "
        "'done' means the unit's .done manifest is valid and every file it lists exists. "
        "'incomplete' means output files exist without a valid .done (an interrupted "
        "write, or 0.3.x outputs not yet migrated with 'neurodags migrate'); the next run "
        "recomputes them. "
        "'skipped' means a .skip marker exists — the derivative was intentionally not "
        "computed for that source file (e.g. condition absent from the recording)."
    )
    if n_migrated:
        print(
            f"\n{n_migrated} of the done unit(s) are done (migrated): their .done was written "
            "by 'neurodags migrate' from 0.3.x outputs (column '(migrated)')."
        )
    if n_unlisted:
        print(
            f"\n{n_unlisted} file(s) match an output prefix but are not listed in its .done "
            "manifest; they are ignored (see --format json --include-files)."
        )
    _print_input_list_report()

    if planning_error_message:
        print(f"\n{planning_error_message}")
        print(f"neurodags: error: {planning_error_message}", file=sys.stderr)

    if grand["errored"]:
        print(f"\n{grand['errored']} error(s) found.", end="")
        if not args.list_errors:
            print("  Run with --list-errors for details.", end="")
        print()

    if grand["skipped"]:
        print(
            f"\n{grand['skipped']} derivative(s) skipped (not applicable for those source files)."
        )

    if grand["incomplete"]:
        print(f"{grand['incomplete']} derivative(s) incomplete (no valid .done manifest).", end="")
        if not args.list_missing:
            print("  Run with --list-missing for details.", end="")
        print()

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
                if r["migrated"]:
                    tag += " (migrated)"
                print(f"    {tag} {r['file_path']}")
                if show_error_path and r["error_path"]:
                    print(f"      error file: {r['error_path']}")

    if args.list_errors:
        _print_list("Errored files:", "errored", show_error_path=True)

    if args.list_missing:
        _print_list("Incomplete files:", "incomplete", show_error_path=False)
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
        help="Show how many units are done, skipped, incomplete, missing or errored per derivative.",
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
            "Print the file index and path of incomplete and missing (not yet computed) "
            "files below the summary table. With --format indices, restrict the output to "
            "incomplete and missing indices."
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
            "space-separated file indices that are missing, incomplete or errored — narrow "
            "with --list-missing / --list-errors — ready for --only-index or sbatch --array). "
            "The exit code is 1 whenever anything is missing, incomplete or errored, or "
            "could not be planned, in every format (unless --exit-zero)."
        ),
    )
    status_parser.add_argument(
        "--exit-zero",
        action="store_true",
        help=(
            "Exit 0 whenever the report was produced, even if files are missing, incomplete "
            "or errored "
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

    migrate_parser = subparsers.add_parser(
        "migrate",
        help="Upgrade a 0.3.x configuration and its outputs to naming_version 1 "
        "(preview by default).",
        description=(
            "Convert a neurodags 0.3.x pipeline: add naming_version and source_root, rewrite "
            "relative dataset paths, rename and move the existing outputs to the 0.4.0 "
            "layout and write .done manifests. Prints the plan and changes nothing unless "
            "--apply is given. Run once per output tree (per mount)."
        ),
    )
    migrate_parser.add_argument("config", help="Path to the 0.3.x pipeline YAML configuration.")
    migrate_parser.add_argument(
        "-d",
        "--datasets",
        default=None,
        help=(
            "Optional path to the 0.3.x datasets YAML that overrides the pipeline's 'datasets' "
            "key. Resolved relative to the current working directory (~ and $VARS are "
            "expanded). Relative paths inside it are read with the 0.3.x rule (relative to "
            "--legacy-cwd) and rewritten relative to the YAML's own directory."
        ),
    )
    migrate_parser.add_argument(
        "--mount",
        default=None,
        metavar="NAME",
        help="Mount point whose output tree to migrate (default: the pipeline's mount_point).",
    )
    migrate_parser.add_argument(
        "--apply", action="store_true", help="Execute the plan (default: preview only)."
    )
    migrate_parser.add_argument(
        "--report", default=None, metavar="PLAN.json", help="Write the full plan as JSON."
    )
    migrate_parser.add_argument(
        "--map",
        action="append",
        default=None,
        metavar="OLD=NEW",
        help="Rename a derivative while migrating (repeatable). Default: keep the YAML name.",
    )
    migrate_parser.add_argument(
        "--legacy-cwd",
        default=None,
        metavar="DIR",
        help="Directory the 0.3.x commands were run from (relative dataset paths were "
        "resolved against it). Default: the current working directory.",
    )
    migrate_parser.add_argument(
        "--no-synthesize",
        action="store_true",
        help="Do not write .done manifests for migrated outputs (they are recomputed).",
    )
    migrate_parser.add_argument(
        "--source-root",
        action="append",
        default=None,
        metavar="DATASET[:MOUNT]=PATH",
        help="source_root to write for a dataset (repeatable), instead of the suggestion. "
        "Required when the suggestion would be a filesystem root.",
    )
    migrate_parser.add_argument(
        "--upstream-map",
        action="append",
        default=None,
        metavar="OLDTOKEN=NEWNAME",
        help="Chained pipelines: an upstream derivative renamed on disk by its own migration "
        "(e.g. CleanPrep=clean_prep), so outputs named after the old upstream file match.",
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
        # Errors were raised when the configuration was loaded; this returns the warnings.
        name_warnings = check_pipeline_names(config)
        for name in getattr(args, "derivatives", None) or []:
            validate_derivative_name(name)
        layout_warnings = output_tree_warnings(datasets, mount_point)
    except (ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    defs = config.get("DerivativeDefinitions") or {}
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
    for warning in [*name_warnings, *layout_warnings]:
        print(f"warning: {warning}", file=sys.stderr)
    print(f"derivatives_defined ({len(defined)}): {', '.join(defined) or 'none'}")
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
        if not result.empty:
            message = _planning_error_message(_status_rows(result))
            if message:
                # The saved plan lacks these units: a failed inspection, not a plan.
                print(f"neurodags: error: {message}", file=sys.stderr)
                return 1
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
        script = _slurm_per_file(args.config, ordered, args.datasets)
        if args.output:
            Path(args.output).write_text(script)
            print(f"wrote {args.output}")
        else:
            print(script)
    elif args.pattern == "flat":
        script = _slurm_flat(args.config, ordered, args.datasets)
        if args.output:
            Path(args.output).write_text(script)
            print(f"wrote {args.output}")
        else:
            print(script)
    else:
        submit, worker = _slurm_chained(args.config, ordered, args.datasets)
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


def _cmd_migrate(args: argparse.Namespace) -> int:
    from neurodags import migrate

    plan = migrate.plan_migration(
        args.config,
        args.datasets,
        mount=args.mount,
        renames=args.map,
        legacy_cwd=args.legacy_cwd,
        synthesize=not args.no_synthesize,
        source_roots=args.source_root,
        upstream_renames=args.upstream_map,
    )
    print(migrate.format_preview(plan, apply=args.apply))
    if args.report:
        print(f"\nplan written to {migrate.write_report(plan, args.report)}")
    if plan.refusals:
        print(
            f"neurodags: error: migration refused ({len(plan.refusals)} problem(s) above); "
            "nothing was changed.",
            file=sys.stderr,
        )
        return 2
    if not args.apply or not plan.has_work:
        return 0
    result = migrate.apply_migration(plan)
    print()
    print(migrate.format_apply_summary(result))
    if args.report:
        migrate.write_report(plan, args.report, extra=result.to_dict())
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
    except (
        ConfigVersionError,
        DatasetConfigError,
        InputsError,
        NamingError,
        MigrationError,
    ) as exc:
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
    if args.command == "migrate":
        return _cmd_migrate(args)

    parser.error(f"Unknown command '{args.command}'")
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
