import inspect
import os
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import structlog
from joblib import Parallel, delayed

from neurodags.dag import collect_derivative_for_dataframe, run_derivative
from neurodags.datasets import get_datasets_and_mount_point_from_pipeline_configuration
from neurodags.definitions import DatasetConfig
from neurodags.derivatives import get_derivative, list_derivatives
from neurodags.derivatives.pipeline import register_derivatives_from_dict
from neurodags.iterators import get_all_files_from_pipeline_configuration
from neurodags.loaders import load_configuration
from neurodags.loggers import get_logger
from neurodags.nodes import get_node, list_nodes
from neurodags.nodes.loader import load_node_definitions
from neurodags.utils import (
    check_derivative_names,
    expand_path,
    get_path,
    resolve_dataset_paths,
)

log = get_logger(__name__)


@dataclass(slots=False)  # slots=True seems to break parallelization
class _FileJob:
    index: int
    dataset: str
    file_path: str
    dataset_config: dict[str, Any]
    mount_point: Any
    common_root: str | None
    is_node: bool
    node_name: str | None
    derivative_name: str | None
    derivative_definition: dict[str, Any] | None
    derivative_label: str
    dry_run: bool
    skip_errors: bool = False
    custom_node_paths: tuple[str, ...] = ()
    # Full DerivativeDefinitions map, re-registered inside each worker so that
    # uncached inter-derivative references resolve (workers start with an empty
    # derivative registry — see #18). Mirrors how custom_node_paths re-registers
    # nodes.
    derivative_definitions: dict[str, Any] = field(default_factory=dict)
    # Absolute spelling of ``file_path`` (resolved in the parent) used for all
    # processing; ``file_path`` keeps the discovered spelling for reporting.
    # Reused loky workers keep a stale cwd, so they must never see a relative
    # path (#35).
    source_path: str | None = None


@dataclass(slots=False)  # break parallelization?
class _FileResult:
    index: int
    dataset: str
    file_path: str
    derivative_label: str
    success: bool
    stage: str | None = None
    dry_run_payload: dict[str, Any] | None = None
    error_message: str | None = None
    error_type: str | None = None
    traceback: str | None = None


def _sanitize_reference_base(path: str) -> str:
    """Replace ``@`` with ``&`` in the filename component of *path*.

    ``@`` is the delimiter between source name and derivative name
    (``source@DerivativeName.ext``).  When a neurodags derivative is itself
    used as a source, its filename already contains ``@``.  Replacing that
    ``@`` with ``&`` preserves the one-``@``-per-filename invariant so that
    derivative paths remain unambiguous.

    Only the basename is rewritten; directory components are left untouched.
    """
    dirpart, basename = os.path.split(path)
    sanitized = basename.replace("@", "&")
    return os.path.join(dirpart, sanitized) if dirpart else sanitized


def _dataset_job_payload(dataset_config: Any, mount_point: Any) -> dict:
    """Serialize *dataset_config* for a worker with ``derivatives_path`` resolved.

    The active ``derivatives_path`` is selected and ``~``/``$VAR``-expanded in the
    parent process, so workers (which may be reused loky processes holding a stale
    environment) write exactly where the parent resolved (#28).
    """
    if hasattr(dataset_config, "model_dump"):
        payload = dataset_config.model_dump(mode="json")
    else:  # pragma: no cover - pydantic v1 fallback
        payload = dataset_config.dict()
    # Resolve the active path fields here and make them absolute against the
    # *parent's* cwd: reused loky workers keep the cwd (and environment) they
    # were started with, so a relative path would resolve elsewhere there (#35).
    for attr in ("file_pattern", "derivatives_path"):
        if payload.get(attr):
            payload[attr] = _absolute(
                get_path(
                    payload[attr],
                    mount_point=mount_point,
                    field=f"datasets.{payload.get('name')}.{attr}",
                )
            )
    if payload.get("exclude_pattern"):
        payload["exclude_pattern"] = _absolute(
            expand_path(
                payload["exclude_pattern"],
                field=f"datasets.{payload.get('name')}.exclude_pattern",
            )
        )
    return payload


def _absolute(path: Any) -> Any:
    """Return *path* made absolute against the current (parent) cwd; non-strings pass through."""
    if isinstance(path, str | os.PathLike):
        return os.path.abspath(path)
    return path


def _worker_source_paths(file_path: str, common_root: str | None) -> tuple[str, str | None]:
    """Absolute ``(file_path, common_root)`` for a worker job (#35).

    Both are made absolute against the parent's cwd, so the relative structure
    ``relpath(file_path, common_root)`` (and hence the derivative layout) is
    unchanged. An empty ``common_root`` (files directly in the cwd) means the
    cwd itself.
    """
    abs_root = os.path.abspath(common_root) if common_root is not None else None
    return os.path.abspath(file_path), abs_root


def _check_derivative_names(config_dict: dict) -> None:
    """Preflight: reject invalid or colliding derivative names (#42).

    Every name in ``DerivativeDefinitions`` is checked, not only the run set:
    dependencies are pulled in from outside the run set, and all derivatives of
    a pipeline share one derivatives directory across runs, so a collision
    between any two definitions corrupts cached outputs sooner or later. Pure
    computation, so it runs before anything is written.
    """
    defs = config_dict.get("DerivativeDefinitions") or {}
    mapping = check_derivative_names(defs)
    renamed = {name: file_name for name, file_name in mapping.items() if name != file_name}
    if renamed:
        log.debug("Derivative on-disk names differ from YAML names", renamed=renamed)


def _resolve_definition_paths(
    config_dict: dict, pipeline_configuration: Any
) -> tuple[list[Any], Path, tuple[str, ...]]:
    """Return ``(raw_paths, base_dir, absolute_paths)`` for ``new_definitions``.

    Relative entries resolve against the pipeline file's directory (or the cwd for
    an in-memory config). The absolute paths are what workers receive: fresh loky
    workers have no custom nodes and reused ones have a stale cwd (#35, #36).
    """
    new_definitions = config_dict.get("new_definitions")
    if not new_definitions:
        return [], Path.cwd(), ()
    if isinstance(new_definitions, str | os.PathLike):
        definition_paths = [new_definitions]
    elif isinstance(new_definitions, list | tuple | set):
        definition_paths = list(new_definitions)
    else:
        raise TypeError("new_definitions must be a string or list of paths")
    is_path_like = isinstance(pipeline_configuration, str | os.PathLike)
    base_dir = Path(pipeline_configuration).resolve().parent if is_path_like else Path.cwd()
    resolved_definition_paths = []
    for candidate in definition_paths:
        candidate_path = Path(candidate).expanduser()
        resolved_path = (
            (base_dir / candidate_path).resolve()
            if not candidate_path.is_absolute()
            else candidate_path.resolve()
        )
        resolved_definition_paths.append(str(resolved_path))
    return definition_paths, base_dir, tuple(resolved_definition_paths)


def _resolve_reference_base(
    file_path: str,
    dataset_config: DatasetConfig,
    common_root: str | None,
    mount_point: Any,
    create_dirs: bool = False,
) -> tuple[str, Path]:
    """Return the per-file output prefix (``reference_base``) as ``(str, Path)``.

    Pure path computation unless *create_dirs* is True, in which case the
    reference base's parent directory is created. Inspection paths (dry-run,
    ``status``, ``count-inputs``) must leave it False so they never write (#33).
    """
    derivatives_path = dataset_config.derivatives_path
    if derivatives_path:
        derivatives_path = get_path(
            derivatives_path,
            mount_point=mount_point,
            field=f"datasets.{dataset_config.name}.derivatives_path",
        )
        reference_base = os.path.relpath(file_path, start=common_root) if common_root else file_path
        if reference_base == os.curdir:
            # common_root is the file itself (e.g. a hand-passed root): keep the
            # output inside derivatives_path instead of "<derivatives>@Deriv" (#34).
            reference_base = os.path.basename(file_path)
        reference_base = os.path.join(derivatives_path, reference_base)
        reference_base = _sanitize_reference_base(reference_base)
        if create_dirs:
            os.makedirs(os.path.dirname(reference_base), exist_ok=True)
    else:
        reference_base = _sanitize_reference_base(file_path)
    return reference_base, Path(reference_base)


def _process_file_job(job: _FileJob) -> _FileResult:
    if job.custom_node_paths:
        load_node_definitions(job.custom_node_paths)

    # Re-register YAML derivatives in this (possibly fresh) worker process so that
    # uncached inter-derivative references (derivative: A.nc) resolve via the
    # registry instead of raising "Unknown derivative" (#18).
    if job.derivative_definitions:
        register_derivatives_from_dict({"DerivativeDefinitions": job.derivative_definitions})

    dataset_config = DatasetConfig(**job.dataset_config)
    source_path = job.source_path or job.file_path
    try:
        reference_base_str, reference_base_path = _resolve_reference_base(
            source_path,
            dataset_config,
            job.common_root,
            job.mount_point,
            create_dirs=not job.dry_run,
        )
    except Exception as setup_error:  # pragma: no cover - safety net
        return _FileResult(
            index=job.index,
            dataset=job.dataset,
            file_path=job.file_path,
            derivative_label=job.derivative_label,
            success=False,
            stage="setup",
            error_message=str(setup_error),
            error_type=type(setup_error).__name__,
            traceback=traceback.format_exc(),
        )

    try:
        if job.is_node and job.node_name:
            node_callable = get_node(job.node_name)
            node_parameters = inspect.signature(node_callable).parameters
            node_kwargs: dict[str, Any] = {}
            if "reference_base" in node_parameters:
                node_kwargs["reference_base"] = reference_base_str
            if "dataset_config" in node_parameters:
                node_kwargs["dataset_config"] = dataset_config
            if "mount_point" in node_parameters:
                node_kwargs["mount_point"] = job.mount_point
            node_callable(source_path, **node_kwargs)
            dry_payload: dict[str, Any] | None = None
        else:
            if job.derivative_definition is None or job.derivative_name is None:
                raise ValueError("Derivative job requires both definition and name.")
            dry_payload = None
            result = run_derivative(
                job.derivative_definition,
                job.derivative_name,
                source_path,
                reference_base=reference_base_path,
                dataset_config=dataset_config,
                mount_point=job.mount_point,
                dry_run=job.dry_run,
                skip_errors=job.skip_errors,
            )
            if job.dry_run:
                dry_payload = result

        return _FileResult(
            index=job.index,
            dataset=job.dataset,
            file_path=job.file_path,
            derivative_label=job.derivative_label,
            success=True,
            dry_run_payload=dry_payload,
        )
    except Exception as run_error:  # pragma: no cover - safety net
        return _FileResult(
            index=job.index,
            dataset=job.dataset,
            file_path=job.file_path,
            derivative_label=job.derivative_label,
            success=False,
            stage="run",
            error_message=str(run_error),
            error_type=type(run_error).__name__,
            traceback=traceback.format_exc(),
        )


def iterate_derivative_pipeline(
    pipeline_configuration: dict | str,
    derivative: Callable | str,
    datasets_configuration: dict | str | None = None,
    max_files_per_dataset: int | None = None,
    dry_run: bool = False,
    only_index: int | list[int] | None = None,
    raise_on_error: bool = False,
    skip_errors: bool = False,
    n_jobs: int | None = None,
    joblib_backend: str | None = None,
    joblib_prefer: str | None = None,
) -> pd.DataFrame | None:
    """
    Iterate over all files specified in the pipeline configuration and call a given function on each file.

    Parameters
    ----------
    pipeline_configuration : dict | str
        The pipeline configuration containing dataset information.
    derivative : callable | str
        The derivative or node to execute for each file. May be a registered derivative/node name, a
        DerivativeEntry, or a callable node.
    datasets_configuration : dict | str, optional
        Optional datasets configuration. If provided, it overrides the 'datasets' section
        of the pipeline configuration. A relative path string is resolved relative to the
        pipeline YAML's directory when ``pipeline_configuration`` is a path (relative to the
        current working directory otherwise), after expanding ``~`` and ``$VARS`` -- the same
        rule as the ``datasets:`` key. The ``neurodags`` CLI resolves ``-d/--datasets``
        against the working directory itself and passes an absolute path; pass an absolute
        path here to avoid ambiguity.
    max_files_per_dataset : int, optional
        Maximum number of files to process per dataset. If None, processes all files.
    dry_run : bool, optional
        When True, evaluate the pipeline without executing nodes and return a dataframe describing
        the plan.
    only_index : int | list[int], optional
        Restrict execution to a subset of files by index.
    raise_on_error : bool, optional
        Re-raise the first failure instead of continuing.
    skip_errors : bool, optional
        When True, skip files that have a prior ``.error`` marker from a previous failed run.
        The file will not be retried until the ``.error`` file is deleted manually.
    n_jobs : int, optional
        Number of workers to use with joblib. ``None`` or ``1`` keeps execution serial. Negative
        values follow joblib semantics (e.g. ``-1`` uses all cores).
    joblib_backend : str, optional
        Backend passed to joblib's ``Parallel`` (defaults to joblib's default when not provided).
    joblib_prefer : str, optional
        Preference hint for joblib backend selection (e.g. ``"threads"`` or ``"processes"``).

    Returns
    -------
    None | pandas.DataFrame
        Returns a dataframe with dry-run details when ``dry_run=True``; otherwise ``None``.
    """
    log.debug(
        "iterate_call_pipeline: called",
        pipeline_configuration=pipeline_configuration,
        datasets_configuration=datasets_configuration,
    )

    is_path_like = isinstance(pipeline_configuration, str | os.PathLike)
    config_dict = (
        load_configuration(pipeline_configuration) if is_path_like else pipeline_configuration
    )
    _check_derivative_names(config_dict)

    definition_paths, base_dir, custom_node_paths = _resolve_definition_paths(
        config_dict, pipeline_configuration
    )
    if definition_paths:
        load_node_definitions(definition_paths, base_dir=base_dir)

    if "DerivativeDefinitions" in config_dict:
        register_derivatives_from_dict(config_dict)
    else:
        log.warning(
            "No 'DerivativeDefinitions' found in the configuration. Skipping derivative registration."
        )

    datasets_configs, mount_point = get_datasets_and_mount_point_from_pipeline_configuration(
        pipeline_configuration, datasets_input=datasets_configuration
    )

    files_per_dataset, all_files, common_roots = get_all_files_from_pipeline_configuration(
        pipeline_configuration,
        datasets_input=datasets_configuration,
        max_files_per_dataset=None,  # gather all files, then filter locally
    )

    dry_run_collection: list[dict[str, Any]] = []
    if only_index is not None:
        index_filter = only_index if isinstance(only_index, list) else [only_index]
        filtered_files = [item for item in all_files if item[0] in index_filter]
        if len(filtered_files) != len(index_filter):
            found = [item[0] for item in filtered_files]
            missing = list(set(index_filter) - set(found))
            log.warning(
                "Some specified indices in only_index were not found in the files to process.",
                requested_indices=index_filter,
                found_indices=found,
            )
            log.warning("Missing indices will be ignored.", missing_indices=missing)
            log.warning("Proceeding with available indices.")
        all_files = filtered_files

    if max_files_per_dataset is not None:
        filtered_files = []
        dataset_file_count = dict.fromkeys(files_per_dataset.keys(), 0)
        for item in all_files:
            dataset_name = item[1]
            if dataset_file_count[dataset_name] < max_files_per_dataset:
                filtered_files.append(item)
                dataset_file_count[dataset_name] += 1
        all_files = filtered_files
        log.debug(
            "iterate_call_pipeline: applied max_files_per_dataset filter",
            max_files_per_dataset=max_files_per_dataset,
            total_files=len(all_files),
            per_dataset=dataset_file_count,
        )

    derivative_entry = None
    node_callable: Callable | None = None
    derivative_label: str | None = None
    allowed_derivatives = config_dict.get("DerivativeList")

    if isinstance(derivative, str):
        derivative_label = derivative
        if derivative in list_nodes():
            node_callable = get_node(derivative)
        elif derivative in list_derivatives():
            if allowed_derivatives is not None and derivative not in allowed_derivatives:
                raise KeyError(f"Derivative '{derivative}' is not enabled in DerivativeList.")
            derivative_entry = get_derivative(derivative)
        else:
            raise KeyError(f"Unknown derivative or node '{derivative}'")
    elif hasattr(derivative, "definition") and hasattr(derivative, "func"):
        derivative_entry = derivative
        derivative_label = derivative.name
    elif callable(derivative):
        node_callable = derivative
        derivative_label = getattr(derivative, "__name__", "<callable>")
    else:
        raise TypeError(
            "derivative must be a registered derivative name, node name, DerivativeEntry, or callable node"
        )
    # skip save=False derivatives
    if derivative_entry is not None and "save" in derivative_entry.definition:
        if derivative_entry.definition["save"] is False:
            log.info(
                "Derivative is marked with save=False; skipping execution.",
                derivative=derivative_label,
            )
            return None
    effective_n_jobs = n_jobs if n_jobs is not None else config_dict.get("n_jobs")
    if isinstance(effective_n_jobs, str):
        try:
            effective_n_jobs = int(effective_n_jobs)
        except ValueError as conversion_error:  # pragma: no cover - defensive
            raise ValueError(
                f"Invalid n_jobs value '{effective_n_jobs}'. Expected an integer."
            ) from conversion_error
    if effective_n_jobs == 0:
        effective_n_jobs = None
    backend = joblib_backend or config_dict.get("joblib_backend")
    prefer = joblib_prefer or config_dict.get("joblib_prefer")

    derivative_definitions_map = config_dict.get("DerivativeDefinitions", {}) or {}

    jobs: list[_FileJob] = []
    dataset_payloads: dict[str, dict] = {}
    for index, dataset_name, file_path in all_files:
        log.debug("Processing file", index=index, dataset=dataset_name, file_path=file_path)
        if dataset_name not in dataset_payloads:
            dataset_payloads[dataset_name] = _dataset_job_payload(
                datasets_configs[dataset_name], mount_point
            )
        dataset_payload = dataset_payloads[dataset_name]
        source_path, common_root = _worker_source_paths(file_path, common_roots.get(dataset_name))
        jobs.append(
            _FileJob(
                index=index,
                dataset=dataset_name,
                file_path=file_path,
                source_path=source_path,
                dataset_config=dataset_payload,
                mount_point=mount_point,
                common_root=common_root,
                is_node=node_callable is not None,
                node_name=derivative_label if node_callable is not None else None,
                derivative_name=derivative_entry.name if derivative_entry is not None else None,
                derivative_definition=(
                    derivative_entry.definition if derivative_entry is not None else None
                ),
                derivative_label=derivative_label or "<unknown>",
                dry_run=dry_run,
                skip_errors=skip_errors,
                custom_node_paths=custom_node_paths,
                derivative_definitions=derivative_definitions_map,
            )
        )

    log.info("Starting derivative processing", total_files=len(jobs))
    if not jobs:
        log.info("No files to process")
        return pd.DataFrame(dry_run_collection) if dry_run else None

    parallel_results: list[_FileResult]
    if effective_n_jobs in (None, 1):
        parallel_results = []
        for job in jobs:
            result = _process_file_job(job)
            parallel_results.append(result)
        # parallel_results = [_process_file_job(job) for job in jobs]
    else:
        parallel_kwargs = {"n_jobs": effective_n_jobs}
        if backend is not None:
            parallel_kwargs["backend"] = backend
        if prefer is not None:
            parallel_kwargs["prefer"] = prefer
        parallel_results = Parallel(**parallel_kwargs)(
            delayed(_process_file_job)(job) for job in jobs
        )

    for result in parallel_results:
        if result.success:
            log.info(
                "Processed file successfully",
                index=result.index,
                dataset=result.dataset,
                file_path=result.file_path,
                derivative=result.derivative_label,
            )
            if dry_run and result.dry_run_payload is not None:
                entry = {
                    "index": result.index,
                    "dataset": result.dataset,
                    "file_path": result.file_path,
                }
                entry.update(result.dry_run_payload)
                log.info("Dry run:", **entry)
                dry_run_collection.append(entry)
        else:
            message = (
                "Error running derivative" if result.stage == "run" else "Error processing file"
            )
            log.error(
                message,
                index=result.index,
                dataset=result.dataset,
                file_path=result.file_path,
                derivative=result.derivative_label,
                error=result.error_message,
                error_type=result.error_type,
                traceback=result.traceback,
            )
            if raise_on_error:
                error_summary = (
                    f"Failed to process file '{result.file_path}' for '{result.derivative_label}' "
                    f"(dataset '{result.dataset}', index {result.index}). "
                    f"{result.error_type}: {result.error_message}"
                )
                if result.traceback:
                    error_summary = f"{error_summary}\n{result.traceback}"
                raise RuntimeError(error_summary)

    log.info("Completed derivative processing", total_files=len(jobs))
    if dry_run:
        return pd.DataFrame(dry_run_collection)


def _derivative_topo_order(config_dict: dict, derivatives: list[str]) -> list[str]:
    """Return *derivatives* sorted so each derivative runs after those it depends on."""
    defs: dict[str, dict] = config_dict.get("DerivativeDefinitions", {}) or {}
    deriv_set = set(derivatives)

    deps: dict[str, set[str]] = {d: set() for d in derivatives}
    for deriv_name in derivatives:
        for step in defs.get(deriv_name, {}).get("nodes", []):
            ref = step.get("derivative", "")
            if ref and ref != "SourceFile":
                base = ref.split(".", 1)[0]
                if base in deriv_set:
                    deps[deriv_name].add(base)

    seen: set[str] = set()
    order: list[str] = []

    def visit(name: str) -> None:
        if name in seen:
            return
        for dep in sorted(deps.get(name, set())):
            visit(dep)
        seen.add(name)
        order.append(name)

    for name in sorted(derivatives):
        visit(name)

    return order


def _snapshot_pipeline_config(
    pipeline_configuration: str | os.PathLike,
    config_dict: dict,
    datasets_configs: dict,
    mount_point: Any,
    *,
    datasets_configuration: str | os.PathLike | None = None,
) -> None:
    """Copy pipeline config files into each dataset's ``derivatives_path/code/``.

    Called automatically by :func:`run_pipeline` before derivative execution
    (skipped on dry runs and when ``pipeline_configuration`` is a dict).

    Copied files
    ------------
    - The pipeline YAML itself.
    - Any ``new_definitions`` Python file(s) listed in the pipeline config.
    - The ``datasets`` YAML (resolved from ``datasets_configuration`` override
      or the ``datasets:`` field of the pipeline config, if it is a path).

    Additionally writes ``neurodags_env.json`` containing:

    - ``snapshot_time`` — ISO-8601 UTC timestamp.
    - ``neurodags_version`` — installed package version (if discoverable).
    - ``neurodags_git_commit`` — HEAD commit of the neurodags source repo (if
      the package is installed from a git checkout).

    Each unique ``derivatives_path/code/`` directory receives its own copy.
    Datasets that have no ``derivatives_path`` are silently skipped.  Any
    per-file error is logged as a warning but never propagates — snapshot
    failures must not block the pipeline.
    """
    import json
    import shutil
    from datetime import datetime, timezone

    pipeline_path = Path(pipeline_configuration).resolve()
    pipeline_dir = pipeline_path.parent

    # ---- collect source files --------------------------------------------------
    files_to_copy: list[Path] = [pipeline_path]

    new_defs = config_dict.get("new_definitions")
    if new_defs:
        if isinstance(new_defs, str | os.PathLike):
            new_defs = [new_defs]
        elif not isinstance(new_defs, list | tuple | set):
            new_defs = []
        for nd in new_defs:
            p = Path(nd)
            if not p.is_absolute():
                p = pipeline_dir / p
            p = p.resolve()
            if p.exists():
                files_to_copy.append(p)

    datasets_src = datasets_configuration or config_dict.get("datasets")
    if isinstance(datasets_src, str | os.PathLike):
        p = Path(expand_path(os.fspath(datasets_src), field="datasets"))
        if not p.is_absolute():
            p = pipeline_dir / p
        p = p.resolve()
        if p.exists():
            files_to_copy.append(p)

    # ---- neurodags version info ------------------------------------------------
    version_info: dict = {"snapshot_time": datetime.now(timezone.utc).isoformat()}
    try:
        import importlib.metadata

        version_info["neurodags_version"] = importlib.metadata.version("neurodags")
    except Exception:
        pass
    try:
        import subprocess

        _repo = Path(__file__).resolve().parents[2]
        commit = (
            subprocess.check_output(
                ["git", "-C", str(_repo), "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
                timeout=5,
            )
            .decode()
            .strip()
        )
        version_info["neurodags_git_commit"] = commit
    except Exception:
        pass

    # ---- write to each unique derivatives_path/code/ --------------------------
    seen_code_dirs: set[str] = set()
    for dataset_config in datasets_configs.values():
        if not dataset_config.derivatives_path:
            continue
        if getattr(dataset_config, "skip", False):
            continue
        derivatives_path = get_path(
            dataset_config.derivatives_path,
            mount_point=mount_point,
            field=f"datasets.{dataset_config.name}.derivatives_path",
        )
        code_dir = Path(derivatives_path) / "code"
        if str(code_dir) in seen_code_dirs:
            continue
        seen_code_dirs.add(str(code_dir))
        code_dir.mkdir(parents=True, exist_ok=True)

        for src in files_to_copy:
            dst = code_dir / src.name
            try:
                shutil.copy2(src, dst)
                log.debug("Snapshotted config file", src=str(src), dst=str(dst))
            except Exception as exc:
                log.warning("Failed to snapshot config file", src=str(src), error=str(exc))

        env_path = code_dir / "neurodags_env.json"
        try:
            with open(env_path, "w") as fh:
                json.dump(version_info, fh, indent=2)
            log.debug("Wrote neurodags_env.json", path=str(env_path))
        except Exception as exc:
            log.warning("Failed to write neurodags_env.json", error=str(exc))


def run_pipeline(
    pipeline_configuration: dict | str,
    datasets_configuration: dict | str | None = None,
    derivatives: list[str] | None = None,
    max_files_per_dataset: int | None = None,
    dry_run: bool = False,
    only_index: int | list[int] | None = None,
    raise_on_error: bool = False,
    skip_errors: bool = False,
    n_jobs: int | None = None,
    joblib_backend: str | None = None,
    joblib_prefer: str | None = None,
) -> pd.DataFrame | None:
    """
    Execute multiple derivatives from a pipeline configuration in dependency order.

    Parameters
    ----------
    pipeline_configuration : dict | str
        The pipeline configuration containing dataset information.
    datasets_configuration : dict | str, optional
        Optional datasets configuration. If provided, it overrides the 'datasets' section
        of the pipeline configuration. A relative path string is resolved relative to the
        pipeline YAML's directory when ``pipeline_configuration`` is a path (relative to the
        current working directory otherwise), after expanding ``~`` and ``$VARS`` -- the same
        rule as the ``datasets:`` key. The ``neurodags`` CLI resolves ``-d/--datasets``
        against the working directory itself and passes an absolute path; pass an absolute
        path here to avoid ambiguity.
    derivatives : list[str], optional
        List of derivatives to execute. If None, uses 'DerivativeList' from the configuration.
        Derivatives are automatically sorted in topological order based on inter-derivative
        dependencies before execution.
    max_files_per_dataset : int, optional
        Maximum number of files to process per dataset.
    dry_run : bool, optional
        When True, evaluate the pipeline without executing nodes and return a dataframe.
    only_index : int | list[int], optional
        Restrict execution to a subset of files by index.
    raise_on_error : bool, optional
        Re-raise the first failure instead of continuing.
    skip_errors : bool, optional
        When True, skip files that have a prior ``.error`` marker from a previous failed run.
        The file will not be retried until the ``.error`` file is deleted manually.
    n_jobs : int, optional
        Number of workers for joblib parallelism.
    joblib_backend : str, optional
        Backend passed to joblib's ``Parallel``.
    joblib_prefer : str, optional
        Preference hint for joblib backend selection.

    Returns
    -------
    None | pandas.DataFrame
        Returns a combined dataframe with dry-run details when ``dry_run=True``; otherwise ``None``.
    """
    config = (
        load_configuration(pipeline_configuration)
        if isinstance(pipeline_configuration, str | os.PathLike)
        else pipeline_configuration
    )

    # Preflight (#42): colliding derivative on-disk names fail before anything is
    # written, instead of silently sharing cached files.
    _check_derivative_names(config)

    # Preflight: resolve ~/$VARS in every active dataset's path fields before
    # anything is written (code snapshot, derivative dirs), so an unset variable
    # fails loudly here instead of creating a literal "$VAR" directory or being
    # swallowed by the per-file error records (#28).
    try:
        _ds_configs, _mount = get_datasets_and_mount_point_from_pipeline_configuration(
            pipeline_configuration, datasets_input=datasets_configuration
        )
    except Exception as _load_err:
        # Loading errors resurface (and raise) when each derivative loads the
        # datasets itself; only the snapshot is skipped here, as before.
        log.warning("Could not load datasets before execution", error=str(_load_err))
        _ds_configs = None
    if _ds_configs is not None:
        resolve_dataset_paths(_ds_configs, _mount)

    if (
        not dry_run
        and _ds_configs is not None
        and isinstance(pipeline_configuration, str | os.PathLike)
    ):
        try:
            _snapshot_pipeline_config(
                pipeline_configuration,
                config,
                _ds_configs,
                _mount,
                datasets_configuration=datasets_configuration,
            )
        except Exception as _snap_err:
            log.warning("Config snapshot failed — pipeline continues", error=str(_snap_err))

    if derivatives is None:
        derivatives = config.get("DerivativeList", [])

    if not derivatives:
        log.warning("No derivatives specified or enabled in 'DerivativeList'.")
        return None

    ordered = _derivative_topo_order(config, derivatives)
    log.info("Derivative execution order", order=ordered)

    dry_run_results = []
    for derivative in ordered:
        result = iterate_derivative_pipeline(
            pipeline_configuration=pipeline_configuration,
            derivative=derivative,
            datasets_configuration=datasets_configuration,
            max_files_per_dataset=max_files_per_dataset,
            dry_run=dry_run,
            only_index=only_index,
            raise_on_error=raise_on_error,
            skip_errors=skip_errors,
            n_jobs=n_jobs,
            joblib_backend=joblib_backend,
            joblib_prefer=joblib_prefer,
        )
        if dry_run and result is not None:
            dry_run_results.append(result)

    if dry_run:
        return pd.concat(dry_run_results, ignore_index=True) if dry_run_results else pd.DataFrame()
    return None


def build_derivative_dataframe(
    pipeline_configuration: dict | str,
    *,
    datasets_configuration: dict | str | None = None,
    include_derivatives: list[str] | None = None,
    max_files_per_dataset: int | None = None,
    only_index: int | list[int] | None = None,
    output_format: Literal["wide", "long"] = "wide",
    preserve_complex_values: bool = False,
    raise_on_error: bool = False,
    n_jobs: int | None = None,
) -> pd.DataFrame:
    """
    Assemble a dataframe by collecting derivative artifacts for every file in a pipeline configuration.

    Parameters
    ----------
    pipeline_configuration : dict | str
        Pipeline configuration or path to it.
    datasets_configuration : dict | str, optional
        Optional datasets configuration. If provided, it overrides the 'datasets' section
        of the pipeline configuration. A relative path string is resolved relative to the
        pipeline YAML's directory when ``pipeline_configuration`` is a path (relative to the
        current working directory otherwise), after expanding ``~`` and ``$VARS`` -- the same
        rule as the ``datasets:`` key. The ``neurodags`` CLI resolves ``-d/--datasets``
        against the working directory itself and passes an absolute path; pass an absolute
        path here to avoid ambiguity.
    include_derivatives : list[str], optional
        Restrict collection to this explicit subset of derivative names.
    max_files_per_dataset : int, optional
        Limit the number of files processed per dataset.
    only_index : int | list[int], optional
        Restrict collection to specific indices (matching iterate_derivative_pipeline behaviour).
    output_format : {"wide", "long"}, optional
        Shape of the returned dataframe. ``"wide"`` retains one row per file with derivative columns,
        while ``"long"`` yields one row per collected derivative value without synthesising missing columns.
    preserve_complex_values : bool, optional
        When True, derivative artifacts keep their original (potentially nested) Python structures
        instead of being flattened or coerced into simple dataframe columns.
    raise_on_error : bool, optional
        Re-raise exceptions encountered while collecting derivatives; defaults to False.
    n_jobs : int, optional
        Number of parallel workers for file-level collection. ``-1`` uses all cores, ``1`` or
        ``None`` runs serially. Parallelism is per-file using separate processes (loky) to avoid
        HDF5/NetCDF4 thread-safety issues — both wide and long output formats are fully supported.

    Returns
    -------
    pandas.DataFrame
        A dataframe with one row per processed file and columns derived from collected derivative artifacts.
    """

    log.debug(
        "build_derivative_dataframe: called",
        pipeline_configuration=pipeline_configuration,
        datasets_configuration=datasets_configuration,
    )

    config_dict = (
        load_configuration(pipeline_configuration)
        if isinstance(pipeline_configuration, str)
        else pipeline_configuration
    )

    if "DerivativeDefinitions" not in config_dict:
        log.warning(
            "No 'DerivativeDefinitions' found in the configuration. Returning empty dataframe."
        )
        return pd.DataFrame()
    _check_derivative_names(config_dict)

    # Load custom node definitions so save=False derivatives using custom nodes work.
    # The absolute paths are also sent to the workers, which load them too (#36).
    definition_paths, base_dir, custom_node_paths = _resolve_definition_paths(
        config_dict, pipeline_configuration
    )
    if definition_paths:
        load_node_definitions(definition_paths, base_dir=base_dir)

    register_derivatives_from_dict(config_dict)

    derivative_definitions: dict[str, dict] = config_dict.get("DerivativeDefinitions", {}) or {}
    selected_derivatives: list[str] = []

    include_derivatives = (
        config_dict.get("DerivativeList", [])
        if include_derivatives is None
        else include_derivatives
    )

    for derivative_name, derivative_def in derivative_definitions.items():
        if include_derivatives is not None and derivative_name not in include_derivatives:
            continue
        if not derivative_def.get("for_dataframe", False):
            continue
        selected_derivatives.append(derivative_name)
    if include_derivatives:
        missing_derivatives = sorted(set(include_derivatives) - set(selected_derivatives))
        if missing_derivatives:
            log.warning(
                "Some requested derivatives are either undefined or flagged out of dataframe collection.",
                missing_derivatives=missing_derivatives,
            )

    if not selected_derivatives:
        log.warning(
            "No derivatives eligible for dataframe collection were found. Returning empty dataframe."
        )
        return pd.DataFrame(columns=["index", "dataset", "file_path"])

    datasets_configs, mount_point = get_datasets_and_mount_point_from_pipeline_configuration(
        pipeline_configuration, datasets_input=datasets_configuration
    )
    # Preflight (#28): fail on unresolved ~/$VARS before any directory is touched.
    resolve_dataset_paths(datasets_configs, mount_point)

    _, _, common_roots = get_all_files_from_pipeline_configuration(
        pipeline_configuration, datasets_input=datasets_configuration, max_files_per_dataset=None
    )

    files_per_dataset, all_files, _ = get_all_files_from_pipeline_configuration(
        pipeline_configuration,
        datasets_input=datasets_configuration,
        max_files_per_dataset=max_files_per_dataset,
    )

    log.debug(
        "build_derivative_dataframe: enumerated files",
        total_files=len(all_files),
        per_dataset=files_per_dataset,
    )

    if only_index is not None:
        index_filter = only_index if isinstance(only_index, list) else [only_index]
        filtered = [item for item in all_files if item[0] in index_filter]
        if len(filtered) != len(index_filter):
            detected = [item[0] for item in filtered]
            missing = list(set(index_filter) - set(detected))
            log.warning(
                "Some indices requested for dataframe collection were not found.",
                requested=index_filter,
                missing=missing,
            )
        all_files = filtered

    if output_format not in {"wide", "long"}:
        raise ValueError(f"Unsupported output_format '{output_format}'. Expected 'wide' or 'long'.")

    collect_long_rows = output_format == "long"

    n_files = len(all_files)
    effective_n_jobs = n_jobs if (n_jobs is not None and n_jobs != 1) else 1

    # Serialize DatasetConfig objects once — joblib workers reconstruct them.
    # derivatives_path is resolved here, in the parent (#28).
    dataset_payloads = {
        name: _dataset_job_payload(datasets_configs[name], mount_point)
        for name in {dataset_name for _, dataset_name, _ in all_files}
    }

    job_args = [
        (
            index,
            dataset_name,
            file_path,
            dataset_payloads[dataset_name],
            *_worker_source_paths(file_path, common_roots.get(dataset_name)),
            mount_point,
            selected_derivatives,
            derivative_definitions,
            preserve_complex_values,
            collect_long_rows,
            raise_on_error,
            n_files,
            custom_node_paths,
        )
        for index, dataset_name, file_path in all_files
    ]

    if effective_n_jobs == 1:
        results = [_collect_dataframe_file(*args) for args in job_args]
    else:
        results = Parallel(n_jobs=effective_n_jobs, backend="loky")(
            delayed(_collect_dataframe_file)(*args) for args in job_args
        )

    rows: list[dict[str, Any]] = []
    long_rows: list[dict[str, Any]] = []
    for row, file_long_rows in results:
        if row is not None:
            rows.append(row)
        long_rows.extend(file_long_rows)

    if output_format == "long":
        if not long_rows:
            return pd.DataFrame(columns=["index", "dataset", "file_path", "derivative", "value"])
        return pd.DataFrame(long_rows)

    return pd.DataFrame(rows)


def _build_reference_base(
    file_path: str, dataset_config, common_root: str | None, mount_point: Path | None
) -> Path:
    # Read-only: dataframe collection mostly reads cached artifacts. When it must
    # compute, run_derivative creates the output directory itself (#33).
    _, reference_base_path = _resolve_reference_base(
        file_path, dataset_config, common_root, mount_point
    )
    return reference_base_path


def _collect_dataframe_file(
    index: int,
    dataset_name: str,
    file_path: str,
    dataset_config_dict: dict,
    source_path: str,
    common_root: str | None,
    mount_point: Any,
    selected_derivatives: list[str],
    derivative_definitions: dict,
    preserve_complex_values: bool,
    collect_long_rows: bool,
    raise_on_error: bool,
    n_files: int,
    custom_node_paths: tuple[str, ...] = (),
) -> tuple[dict | None, list[dict]]:
    """Worker function for parallel dataframe collection — one call per file.

    ``file_path`` is the discovered spelling reported in the row; ``source_path``
    and ``common_root`` are absolute (resolved in the parent, #35) and used for
    all processing.
    """
    from neurodags.loggers import configure_logging

    configure_logging()  # no-op in main process; configures structlog in fresh loky workers
    from neurodags.definitions import DatasetConfig

    # Fresh loky workers have no custom nodes: load them so uncached / save=False
    # derivatives using them can be computed here (#36). No-op once loaded.
    if custom_node_paths:
        load_node_definitions(custom_node_paths)

    # Re-register YAML derivatives in this (possibly fresh) worker so uncached
    # inter-derivative references resolve via the registry rather than raising
    # "Unknown derivative" (#18). Idempotent (override=True) in the main process.
    if derivative_definitions:
        register_derivatives_from_dict({"DerivativeDefinitions": derivative_definitions})

    dataset_config = DatasetConfig(**dataset_config_dict)

    with structlog.contextvars.bound_contextvars(
        file_path=file_path, dataset=dataset_name, index=index
    ):
        log.info("Collecting dataframe row", total=n_files)
        try:
            reference_base = _build_reference_base(
                source_path, dataset_config, common_root, mount_point
            )
            row: dict[str, Any] = {"index": index, "dataset": dataset_name, "file_path": file_path}
            # Which derivative wrote each column, so cross-derivative collisions can be reported.
            column_owner: dict[str, str] = dict.fromkeys(row, "<row metadata>")
            long_rows: list[dict] = []
            for derivative_name in selected_derivatives:
                derivative_def = derivative_definitions.get(derivative_name, {}) or {}
                with structlog.contextvars.bound_contextvars(derivative=derivative_name):
                    try:
                        derivative_values = collect_derivative_for_dataframe(
                            derivative_def,
                            derivative_name,
                            source_path,
                            reference_base=reference_base,
                            dataset_config=dataset_config,
                            mount_point=mount_point,
                            preserve_complex_values=preserve_complex_values,
                            flatten_xarray_artifacts=not preserve_complex_values,
                        )
                        if derivative_values:
                            collisions = sorted(set(derivative_values) & set(column_owner))
                            if collisions:
                                log.warning(
                                    "Dataframe column collision: derivative produces columns "
                                    "already present in the row; earlier values are overwritten",
                                    columns=collisions,
                                    previous_owners=sorted({column_owner[c] for c in collisions}),
                                )
                            row.update(derivative_values)
                            column_owner.update(dict.fromkeys(derivative_values, derivative_name))
                            if collect_long_rows:
                                for column_name, value in derivative_values.items():
                                    long_rows.append(
                                        {
                                            "index": index,
                                            "dataset": dataset_name,
                                            "file_path": file_path,
                                            "derivative": column_name,
                                            "value": value,
                                        }
                                    )
                    except Exception as derivative_error:
                        log.error(
                            "Error collecting derivative for dataframe",
                            error=str(derivative_error),
                            exc_info=True,
                        )
                        if raise_on_error:
                            raise
                        row[f"{derivative_name}__error"] = str(derivative_error)
                        if collect_long_rows:
                            long_rows.append(
                                {
                                    "index": index,
                                    "dataset": dataset_name,
                                    "file_path": file_path,
                                    "derivative": f"{derivative_name}__error",
                                    "value": str(derivative_error),
                                }
                            )
            log.debug("Collected dataframe row", collected_derivatives=len(selected_derivatives))
            return row, long_rows
        except Exception as error:
            log.error("Error collecting dataframe row", error=str(error), exc_info=True)
            if raise_on_error:
                raise
            return None, []


if __name__ == "__main__":  # pragma: no cover
    # Parse args and run iterate_derivative_pipeline
    import argparse

    parser = argparse.ArgumentParser(description="Iterate derivative pipeline.")
    parser.add_argument(
        "config", type=str, help="Path to the pipeline configuration file (YAML or JSON)."
    )
    parser.add_argument(
        "-d",
        "--datasets",
        type=str,
        default=None,
        help="Optional path to the datasets configuration file.",
    )
    parser.add_argument(
        "--max_files_per_dataset",
        type=int,
        default=None,
        help="Maximum number of files to process per dataset.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="If set, perform a dry run without actual processing.",
    )
    parser.add_argument(
        "--only_index",
        type=int,
        nargs="*",
        default=None,
        help="Only process files with these indices.",
    )
    parser.add_argument(
        "--raise_on_error",
        action="store_true",
        help="If set, raise exceptions on errors instead of logging them.",
    )
    parser.add_argument(
        "--n-jobs",
        dest="n_jobs",
        type=int,
        default=None,
        help="Number of parallel workers to use (joblib semantics; default serial).",
    )
    parser.add_argument(
        "--joblib-backend",
        dest="joblib_backend",
        type=str,
        default=None,
        help="Joblib backend to use (e.g. 'loky', 'threading').",
    )
    parser.add_argument(
        "--joblib-prefer",
        dest="joblib_prefer",
        type=str,
        default=None,
        help="Joblib execution preference hint (e.g. 'processes', 'threads').",
    )
    parser.add_argument(
        "--make_final_dataframe",
        action="store_true",
        help="If set, build the final derivative dataframe after processing.",
    )
    parser.add_argument(
        "--dataframe_output",
        type=str,
        default=None,
        help="Optional path where the dataframe should be written (CSV by default, or Parquet if the extension is .parquet).",
    )
    parser.add_argument(
        "--dataframe_derivatives",
        nargs="*",
        default=None,
        help="Optional subset of derivative names to include when building the dataframe.",
    )
    parser.add_argument(
        "--dataframe_format",
        choices=("wide", "long"),
        default="wide",
        help="Layout of the generated dataframe; 'long' emits one row per collected derivative value.",
    )
    parser.add_argument(
        "--preserve_complex_values",
        action="store_true",
        help="Retain complex derivative artifacts without flattening or converting them for dataframe storage.",
    )
    args = parser.parse_args()

    from neurodags.loaders import load_configuration

    pipeline_configuration_dict = load_configuration(args.config)

    derivative_list = pipeline_configuration_dict.get("DerivativeList", [])
    if not derivative_list:
        log.error(
            "No derivatives specified in the pipeline configuration under 'DerivativeList'. Exiting."
        )
        exit(1)

    if args.dry_run:
        log.info("Performing dry run of derivative pipeline", derivatives=derivative_list)
        dry_run_results = []

    if not args.make_final_dataframe:  # and args.dataframe_output:
        for derivative in derivative_list:
            output = iterate_derivative_pipeline(
                pipeline_configuration=args.config,
                derivative=derivative,
                datasets_configuration=args.datasets,
                max_files_per_dataset=args.max_files_per_dataset,
                dry_run=args.dry_run,
                only_index=args.only_index,
                raise_on_error=True if args.raise_on_error else False,
                n_jobs=args.n_jobs,
                joblib_backend=args.joblib_backend,
                joblib_prefer=args.joblib_prefer,
            )

            if args.dry_run and output is not None:
                dry_run_results.append((derivative, output))
        if args.dry_run:
            # merge dry run results
            # simple concatenation for now
            dry_run_results = [x for x in dry_run_results if x is not None]
            df = pd.concat([result for _, result in dry_run_results], ignore_index=True)
            # save
            if args.dataframe_output:
                output_path = Path(args.dataframe_output)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                if output_path.suffix.lower() == ".parquet":
                    try:
                        df.to_parquet(output_path, index=False)
                    except (ImportError, ValueError) as parquet_error:
                        log.warning(
                            "Parquet export failed; saving as CSV instead.",
                            path=str(output_path),
                            error=str(parquet_error),
                        )
                        csv_fallback = output_path.with_suffix(".csv")
                        df.to_csv(csv_fallback, index=False)
                        log.info(
                            "Saved dry run results",
                            path=str(csv_fallback),
                            rows=len(df),
                            columns=list(df.columns),
                        )
                    else:
                        log.info(
                            "Saved dry run results",
                            path=str(output_path),
                            rows=len(df),
                            columns=list(df.columns),
                        )
                else:
                    df.to_csv(output_path, index=False)
                    log.info(
                        "Saved dry run results",
                        path=str(output_path),
                        rows=len(df),
                        columns=list(df.columns),
                    )
            else:
                df.to_csv("dry_run_results.csv", index=False)
                log.info(
                    "Saved dry run results to dry_run_results.csv",
                    rows=len(df),
                    columns=list(df.columns),
                )
            log.info("Dry run completed", total_rows=len(df))

    if args.make_final_dataframe:
        log.info("Building derivative dataframe", derivatives=args.dataframe_derivatives)
        dataframe = build_derivative_dataframe(
            pipeline_configuration=args.config,
            datasets_configuration=args.datasets,
            include_derivatives=args.dataframe_derivatives,
            max_files_per_dataset=args.max_files_per_dataset,
            only_index=args.only_index,
            output_format=args.dataframe_format,
            preserve_complex_values=args.preserve_complex_values,
            raise_on_error=True if args.raise_on_error else False,
        )
        if args.dataframe_output:
            output_path = Path(args.dataframe_output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if output_path.suffix.lower() == ".parquet":
                try:
                    dataframe.to_parquet(output_path, index=False)
                except (ImportError, ValueError) as parquet_error:
                    log.warning(
                        "Parquet export failed; saving as CSV instead.",
                        path=str(output_path),
                        error=str(parquet_error),
                    )
                    csv_fallback = output_path.with_suffix(".csv")
                    dataframe.to_csv(csv_fallback, index=False)
                    log.info(
                        "Saved derivative dataframe",
                        path=str(csv_fallback),
                        rows=len(dataframe),
                        columns=list(dataframe.columns),
                    )
                else:
                    log.info(
                        "Saved derivative dataframe",
                        path=str(output_path),
                        rows=len(dataframe),
                        columns=list(dataframe.columns),
                    )
            else:
                dataframe.to_csv(output_path, index=False)
                log.info(
                    "Saved derivative dataframe",
                    path=str(output_path),
                    rows=len(dataframe),
                    columns=list(dataframe.columns),
                )
        else:
            log.info(
                "Derivative dataframe built (not saved to disk)",
                rows=len(dataframe),
                columns=list(dataframe.columns),
            )
            print(dataframe)
