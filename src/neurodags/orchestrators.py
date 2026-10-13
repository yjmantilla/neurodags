import inspect
import os
import traceback
from collections.abc import Callable, Iterable, Mapping
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
from neurodags.iterators import (  # noqa: F401 - re-exported for compatibility
    InputEntry,
    InputEnumeration,
    enumerate_inputs,
    get_all_files_from_pipeline_configuration,
)
from neurodags.loaders import load_pipeline_configuration
from neurodags.loggers import get_logger
from neurodags.naming import ColumnCollisionError, error_column, validate_derivative_name
from neurodags.nodes import get_node, list_nodes
from neurodags.nodes.loader import load_node_definitions
from neurodags.outputs import (
    check_legacy_chained_outputs,
    check_output_collisions,
    entry_reference_base,
)
from neurodags.utils import (
    expand_path,
    resolve_dataset_field,
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
    # Absolute reference base of the source (spec 4.1), resolved in the parent with
    # neurodags.outputs.entry_reference_base; outputs are "<reference_base>@<name>...".
    reference_base: str
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
    # Source path relative to source_root (POSIX), recorded in the .done manifest.
    source_rel: str | None = None


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
    # The entry of a frozen input list whose file no longer exists (spec §8.6).
    missing_source: bool = False


def _missing_source_result(entry: InputEntry, inputs_file: str | None, label: str) -> _FileResult:
    """Per-file error record for a listed input that no longer exists (not processed)."""
    return _FileResult(
        index=entry.index,
        dataset=entry.dataset,
        file_path=entry.path,
        derivative_label=label,
        success=False,
        stage="source",
        error_message=(
            f"missing-source: {entry.dataset}/{entry.rel} is listed in {inputs_file} but "
            f"no longer exists ({entry.path}); not processed."
        ),
        error_type="MissingSourceError",
        missing_source=True,
    )


def _dataset_job_payload(dataset_config: Any, mount_point: Any) -> dict:
    """Serialize *dataset_config* for a worker with its path fields resolved.

    The active value of every path field (``source_root``, ``file_pattern``,
    ``exclude_pattern``, ``derivatives_path``) is selected, ``~``/``$VAR``-expanded and
    resolved against the dataset's origin directory in the parent process, so workers
    (which may be reused loky processes holding a stale environment and cwd) see
    absolute paths and write exactly where the parent resolved (#28, #35).
    """
    if hasattr(dataset_config, "model_dump"):
        payload = dataset_config.model_dump(mode="json")
    else:  # pragma: no cover - pydantic v1 fallback
        payload = dataset_config.dict()
    for attr in ("source_root", "file_pattern", "exclude_pattern", "derivatives_path"):
        if payload.get(attr):
            payload[attr] = _absolute(resolve_dataset_field(dataset_config, attr, mount_point))
    return payload


def _absolute(path: Any) -> Any:
    """Return *path* made absolute against the current (parent) cwd; non-strings pass through."""
    if isinstance(path, str | os.PathLike):
        return os.path.abspath(path)
    return path


def _reference_bases(
    enumeration: InputEnumeration,
    datasets_configs: Mapping[str, Any],
    mount_point: Any,
    names: Iterable[str] = (),
) -> dict[int, str]:
    """Absolute reference base per enumerated input, checked for collisions (spec 4.1, 4.3).

    Uses the one layout rule of :func:`neurodags.outputs.entry_reference_base` (from
    ``source_root``, never from the discovered set). Nothing is written. Raises
    :class:`neurodags.utils.DatasetConfigError` when two sources map to the same outputs,
    or when outputs of the derivatives *names* are still in the pre-0.4.0 ``&`` layout
    (:func:`neurodags.outputs.check_legacy_chained_outputs`; scanned once per
    enumeration, which every derivative of one invocation shares).
    """
    entries = enumeration.entries
    bases = {
        entry.index: entry_reference_base(entry, datasets_configs[entry.dataset], mount_point)
        for entry in entries
    }
    check_output_collisions(
        (f"{entry.dataset}: {entry.rel}", bases[entry.index]) for entry in entries
    )
    names = list(names)
    if names and not enumeration.legacy_checked:
        check_legacy_chained_outputs(datasets_configs, mount_point, names, entries)
        enumeration.legacy_checked = True
    return bases


def _resolve_definition_paths(
    config_dict: dict, pipeline_configuration: Any, memory_base_dir: Any = None
) -> tuple[list[Any], Path, tuple[str, ...]]:
    """Return ``(raw_paths, base_dir, absolute_paths)`` for ``new_definitions``.

    Relative entries resolve against the pipeline file's directory (or, for an
    in-memory config, *memory_base_dir* when given, else the cwd). The absolute paths
    are what workers receive: fresh loky workers have no custom nodes and reused ones
    have a stale cwd (#35, #36).
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
    if is_path_like:
        base_dir = Path(pipeline_configuration).resolve().parent
    elif memory_base_dir is not None:
        base_dir = Path(os.path.abspath(os.fspath(memory_base_dir)))
    else:
        base_dir = Path.cwd()
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
    reference_base_str, reference_base_path = job.reference_base, Path(job.reference_base)
    try:
        if job.is_node and not job.dry_run:
            # Direct node jobs write next to reference_base themselves.
            os.makedirs(os.path.dirname(reference_base_str), exist_ok=True)
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
                source_dataset=job.dataset,
                source_rel=job.source_rel,
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
    *,
    base_dir: str | os.PathLike | None = None,
    inputs: str | os.PathLike | None = None,
    enumeration: InputEnumeration | None = None,
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
        Maximum number of files to process per dataset: the first N in sorted order, applied
        before indexing (so indices refer to the limited list). If None, processes all
        files. Cannot be combined with ``inputs``.
    dry_run : bool, optional
        When True, evaluate the pipeline without executing nodes and return a dataframe describing
        the plan.
    only_index : int | list[int], optional
        Restrict execution to a subset of files by index. A negative or unknown index raises
        :class:`neurodags.iterators.IndexSelectionError` (a ``ValueError``).
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
    base_dir : str | os.PathLike, optional
        Directory that relative paths of in-memory configurations resolve against: the
        ``source_root``/``file_pattern``/``exclude_pattern``/``derivatives_path`` of a
        datasets mapping passed as a dict (inline in a dict ``pipeline_configuration``, or
        as ``datasets_configuration``), a relative ``datasets:`` reference and
        ``new_definitions`` of a dict pipeline. Datasets read from a YAML file resolve
        against that file's directory instead. Without it, relative dataset paths in a
        mapping are an error.
    inputs : str | os.PathLike, optional
        Frozen input list written by ``neurodags inputs`` (relative paths resolve against
        the current working directory). It replaces discovery for selection and indexing;
        listed files that no longer exist are reported as ``missing-source`` and not
        processed. See :func:`neurodags.iterators.load_inputs`.
    enumeration : neurodags.iterators.InputEnumeration, optional
        An already computed enumeration (used by :func:`run_pipeline` so that all
        derivatives share one). Cannot be combined with ``inputs`` or
        ``max_files_per_dataset``.

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

    # Refuses an unmigrated config and invalid derivative names before anything runs.
    config_dict = load_pipeline_configuration(pipeline_configuration)

    definition_paths, definitions_dir, custom_node_paths = _resolve_definition_paths(
        config_dict, pipeline_configuration, base_dir
    )
    if definition_paths:
        load_node_definitions(definition_paths, base_dir=definitions_dir)

    if "DerivativeDefinitions" in config_dict:
        register_derivatives_from_dict(config_dict)
    else:
        log.warning(
            "No 'DerivativeDefinitions' found in the configuration. Skipping derivative registration."
        )

    datasets_configs, mount_point = get_datasets_and_mount_point_from_pipeline_configuration(
        pipeline_configuration, datasets_input=datasets_configuration, base_dir=base_dir
    )
    # Preflight: unresolved paths, missing/invalid source_root, invalid
    # derivatives_subdir and shared derivatives_path fail before any file is touched.
    resolve_dataset_paths(datasets_configs, mount_point)

    # One enumeration per invocation (spec §8): run_pipeline passes its own down.
    if enumeration is None:
        enumeration = enumerate_inputs(
            datasets_configs,
            mount_point,
            max_files_per_dataset=max_files_per_dataset,
            inputs=inputs,
        )
    elif inputs is not None or max_files_per_dataset is not None:
        raise ValueError(
            "enumeration= cannot be combined with inputs= or max_files_per_dataset= "
            "(they are already applied to the enumeration)."
        )
    # Output locations from source_root (spec 4.1); collisions fail before any write.
    reference_bases = _reference_bases(
        enumeration,
        datasets_configs,
        mount_point,
        list(config_dict.get("DerivativeDefinitions") or {}),
    )
    # Sorted, limited, indexed; only_index filters it (unknown indices raise).
    selected_entries = enumeration.select(only_index)

    dry_run_collection: list[dict[str, Any]] = []

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
    missing_results: list[_FileResult] = []
    dataset_payloads: dict[str, dict] = {}
    for input_entry in selected_entries:
        index, dataset_name, file_path = input_entry.index, input_entry.dataset, input_entry.path
        if input_entry.missing:
            missing_results.append(
                _missing_source_result(
                    input_entry, enumeration.inputs_file, derivative_label or "<unknown>"
                )
            )
            continue
        log.debug("Processing file", index=index, dataset=dataset_name, file_path=file_path)
        if dataset_name not in dataset_payloads:
            dataset_payloads[dataset_name] = _dataset_job_payload(
                datasets_configs[dataset_name], mount_point
            )
        dataset_payload = dataset_payloads[dataset_name]
        jobs.append(
            _FileJob(
                index=index,
                dataset=dataset_name,
                file_path=file_path,
                source_path=os.path.abspath(file_path),
                source_rel=input_entry.rel,
                dataset_config=dataset_payload,
                mount_point=mount_point,
                reference_base=reference_bases[index],
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
    if not jobs and not missing_results:
        log.info("No files to process")
        return pd.DataFrame(dry_run_collection) if dry_run else None

    parallel_results: list[_FileResult]
    if not jobs:
        parallel_results = []
    elif effective_n_jobs in (None, 1):
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
    if missing_results:
        parallel_results = sorted([*parallel_results, *missing_results], key=lambda r: r.index)

    for result in parallel_results:
        if dry_run and result.missing_source:
            # status shows these as "missing-source" at their original index.
            dry_run_collection.append(
                {
                    "index": result.index,
                    "dataset": result.dataset,
                    "file_path": result.file_path,
                    "derivative": result.derivative_label,
                    "source_status": "missing-source",
                    "plan": [],
                }
            )
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
        # The snapshot stays at <derivatives_path>/code/, not under derivatives_subdir.
        derivatives_path = resolve_dataset_field(dataset_config, "derivatives_path", mount_point)
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
    *,
    base_dir: str | os.PathLike | None = None,
    inputs: str | os.PathLike | None = None,
    enumeration: InputEnumeration | None = None,
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
        Maximum number of files to process per dataset: the first N in sorted order, applied
        before indexing. Cannot be combined with ``inputs``.
    dry_run : bool, optional
        When True, evaluate the pipeline without executing nodes and return a dataframe.
    only_index : int | list[int], optional
        Restrict execution to a subset of files by index. A negative or unknown index raises
        :class:`neurodags.iterators.IndexSelectionError` (a ``ValueError``) before anything
        is written.
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
    base_dir : str | os.PathLike, optional
        Directory that relative paths of in-memory configurations resolve against: the
        ``source_root``/``file_pattern``/``exclude_pattern``/``derivatives_path`` of a
        datasets mapping passed as a dict (inline in a dict ``pipeline_configuration``, or
        as ``datasets_configuration``), a relative ``datasets:`` reference and
        ``new_definitions`` of a dict pipeline. Datasets read from a YAML file resolve
        against that file's directory instead. Without it, relative dataset paths in a
        mapping are an error.
    inputs : str | os.PathLike, optional
        Frozen input list written by ``neurodags inputs`` (relative paths resolve against
        the current working directory); replaces discovery for selection and indexing. See
        :func:`iterate_derivative_pipeline`.
    enumeration : neurodags.iterators.InputEnumeration, optional
        An already computed enumeration to use instead of enumerating the inputs (the CLI
        ``status`` command passes its own to report files not in an input list). Cannot be
        combined with ``inputs`` or ``max_files_per_dataset``.

    Returns
    -------
    None | pandas.DataFrame
        Returns a combined dataframe with dry-run details when ``dry_run=True``; otherwise ``None``.
        Inputs are enumerated once and shared by every derivative.
    """
    # Refuses a configuration without naming_version: 1, or with invalid derivative
    # names (spec 2.2), before anything else.
    config = load_pipeline_configuration(pipeline_configuration)
    for name in derivatives or []:
        validate_derivative_name(name)

    # Preflight: resolve ~/$VARS in every active dataset's path fields before
    # anything is written (code snapshot, derivative dirs), so an unset variable
    # fails loudly here instead of creating a literal "$VAR" directory or being
    # swallowed by the per-file error records (#28).
    try:
        _ds_configs, _mount = get_datasets_and_mount_point_from_pipeline_configuration(
            pipeline_configuration, datasets_input=datasets_configuration, base_dir=base_dir
        )
    except Exception as _load_err:
        # Loading errors resurface (and raise) when each derivative loads the
        # datasets itself; only the snapshot is skipped here, as before.
        log.warning("Could not load datasets before execution", error=str(_load_err))
        _ds_configs = None
    if _ds_configs is not None:
        resolve_dataset_paths(_ds_configs, _mount)
        # One enumeration per invocation, shared by every derivative (spec §8). Done
        # before the snapshot so a bad input list or index fails before any write.
        if enumeration is None:
            enumeration = enumerate_inputs(
                _ds_configs,
                _mount,
                max_files_per_dataset=max_files_per_dataset,
                inputs=inputs,
            )
        elif inputs is not None or max_files_per_dataset is not None:
            raise ValueError(
                "enumeration= cannot be combined with inputs= or max_files_per_dataset= "
                "(they are already applied to the enumeration)."
            )
        enumeration.select(only_index)  # validate --only-index up front
        # Output collisions (spec 4.3) fail before the snapshot writes anything.
        _reference_bases(
            enumeration,
            _ds_configs,
            _mount,
            list(config.get("DerivativeDefinitions") or {}),
        )

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
            # Already applied to the shared enumeration; when the datasets could not be
            # loaded above, iterate re-raises the loading error itself.
            max_files_per_dataset=None if enumeration is not None else max_files_per_dataset,
            dry_run=dry_run,
            only_index=only_index,
            raise_on_error=raise_on_error,
            skip_errors=skip_errors,
            n_jobs=n_jobs,
            joblib_backend=joblib_backend,
            joblib_prefer=joblib_prefer,
            base_dir=base_dir,
            inputs=None if enumeration is not None else inputs,
            enumeration=enumeration,
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
    base_dir: str | os.PathLike | None = None,
    inputs: str | os.PathLike | None = None,
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
        Limit the number of files processed per dataset (the first N in sorted order).
        Cannot be combined with ``inputs``.
    only_index : int | list[int], optional
        Restrict collection to specific indices (matching iterate_derivative_pipeline behaviour;
        a negative or unknown index raises ``IndexSelectionError``, a ``ValueError``).
    output_format : {"wide", "long"}, optional
        Shape of the returned dataframe. ``"wide"`` retains one row per file with derivative columns,
        while ``"long"`` yields one row per collected derivative value without synthesising missing columns.
    preserve_complex_values : bool, optional
        When True, derivative artifacts keep their original (potentially nested) Python structures
        instead of being flattened or coerced into simple dataframe columns.
    raise_on_error : bool, optional
        Re-raise exceptions encountered while collecting derivatives; defaults to False.
        When False, a derivative that fails for a file is recorded in its ``name.error``
        column (wide) or as a ``derivative = "name.error"`` row (long). Column collisions
        are never recorded this way (see Raises).
    n_jobs : int, optional
        Number of parallel workers for file-level collection. ``-1`` uses all cores, ``1`` or
        ``None`` runs serially. Parallelism is per-file using separate processes (loky) to avoid
        HDF5/NetCDF4 thread-safety issues — both wide and long output formats are fully supported.
    base_dir : str | os.PathLike, optional
        Directory that relative paths of in-memory configurations resolve against: the
        ``source_root``/``file_pattern``/``exclude_pattern``/``derivatives_path`` of a
        datasets mapping passed as a dict (inline in a dict ``pipeline_configuration``, or
        as ``datasets_configuration``), a relative ``datasets:`` reference and
        ``new_definitions`` of a dict pipeline. Datasets read from a YAML file resolve
        against that file's directory instead. Without it, relative dataset paths in a
        mapping are an error.
    inputs : str | os.PathLike, optional
        Frozen input list written by ``neurodags inputs``; replaces discovery for selection
        and indexing. Listed files that no longer exist (``missing-source``) get no row.

    Returns
    -------
    pandas.DataFrame
        A dataframe with one row per processed file and columns derived from collected derivative artifacts.
        Generated columns follow :func:`neurodags.naming.format_column`
        (``name.suffix[@dim-value_...]``) and split back with
        :func:`neurodags.naming.parse_column`.

    Raises
    ------
    neurodags.naming.NamingError
        If a selected derivative name does not follow the name grammar.
    neurodags.naming.ColumnCollisionError
        If two coordinates, artifacts or derivatives would produce the same column. Raised
        whatever ``raise_on_error`` is: it is a structural problem, not a per-file failure.
    """

    log.debug(
        "build_derivative_dataframe: called",
        pipeline_configuration=pipeline_configuration,
        datasets_configuration=datasets_configuration,
    )

    config_dict = load_pipeline_configuration(pipeline_configuration)

    if "DerivativeDefinitions" not in config_dict:
        log.warning(
            "No 'DerivativeDefinitions' found in the configuration. Returning empty dataframe."
        )
        return pd.DataFrame()

    # Load custom node definitions so save=False derivatives using custom nodes work.
    # The absolute paths are also sent to the workers, which load them too (#36).
    definition_paths, definitions_dir, custom_node_paths = _resolve_definition_paths(
        config_dict, pipeline_configuration, base_dir
    )
    if definition_paths:
        load_node_definitions(definition_paths, base_dir=definitions_dir)

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
        pipeline_configuration, datasets_input=datasets_configuration, base_dir=base_dir
    )
    # Preflight (#28): fail on unresolved ~/$VARS before any directory is touched.
    resolve_dataset_paths(datasets_configs, mount_point)

    # One enumeration (spec §8); the layout roots come from the unlimited discovery.
    enumeration = enumerate_inputs(
        datasets_configs,
        mount_point,
        max_files_per_dataset=max_files_per_dataset,
        inputs=inputs,
    )
    reference_bases = _reference_bases(
        enumeration, datasets_configs, mount_point, list(derivative_definitions)
    )
    selected_entries = enumeration.select(only_index)
    for missing_entry in (e for e in selected_entries if e.missing):
        log.error(
            "Missing source file listed in the input list; no dataframe row",
            index=missing_entry.index,
            dataset=missing_entry.dataset,
            file_path=missing_entry.path,
            inputs_file=enumeration.inputs_file,
        )
    all_files = [(e.index, e.dataset, e.path, e.rel) for e in selected_entries if not e.missing]

    log.debug("build_derivative_dataframe: enumerated files", total_files=len(all_files))

    if output_format not in {"wide", "long"}:
        raise ValueError(f"Unsupported output_format '{output_format}'. Expected 'wide' or 'long'.")

    collect_long_rows = output_format == "long"

    n_files = len(all_files)
    effective_n_jobs = n_jobs if (n_jobs is not None and n_jobs != 1) else 1

    # Serialize DatasetConfig objects once — joblib workers reconstruct them.
    # derivatives_path is resolved here, in the parent (#28).
    dataset_payloads = {
        name: _dataset_job_payload(datasets_configs[name], mount_point)
        for name in {dataset_name for _, dataset_name, _, _ in all_files}
    }

    job_args = [
        (
            index,
            dataset_name,
            file_path,
            dataset_payloads[dataset_name],
            os.path.abspath(file_path),
            reference_bases[index],
            rel,
            mount_point,
            selected_derivatives,
            derivative_definitions,
            preserve_complex_values,
            collect_long_rows,
            raise_on_error,
            n_files,
            custom_node_paths,
        )
        for index, dataset_name, file_path, rel in all_files
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


def _build_reference_base(reference_base: str) -> Path:
    """The (already resolved, absolute) reference base of one dataframe row as a Path."""
    return Path(reference_base)


def _check_row_collisions(
    new_columns: Mapping[str, Any], column_owner: dict[str, str], derivative_name: str
) -> None:
    """Raise :class:`ColumnCollisionError` if ``new_columns`` are already in the row.

    Generated columns always contain ``.`` and start with their derivative's name, so this
    cannot happen with valid names; it guards the row merge so no value is ever silently
    overwritten (spec 7.1).
    """
    collisions = sorted(set(new_columns) & set(column_owner))
    if collisions:
        details = "; ".join(
            f"{column!r} from derivative {column_owner[column]!r} and derivative "
            f"{derivative_name!r}"
            for column in collisions
        )
        raise ColumnCollisionError(f"Dataframe column collision: {details}.")


def _collect_dataframe_file(
    index: int,
    dataset_name: str,
    file_path: str,
    dataset_config_dict: dict,
    source_path: str,
    reference_base: str,
    source_rel: str,
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
    and ``reference_base`` are absolute (resolved in the parent, #35) and used for
    all processing. Reading is read-only: when a derivative must be computed,
    run_derivative creates the output directory itself (#33).
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
            reference_base_path = _build_reference_base(reference_base)
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
                            reference_base=reference_base_path,
                            dataset_config=dataset_config,
                            mount_point=mount_point,
                            source_dataset=dataset_name,
                            source_rel=source_rel,
                            preserve_complex_values=preserve_complex_values,
                            flatten_xarray_artifacts=not preserve_complex_values,
                        )
                        if derivative_values:
                            _check_row_collisions(derivative_values, column_owner, derivative_name)
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
                    except ColumnCollisionError:
                        # Structural (spec 7.1): never recorded as a per-file error.
                        raise
                    except Exception as derivative_error:
                        log.error(
                            "Error collecting derivative for dataframe",
                            error=str(derivative_error),
                            exc_info=True,
                        )
                        if raise_on_error:
                            raise
                        error_col = error_column(derivative_name)
                        _check_row_collisions({error_col: None}, column_owner, derivative_name)
                        row[error_col] = str(derivative_error)
                        column_owner[error_col] = derivative_name
                        if collect_long_rows:
                            long_rows.append(
                                {
                                    "index": index,
                                    "dataset": dataset_name,
                                    "file_path": file_path,
                                    "derivative": error_col,
                                    "value": str(derivative_error),
                                }
                            )
            log.debug("Collected dataframe row", collected_derivatives=len(selected_derivatives))
            return row, long_rows
        except ColumnCollisionError:
            raise
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

    pipeline_configuration_dict = load_pipeline_configuration(args.config)

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
