import inspect
import os
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Literal

from joblib import Parallel, delayed

from cocofeats.datasets import get_datasets_and_mount_point_from_pipeline_configuration
from cocofeats.iterators import get_all_files_from_pipeline_configuration
from cocofeats.loggers import get_logger
from cocofeats.utils import get_path
from cocofeats.nodes import get_node, list_nodes
from cocofeats.nodes.loader import load_node_definitions
from cocofeats.features import get_feature, list_features
from cocofeats.dag import collect_feature_for_dataframe, run_feature
from cocofeats.loaders import load_configuration
from cocofeats.features.pipeline import register_features_from_dict
from cocofeats.definitions import DatasetConfig
import pandas as pd

log = get_logger(__name__)


@dataclass(slots=False) # slots=True seems to break parallelization
class _FileJob:
    index: int
    dataset: str
    file_path: str
    dataset_config: dict[str, Any]
    mount_point: Any
    common_root: str | None
    is_node: bool
    node_name: str | None
    feature_name: str | None
    feature_definition: dict[str, Any] | None
    feature_label: str
    dry_run: bool
    custom_node_paths: tuple[str, ...] = ()


@dataclass(slots=False) # break parallelization?
class _FileResult:
    index: int
    dataset: str
    file_path: str
    feature_label: str
    success: bool
    stage: str | None = None
    dry_run_payload: dict[str, Any] | None = None
    error_message: str | None = None
    error_type: str | None = None
    traceback: str | None = None


def _resolve_reference_base(
    file_path: str,
    dataset_config: DatasetConfig,
    common_root: str | None,
    mount_point: Any,
) -> tuple[str, Path]:
    derivatives_path = dataset_config.derivatives_path
    if derivatives_path:
        derivatives_path = get_path(derivatives_path, mount_point=mount_point)
        os.makedirs(derivatives_path, exist_ok=True)
        reference_base = (
            os.path.relpath(file_path, start=common_root) if common_root else file_path
        )
        reference_base = os.path.join(derivatives_path, reference_base)
        os.makedirs(os.path.dirname(reference_base), exist_ok=True)
    else:
        reference_base = file_path
    return reference_base, Path(reference_base)


def _process_file_job(job: _FileJob) -> _FileResult:
    if job.custom_node_paths:
        load_node_definitions(job.custom_node_paths)

    dataset_config = DatasetConfig(**job.dataset_config)
    try:
        reference_base_str, reference_base_path = _resolve_reference_base(
            job.file_path, dataset_config, job.common_root, job.mount_point
        )
    except Exception as setup_error:  # pragma: no cover - safety net
        return _FileResult(
            index=job.index,
            dataset=job.dataset,
            file_path=job.file_path,
            feature_label=job.feature_label,
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
            node_callable(job.file_path, **node_kwargs)
            dry_payload: dict[str, Any] | None = None
        else:
            if job.feature_definition is None or job.feature_name is None:
                raise ValueError("Feature job requires both definition and name.")
            dry_payload = None
            result = run_feature(
                job.feature_definition,
                job.feature_name,
                job.file_path,
                reference_base=reference_base_path,
                dataset_config=dataset_config,
                mount_point=job.mount_point,
                dry_run=job.dry_run,
            )
            if job.dry_run:
                dry_payload = result

        return _FileResult(
            index=job.index,
            dataset=job.dataset,
            file_path=job.file_path,
            feature_label=job.feature_label,
            success=True,
            dry_run_payload=dry_payload,
        )
    except Exception as run_error:  # pragma: no cover - safety net
        return _FileResult(
            index=job.index,
            dataset=job.dataset,
            file_path=job.file_path,
            feature_label=job.feature_label,
            success=False,
            stage="run",
            error_message=str(run_error),
            error_type=type(run_error).__name__,
            traceback=traceback.format_exc(),
        )


def iterate_feature_pipeline(
    pipeline_configuration: dict,
    feature: Callable | str,
    max_files_per_dataset: int | None = None,
    dry_run: bool = False,
    only_index: int | List[int] | None = None,
    raise_on_error: bool = False,
    n_jobs: int | None = None,
    joblib_backend: str | None = None,
    joblib_prefer: str | None = None,
) -> None:
    """
    Iterate over all files specified in the pipeline configuration and call a given function on each file.

    Parameters
    ----------
    pipeline_configuration : dict
        The pipeline configuration containing dataset information.
    feature : callable | str
        The feature or node to execute for each file. May be a registered feature/node name, a
        FeatureEntry, or a callable node.
    max_files_per_dataset : int, optional
        Maximum number of files to process per dataset. If None, processes all files.
    dry_run : bool, optional
        When True, evaluate the pipeline without executing nodes and return a dataframe describing
        the plan.
    only_index : int | list[int], optional
        Restrict execution to a subset of files by index.
    raise_on_error : bool, optional
        Re-raise the first failure instead of continuing.
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
    log.debug("iterate_call_pipeline: called", pipeline_configuration=pipeline_configuration)

    is_path_like = isinstance(pipeline_configuration, (str, os.PathLike))
    config_dict = (
        load_configuration(pipeline_configuration) if is_path_like else pipeline_configuration
    )

    custom_node_paths: tuple[str, ...] = tuple()
    new_definitions = config_dict.get("new_definitions")
    if new_definitions:
        if isinstance(new_definitions, (str, os.PathLike)):
            definition_paths = [new_definitions]
        elif isinstance(new_definitions, (list, tuple, set)):
            definition_paths = list(new_definitions)
        else:
            raise TypeError("new_definitions must be a string or list of paths")
        base_dir = Path(pipeline_configuration).resolve().parent if is_path_like else Path.cwd()
        resolved_definition_paths = []
        for candidate in definition_paths:
            candidate_path = Path(candidate)
            resolved_path = (
                (base_dir / candidate_path).resolve()
                if not candidate_path.is_absolute()
                else candidate_path.resolve()
            )
            resolved_definition_paths.append(str(resolved_path))
        load_node_definitions(definition_paths, base_dir=base_dir)
        custom_node_paths = tuple(resolved_definition_paths)

    if "FeatureDefinitions" in config_dict:
        register_features_from_dict(config_dict)
    else:
        log.warning(
            "No 'FeatureDefinitions' found in the configuration. Skipping feature registration."
        )

    datasets_configs, mount_point = get_datasets_and_mount_point_from_pipeline_configuration(
        pipeline_configuration
    )

    files_per_dataset, all_files, common_roots = get_all_files_from_pipeline_configuration(
        pipeline_configuration,
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
        dataset_file_count = {dataset: 0 for dataset in files_per_dataset.keys()}
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

    feature_entry = None
    node_callable: Callable | None = None
    feature_label: str | None = None

    if isinstance(feature, str):
        feature_label = feature
        if feature in list_nodes():
            node_callable = get_node(feature)
        elif feature in list_features():
            feature_entry = get_feature(feature)
        else:
            raise KeyError(f"Unknown feature or node '{feature}'")
    elif hasattr(feature, "definition") and hasattr(feature, "func"):
        feature_entry = feature
        feature_label = feature.name
    elif callable(feature):
        node_callable = feature
        feature_label = getattr(feature, "__name__", "<callable>")
    else:
        raise TypeError(
            "feature must be a registered feature name, node name, FeatureEntry, or callable node"
        )
    # skip save=False features
    if 'save' in feature_entry.definition:
        if feature_entry.definition['save'] is False:
            log.info(
                "Feature is marked with save=False; skipping execution.",
                feature=feature_label
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

    jobs: list[_FileJob] = []
    for index, dataset_name, file_path in all_files:
        log.debug("Processing file", index=index, dataset=dataset_name, file_path=file_path)
        dataset_config = datasets_configs[dataset_name]
        if hasattr(dataset_config, "model_dump"):
            dataset_payload = dataset_config.model_dump(mode="json")  # type: ignore[arg-type]
        else:  # pragma: no cover - pydantic v1 fallback
            dataset_payload = dataset_config.dict()
        jobs.append(
            _FileJob(
                index=index,
                dataset=dataset_name,
                file_path=file_path,
                dataset_config=dataset_payload,
                mount_point=mount_point,
                common_root=common_roots.get(dataset_name),
                is_node=node_callable is not None,
                node_name=feature_label if node_callable is not None else None,
                feature_name=feature_entry.name if feature_entry is not None else None,
                feature_definition=feature_entry.definition if feature_entry is not None else None,
                feature_label=feature_label or "<unknown>",
                dry_run=dry_run,
                custom_node_paths=custom_node_paths,
            )
        )

    log.info("iterate_call_pipeline: starting processing", total_files=len(jobs))
    if not jobs:
        log.info("iterate_call_pipeline: completed processing", total_files=0)
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
                feature=result.feature_label,
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
                "Error running feature"
                if result.stage == "run"
                else "Error processing file"
            )
            log.error(
                message,
                index=result.index,
                dataset=result.dataset,
                file_path=result.file_path,
                feature=result.feature_label,
                error=result.error_message,
                error_type=result.error_type,
                traceback=result.traceback,
            )
            if raise_on_error:
                error_summary = (
                    f"Failed to process file '{result.file_path}' for '{result.feature_label}' "
                    f"(dataset '{result.dataset}', index {result.index}). "
                    f"{result.error_type}: {result.error_message}"
                )
                if result.traceback:
                    error_summary = f"{error_summary}\n{result.traceback}"
                raise RuntimeError(error_summary)

    log.info("iterate_call_pipeline: completed processing", total_files=len(jobs))
    if dry_run:
        return pd.DataFrame(dry_run_collection)


def build_feature_dataframe(
    pipeline_configuration: dict | str,
    *,
    include_features: List[str] | None = None,
    max_files_per_dataset: int | None = None,
    only_index: int | List[int] | None = None,
    output_format: Literal["wide", "long"] = "wide",
    preserve_complex_values: bool = False,
    raise_on_error: bool = False,
) -> pd.DataFrame:
    """
    Assemble a dataframe by collecting feature artifacts for every file in a pipeline configuration.

    Parameters
    ----------
    pipeline_configuration : dict | str
        Pipeline configuration or path to it.
    include_features : list[str], optional
        Restrict collection to this explicit subset of feature names.
    max_files_per_dataset : int, optional
        Limit the number of files processed per dataset.
    only_index : int | list[int], optional
        Restrict collection to specific indices (matching iterate_feature_pipeline behaviour).
    output_format : {"wide", "long"}, optional
        Shape of the returned dataframe. ``"wide"`` retains one row per file with feature columns,
        while ``"long"`` yields one row per collected feature value without synthesising missing columns.
    preserve_complex_values : bool, optional
        When True, feature artifacts keep their original (potentially nested) Python structures
        instead of being flattened or coerced into simple dataframe columns.
    raise_on_error : bool, optional
        Re-raise exceptions encountered while collecting features; defaults to False.

    Returns
    -------
    pandas.DataFrame
        A dataframe with one row per processed file and columns derived from collected feature artifacts.
    """

    log.debug("build_feature_dataframe: called", pipeline_configuration=pipeline_configuration)

    config_dict = load_configuration(pipeline_configuration) if isinstance(pipeline_configuration, str) else pipeline_configuration

    if "FeatureDefinitions" not in config_dict:
        log.warning("No 'FeatureDefinitions' found in the configuration. Returning empty dataframe.")
        return pd.DataFrame()

    register_features_from_dict(config_dict)

    feature_definitions: dict[str, dict] = config_dict.get("FeatureDefinitions", {}) or {}
    selected_features: list[str] = []
    
    include_features = config_dict.get("FeatureList", []) if include_features is None else include_features

    for feature_name, feature_def in feature_definitions.items():
        if include_features is not None and feature_name not in include_features:
            continue
        if not feature_def.get("for_dataframe", True):
            continue
        selected_features.append(feature_name)
    if include_features:
        missing_features = sorted(set(include_features) - set(selected_features))
        if missing_features:
            log.warning("Some requested features are either undefined or flagged out of dataframe collection.", missing_features=missing_features)

    if not selected_features:
        log.warning("No features eligible for dataframe collection were found. Returning empty dataframe.")
        return pd.DataFrame(columns=["index", "dataset", "file_path"])

    datasets_configs, mount_point = get_datasets_and_mount_point_from_pipeline_configuration(
        pipeline_configuration
    )

    _, _, common_roots = get_all_files_from_pipeline_configuration(
        pipeline_configuration, max_files_per_dataset=None
    )

    files_per_dataset, all_files, _ = get_all_files_from_pipeline_configuration(
        pipeline_configuration, max_files_per_dataset=max_files_per_dataset
    )


    log.debug("build_feature_dataframe: enumerated files", total_files=len(all_files), per_dataset=files_per_dataset)

    if only_index is not None:
        index_filter = only_index if isinstance(only_index, list) else [only_index]
        filtered = [item for item in all_files if item[0] in index_filter]
        if len(filtered) != len(index_filter):
            detected = [item[0] for item in filtered]
            missing = list(set(index_filter) - set(detected))
            log.warning("Some indices requested for dataframe collection were not found.", requested=index_filter, missing=missing)
        all_files = filtered

    if output_format not in {"wide", "long"}:
        raise ValueError(f"Unsupported output_format '{output_format}'. Expected 'wide' or 'long'.")

    collect_long_rows = output_format == "long"

    rows: list[dict[str, Any]] = []
    long_rows: list[dict[str, Any]] = []
    for index, dataset_name, file_path in all_files:
        try:
            dataset_config = datasets_configs[dataset_name]
            common_root = common_roots.get(dataset_name)
            reference_base = _build_reference_base(file_path, dataset_config, common_root, mount_point)
            row: dict[str, Any] = {
                "index": index,
                "dataset": dataset_name,
                "file_path": file_path,
            }
            for feature_name in selected_features:
                feature_def = feature_definitions.get(feature_name, {}) or {}
                try:
                    feature_values = collect_feature_for_dataframe(
                        feature_def,
                        feature_name,
                        file_path,
                        reference_base=reference_base,
                        dataset_config=dataset_config,
                        mount_point=mount_point,
                        preserve_complex_values=preserve_complex_values,
                        flatten_xarray_artifacts=not preserve_complex_values,
                    )
                    if feature_values:
                        row.update(feature_values)
                        if collect_long_rows:
                            for column_name, value in feature_values.items():
                                long_rows.append(
                                    {
                                        "index": index,
                                        "dataset": dataset_name,
                                        "file_path": file_path,
                                        "feature": column_name,
                                        "value": value,
                                    }
                                )
                except Exception as feature_error:
                    log.error(
                        "Error collecting feature for dataframe",
                        feature=feature_name,
                        index=index,
                        dataset=dataset_name,
                        file_path=file_path,
                        error=str(feature_error),
                        exc_info=True,
                    )
                    if raise_on_error:
                        raise
                    row[f"{feature_name}__error"] = str(feature_error)
                    if collect_long_rows:
                        long_rows.append(
                            {
                                "index": index,
                                "dataset": dataset_name,
                                "file_path": file_path,
                                "feature": f"{feature_name}__error",
                                "value": str(feature_error),
                            }
                        )
            rows.append(row)
            log.debug(
                "Collected dataframe row",
                index=index,
                dataset=dataset_name,
                file_path=file_path,
                collected_features=len(selected_features),
            )
        except Exception as error:
            log.error(
                "Error collecting dataframe row",
                index=index,
                dataset=dataset_name,
                file_path=file_path,
                error=str(error),
                exc_info=True,
            )
            if raise_on_error:
                raise

    if output_format == "long":
        if not long_rows:
            return pd.DataFrame(columns=["index", "dataset", "file_path", "feature", "value"])
        return pd.DataFrame(long_rows)

    return pd.DataFrame(rows)


def _build_reference_base(
    file_path: str, dataset_config, common_root: str | None, mount_point: Path | None
) -> Path:
    _, reference_base_path = _resolve_reference_base(file_path, dataset_config, common_root, mount_point)
    return reference_base_path

if __name__ == "__main__":
    # Parse args and run iterate_feature_pipeline
    import argparse
    parser = argparse.ArgumentParser(description="Iterate feature pipeline.")
    parser.add_argument("config", type=str, help="Path to the pipeline configuration file (YAML or JSON).")
    parser.add_argument("--max_files_per_dataset", type=int, default=None, help="Maximum number of files to process per dataset.")
    parser.add_argument("--dry_run", action="store_true", help="If set, perform a dry run without actual processing.")
    parser.add_argument("--only_index", type=int, nargs='*', default=None, help="Only process files with these indices.")
    parser.add_argument("--raise_on_error", action="store_true", help="If set, raise exceptions on errors instead of logging them.")
    parser.add_argument("--n-jobs", dest="n_jobs", type=int, default=None, help="Number of parallel workers to use (joblib semantics; default serial).")
    parser.add_argument("--joblib-backend", dest="joblib_backend", type=str, default=None, help="Joblib backend to use (e.g. 'loky', 'threading').")
    parser.add_argument("--joblib-prefer", dest="joblib_prefer", type=str, default=None, help="Joblib execution preference hint (e.g. 'processes', 'threads').")
    parser.add_argument("--make_final_dataframe", action="store_true", help="If set, build the final feature dataframe after processing.")
    parser.add_argument(
        "--dataframe_output",
        type=str,
        default=None,
        help="Optional path where the dataframe should be written (CSV by default, or Parquet if the extension is .parquet).",
    )
    parser.add_argument(
        "--dataframe_features",
        nargs="*",
        default=None,
        help="Optional subset of feature names to include when building the dataframe.",
    )
    parser.add_argument(
        "--dataframe_format",
        choices=("wide", "long"),
        default="wide",
        help="Layout of the generated dataframe; 'long' emits one row per collected feature value.",
    )
    parser.add_argument(
        "--preserve_complex_values",
        action="store_true",
        help="Retain complex feature artifacts without flattening or converting them for dataframe storage.",
    )
    args = parser.parse_args()

    from cocofeats.loaders import load_configuration
    pipeline_configuration = load_configuration(args.config)

    feature_list = pipeline_configuration.get("FeatureList", [])
    if not feature_list:
        log.error("No features specified in the pipeline configuration under 'FeatureList'. Exiting.")
        exit(1)

    if args.dry_run:
        log.info("Performing dry run of feature pipeline", features=feature_list)
        dry_run_results = []

    if not args.make_final_dataframe:  # and args.dataframe_output:
        for feature in feature_list:
            output = iterate_feature_pipeline(
                pipeline_configuration=pipeline_configuration,
                feature=feature,
                max_files_per_dataset=args.max_files_per_dataset,
                dry_run=args.dry_run,
                only_index=args.only_index,
                raise_on_error=True if args.raise_on_error else False,
                n_jobs=args.n_jobs,
                joblib_backend=args.joblib_backend,
                joblib_prefer=args.joblib_prefer,
            )
            if args.dry_run and output is not None:
                dry_run_results.append((feature, output))
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
                        log.info("Saved dry run results", path=str(csv_fallback), rows=len(df), columns=list(df.columns))
                    else:
                        log.info("Saved dry run results", path=str(output_path), rows=len(df), columns=list(df.columns))
                else:
                    df.to_csv(output_path, index=False)
                    log.info("Saved dry run results", path=str(output_path), rows=len(df), columns=list(df.columns))
            else:
                df.to_csv("dry_run_results.csv", index=False)
                log.info("Saved dry run results to dry_run_results.csv", rows=len(df), columns=list(df.columns))
            log.info("Dry run completed", total_rows=len(df))
            


    if args.make_final_dataframe:
        log.info("Building feature dataframe", features=args.dataframe_features)
        dataframe = build_feature_dataframe(
            pipeline_configuration=pipeline_configuration,
            include_features=args.dataframe_features,
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
                    log.info("Saved feature dataframe", path=str(csv_fallback), rows=len(dataframe), columns=list(dataframe.columns))
                else:
                    log.info("Saved feature dataframe", path=str(output_path), rows=len(dataframe), columns=list(dataframe.columns))
            else:
                dataframe.to_csv(output_path, index=False)
                log.info("Saved feature dataframe", path=str(output_path), rows=len(dataframe), columns=list(dataframe.columns))
        else:
            log.info("Feature dataframe built (not saved to disk)", rows=len(dataframe), columns=list(dataframe.columns))
            print(dataframe)
