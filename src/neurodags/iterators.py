import glob

from neurodags.datasets import get_datasets_and_mount_point_from_pipeline_configuration
from neurodags.definitions import DatasetConfig
from neurodags.loggers import get_logger
from neurodags.utils import find_unique_root, get_path

log = get_logger(__name__)


def get_files_from_pattern(pattern, recursive: bool = True, exclude_filter=None) -> list[str]:
    """
    Get a list of file paths matching the given glob pattern.

    Parameters
    ----------
    pattern : str
        The glob pattern to match files.
    recursive : bool, optional
        Whether to search recursively in subdirectories. Default is True.
    exclude_filter : str, optional
        A glob pattern to exclude certain files from the results.

    Returns
    -------
    list of str
        A list of file paths matching the pattern.
    """

    log.debug("get_files_from_pattern: called", pattern=pattern, recursive=recursive)
    files = glob.glob(pattern, recursive=recursive)
    log.debug("get_files_from_pattern: found files", count=len(files))

    # Apply exclude filter if provided
    if exclude_filter:
        log.debug("get_files_from_pattern: applying exclude filter", exclude_filter=exclude_filter)
        excluded_files = set(glob.glob(exclude_filter, recursive=recursive))
        files = [f for f in files if f not in excluded_files]
        log.debug("get_files_from_pattern: files after exclusion", count=len(files))

    return files


def get_all_files_across_datasets(
    datasets: dict[str, DatasetConfig],
    mount_point: str | None = None,
    max_files_per_dataset: int | None = None,
) -> dict[str, list[str]]:
    """
    Iterate over all datasets and retrieve files based on their patterns.

    Parameters
    ----------
    datasets : dict of str to DatasetConfig
        A dictionary mapping dataset names to their configurations.
    mount_point : str, optional
        The mount point to resolve paths if needed.
    max_files_per_dataset : int, optional
        Maximum number of files to retrieve per dataset. If None, retrieves all files.

    Returns
    -------
    dict of str to list of str
        A dictionary mapping dataset names to lists of file paths.
    list of tuple
        A list of tuples (index, dataset_name, file_path) for all files across datasets
    """

    files_per_dataset = {}
    common_roots = {}
    for dataset_name, dataset_config in datasets.items():
        log.info("get_all_files_across_datasets: processing dataset", dataset=dataset_name)

        if dataset_config.skip:
            log.info(
                "get_all_files_across_datasets: skipping dataset as per configuration",
                dataset=dataset_name,
            )
            continue

        pattern = dataset_config.file_pattern
        if not pattern:
            log.warning(
                "get_all_files_across_datasets: no pattern defined for dataset",
                dataset=dataset_name,
            )
            continue

        resolved_pattern = get_path(pattern, mount_point=mount_point)
        log.debug(
            "get_all_files_across_datasets: resolved pattern",
            dataset=dataset_name,
            pattern=resolved_pattern,
        )

        try:
            files = get_files_from_pattern(
                resolved_pattern, exclude_filter=dataset_config.exclude_pattern
            )
        except Exception as e:
            log.error(
                "get_all_files_across_datasets: error getting files from pattern",
                dataset=dataset_name,
                error=str(e),
            )
            continue

        if max_files_per_dataset is not None:
            files = files[:max_files_per_dataset]
            log.debug(
                "get_all_files_across_datasets: limited files per dataset",
                dataset=dataset_name,
                max_files=max_files_per_dataset,
            )

        if not files:
            log.warning(
                "get_all_files_across_datasets: no files found for dataset", dataset=dataset_name
            )
            continue

        files_per_dataset[dataset_name] = files
        common_root = find_unique_root(files, mode="maximal")
        common_roots[dataset_name] = common_root
        log.info(
            "get_all_files_across_datasets: common root for dataset",
            dataset=dataset_name,
            common_root=common_root,
        )
        log.info(
            "get_all_files_across_datasets: added files for dataset",
            dataset=dataset_name,
            file_count=len(files),
        )
    all_files = []
    for dataset, files in files_per_dataset.items():
        log.info(
            "get_all_files_across_datasets: dataset summary", dataset=dataset, file_count=len(files)
        )
        these_files = [(dataset, f) for f in files]
        all_files.extend(these_files)
    # add index to each file tuple
    all_files = [(i, dataset, f) for i, (dataset, f) in enumerate(all_files)]
    log.info(
        "get_all_files_across_datasets: completed",
        total_datasets=len(files_per_dataset),
        total_files=len(all_files),
    )
    return files_per_dataset, all_files, common_roots


def get_all_files_from_pipeline_configuration(
    pipeline_input, datasets_input=None, max_files_per_dataset=None
):
    """
    Given a pipeline configuration and an optional datasets configuration,
    retrieve all files across datasets.

    Parameters
    ----------
    pipeline_input : dict | path-like
        A dictionary or path to a YAML file containing pipeline configuration.
    datasets_input : dict | path-like, optional
        A dictionary or path to a YAML file containing datasets configuration.
        If provided, it overrides the 'datasets' section of the pipeline configuration.
    max_files_per_dataset : int, optional
        Maximum number of files to retrieve per dataset. If None, retrieves all files.

    Returns
    -------
    dict of str to list of str
        A dictionary mapping dataset names to lists of file paths.
    list of tuple
        A list of tuples (index, dataset_name, file_path) for all files across datasets
    dict of str to str
        A dictionary mapping dataset names to their common roots.
    """
    log.debug("get_all_files_from_pipeline_configuration: called", pipeline_input=pipeline_input)
    datasets, mount_point = get_datasets_and_mount_point_from_pipeline_configuration(
        pipeline_input, datasets_input=datasets_input
    )

    files_per_dataset, all_files, common_roots = get_all_files_across_datasets(
        datasets, mount_point=mount_point, max_files_per_dataset=max_files_per_dataset
    )
    log.debug(
        "get_all_files_from_pipeline_configuration: completed",
        total_datasets=len(files_per_dataset),
        total_files=len(all_files),
    )

    return files_per_dataset, all_files, common_roots
