import os

from neurodags.datasets import generate_dummy_dataset
from neurodags.nodes import get_node, list_nodes
from neurodags.orchestrators import (
    get_all_files_from_pipeline_configuration,
    iterate_derivative_pipeline,
)

dataset_1 = {
    "PATTERN": "sub-%subject%/ses-%session%/sub-%subject%_ses-%session%_task-%task%_acq-%acquisition%_run-%run%",
    "DATASET": "dataset1",
    "NSUBS": 2,
    "NSESSIONS": 2,
    "NTASKS": 2,
    "NACQS": 2,
    "NRUNS": 2,
    "PREFIXES": {"subject": "S", "session": "SE", "task": "T", "acquisition": "A", "run": "R"},
    "ROOT": "_data/dataset1",
}

dataset_2 = {
    "PATTERN": "sub-%subject%/ses-%session%/sub-%subject%_ses-%session%_task-%task%_acq-%acquisition%_run-%run%",
    "DATASET": "dataset2",
    "NSUBS": 3,
    "NSESSIONS": 1,
    "NTASKS": 2,
    "NACQS": 1,
    "NRUNS": 2,
    "PREFIXES": {"subject": "S", "session": "SE", "task": "T", "acquisition": "A", "run": "R"},
    "ROOT": "_data/dataset2",
}

generate_dummy_dataset(data_params=dataset_1)
generate_dummy_dataset(data_params=dataset_2)

datasets = {
    "dataset_1": {
        "name": "Dataset 1",
        "source_root": "_data/dataset1",
        "file_pattern": "_data/dataset1/**/*.vhdr",
        "derivatives_path": "_outputs/dataset1",
        "extra_field": "extra_value",  # Example of an extra field
    },
    "dataset_2": {
        "name": "Dataset 2",
        "source_root": "_data/dataset2",
        "file_pattern": "_data/dataset2/**/*.vhdr",
        "derivatives_path": "_outputs/dataset2",
        "extra_field": "extra_value",  # Example of an extra field
    },
}

# Relative dataset paths in an in-memory mapping resolve against base_dir (0.4.0).
BASE_DIR = os.getcwd()

pipeline_input = {
    "naming_version": 1,
    "datasets": datasets,
    "mount_point": None,
}
files_per_dataset, all_files, common_roots = get_all_files_from_pipeline_configuration(
    pipeline_input, max_files_per_dataset=None, base_dir=BASE_DIR
)

print("Files per dataset:", files_per_dataset)
print("All files with indices:", all_files)
print("Common roots:", common_roots)

# Now run the derivative extraction
print("Registered nodes:", list_nodes())

if False:
    for derivative in ["spectrum", "basic_preprocessing"]:
        iterate_derivative_pipeline(
            pipeline_configuration=pipeline_input,
            derivative=get_node(derivative),
            max_files_per_dataset=2,
            base_dir=BASE_DIR,
        )

# 1) Load and register derivatives
# 2) Use your existing iterate_derivative_pipeline per derivative name

# "BasicPrep1", "CheckLineFrequency",
# "BasicPrep1", "CheckLineFrequency", "InterDerivativeDependence",
for derivative_name in ["SpectrumArrayWelch", "SpectrumArrayMultitaper"]:
    df = iterate_derivative_pipeline(
        pipeline_configuration=pipeline_input,
        derivative=derivative_name,  # the thin wrapper
        max_files_per_dataset=None,
        dry_run=False,
        only_index=[3, 5],
        raise_on_error=True,
        base_dir=BASE_DIR,
    )
