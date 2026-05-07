"""
Quickstart: Synthetic EEG Pipeline
====================================

This example walks through a complete NeuroDAGs pipeline using synthetically
generated EEG data — no real dataset required.

We will:

1. Generate a synthetic multi-subject BrainVision dataset.
2. Define a pipeline in Python (preprocessing → spectral → band power).
3. Inspect the plan with a dry run.
4. Execute the pipeline.
5. Assemble results into a dataframe.
6. Plot band power across subjects.
"""

# %%
# Setup
# -----
# Standard imports and a temporary working directory.

import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from neurodags.datasets import generate_dummy_dataset
from neurodags.orchestrators import build_derivative_dataframe, iterate_derivative_pipeline

WORKDIR = Path(tempfile.mkdtemp(prefix="neurodags_quickstart_"))
DATA_DIR = WORKDIR / "rawdata"
OUT_DIR = WORKDIR / "derivatives"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Working directory: {WORKDIR}")

# %%
# Step 1 — Generate Synthetic Dataset
# -------------------------------------
# :func:`~neurodags.datasets.generate_dummy_dataset` creates BrainVision trios
# (``.vhdr`` / ``.vmrk`` / ``.eeg``) using 1/f^α (pink) noise to mimic realistic
# EEG spectral characteristics.
#
# We generate **3 subjects × 1 session** at 200 Hz, 30 seconds each.

generate_dummy_dataset(
    data_params={
        "DATASET": "quickstart",
        "PATTERN": "sub-%subject%/ses-%session%/sub-%subject%_ses-%session%_task-rest",
        "NSUBS": 3,
        "NSESSIONS": 1,
        "NTASKS": 1,
        "NACQS": 1,
        "NRUNS": 1,
        "PREFIXES": {
            "subject": "S",
            "session": "SE",
            "task": "T",
            "acquisition": "A",
            "run": "R",
        },
        "ROOT": str(DATA_DIR),
    },
    generation_args={
        "NCHANNELS": 8,
        "SFREQ": 200.0,
        "STOP": 30.0,
        "NUMEVENTS": 10,
        "random_state": 0,
    },
)

source_files = sorted(DATA_DIR.rglob("*.vhdr"))
print(f"Generated {len(source_files)} source file(s):")
for f in source_files:
    print(f"  {f.relative_to(WORKDIR)}")

# %%
# Step 2 — Define the Pipeline
# ------------------------------
# Pipelines are pure Python dicts — no YAML file needed (though YAML works too).
#
# This pipeline has three derivatives:
#
# - **BasicPrep**: band-pass filter → 2-second epochs.
# - **Spectrum**: Welch PSD on each epoch.
# - **BandPower**: relative power in δ, θ, α, β bands, averaged across epochs
#   (``save=False`` — computed but not written to disk; ``for_dataframe=True``
#   — included in the aggregated dataframe).

datasets = {
    "quickstart": {
        "name": "Quickstart",
        "file_pattern": str(DATA_DIR / "**" / "*.vhdr"),
        "derivatives_path": str(OUT_DIR),
    }
}

pipeline_config = {
    "datasets": datasets,
    "mount_point": None,
    "DerivativeDefinitions": {
        "BasicPrep": {
            "overwrite": False,
            "nodes": [
                {"id": 0, "derivative": "SourceFile"},
                {
                    "id": 1,
                    "node": "basic_preprocessing",
                    "args": {
                        "mne_object": "id.0",
                        "filter_args": {"l_freq": 1.0, "h_freq": 80.0},
                        "epoch_config": {"duration": 2.0, "overlap": 0.0},
                    },
                },
            ],
        },
        "Spectrum": {
            "overwrite": False,
            "nodes": [
                {"id": 0, "derivative": "BasicPrep.fif"},
                {
                    "id": 1,
                    "node": "mne_spectrum_array",
                    "args": {
                        "meeg": "id.0",
                        "method": "welch",
                        "method_kwargs": {"n_per_seg": 200},
                    },
                },
            ],
        },
        "BandPower": {
            "save": False,
            "for_dataframe": True,
            "nodes": [
                {"id": 0, "derivative": "Spectrum.nc"},
                {
                    "id": 1,
                    "node": "extract_data_var",
                    "args": {"dataset_like": "id.0", "data_var": "spectrum"},
                },
                {
                    "id": 2,
                    "node": "bandpower",
                    "args": {
                        "psd_like": "id.1",
                        "relative": True,
                        "bands": {
                            "delta": [1.0, 4.0],
                            "theta": [4.0, 8.0],
                            "alpha": [8.0, 13.0],
                            "beta":  [13.0, 30.0],
                        },
                    },
                },
                {
                    "id": 3,
                    "node": "aggregate_across_dimension",
                    "args": {
                        "xarray_data": "id.2",
                        "dim": "epochs",
                        "operation": "mean",
                    },
                },
            ],
        },
    },
    "DerivativeList": ["BasicPrep", "Spectrum", "BandPower"],
}

print("Pipeline defined with derivatives:", pipeline_config["DerivativeList"])

# %%
# Step 3 — Dry Run
# -----------------
# Inspect the execution plan for ``BasicPrep`` without running any computation.
# The returned dataframe shows which outputs are cached and which would be computed.

plan = iterate_derivative_pipeline(pipeline_config, "BasicPrep", dry_run=True)
# 'plan' column contains per-step dicts — expand for display
steps = []
for _, row in plan.iterrows():
    for step in row["plan"]:
        steps.append({"file": row["file_path"].split("/")[-1], **step})
print(pd.DataFrame(steps)[["file", "id", "kind", "cached"]].to_string(index=False))

# %%
# Step 4 — Execute the Pipeline
# ------------------------------
# Run each derivative in order. Already-cached outputs are skipped automatically.

for derivative in pipeline_config["DerivativeList"]:
    print(f"\n--- Running: {derivative} ---")
    iterate_derivative_pipeline(pipeline_config, derivative, raise_on_error=True)

# List produced files
produced = sorted(OUT_DIR.rglob("*@*.fif")) + sorted(OUT_DIR.rglob("*@*.nc"))
print(f"\nProduced {len(produced)} derivative file(s):")
for f in produced:
    print(f"  {f.relative_to(WORKDIR)}")

# %%
# Step 5 — Assemble Dataframe
# ----------------------------
# :func:`~neurodags.orchestrators.build_derivative_dataframe` collects every
# ``for_dataframe=True`` derivative into a single dataframe.
#
# ``output_format="wide"`` gives one row per file with derivative columns.

df = build_derivative_dataframe(pipeline_config, output_format="wide")

# Extract readable subject labels from the file path
df["subject"] = df["file_path"].apply(
    lambda p: next(
        (part for part in Path(p).parts if part.startswith("sub-")),
        Path(p).stem.split("_")[0],
    )
)

print(f"DataFrame shape: {df.shape}")
print(df.head())

# %%
# Step 6 — Visualise Band Power
# ------------------------------
# Group by subject and plot mean relative band power per frequency band.

band_cols = [c for c in df.columns if any(b in c for b in ["delta", "theta", "alpha", "beta"])]

if band_cols:
    # Melt to long form for plotting
    df_long = df[["subject"] + band_cols].melt(
        id_vars="subject", var_name="band_channel", value_name="relative_power"
    )
    # Extract band name from column label
    df_long["band"] = df_long["band_channel"].apply(
        lambda x: next((b for b in ["delta", "theta", "alpha", "beta"] if b in x), None)
    )
    band_means = df_long.groupby(["subject", "band"])["relative_power"].mean().reset_index()

    bands = ["delta", "theta", "alpha", "beta"]
    band_means = band_means[band_means["band"].isin(bands)]

    subjects = sorted(band_means["subject"].unique())
    x = np.arange(len(bands))
    width = 0.8 / len(subjects)

    fig, ax = plt.subplots(figsize=(8, 4))
    for i, sub in enumerate(subjects):
        vals = [
            band_means.loc[
                (band_means["subject"] == sub) & (band_means["band"] == b), "relative_power"
            ].mean()
            for b in bands
        ]
        ax.bar(x + i * width, vals, width=width, label=sub)

    ax.set_xticks(x + width * (len(subjects) - 1) / 2)
    ax.set_xticklabels(bands)
    ax.set_ylabel("Relative Power")
    ax.set_title("Mean Relative Band Power per Subject")
    ax.legend(title="Subject")
    plt.tight_layout()
    plt.savefig(WORKDIR / "band_power.png", dpi=100)
    plt.show()
    print(f"Plot saved to {WORKDIR / 'band_power.png'}")
else:
    print("No band power columns found in dataframe.")

# %%
# What's Next
# ------------
# - Swap ``generate_dummy_dataset`` for real BIDS data by pointing ``file_pattern``
#   at your raw EEG files.
# - Move the config dict to a ``pipeline.yml`` and ``datasets.yml`` for
#   version-controlled, reproducible workflows.
# - Add custom nodes via ``new_definitions: my_nodes.py``.
# - Scale up: set ``n_jobs=-1`` for file-level parallelism via joblib.
# - Inspect any ``.nc`` file interactively with the built-in Dash explorer::
#
#       python -m neurodags.visualization path/to/file.nc
