# Terminal User Interface (TUI)

NeuroDAGs includes a built-in Terminal User Interface (TUI) powered by [Textual](https://textual.textualize.io/). It provides a graphical environment within your terminal to manage configurations, visualize pipelines, perform dry runs, execute processing, and explore results.

## Installation

The TUI requires the `[tui]` extra:

```bash
pip install neurodags[tui]
```

## Launching the TUI

Launch the TUI with the unified CLI. You can pass the path to a pipeline
configuration file and optionally a datasets configuration file:

```bash
neurodags tui [pipeline.yml] [-d datasets.yml]
```

Legacy alias:

```bash
neurodags-tui [pipeline.yml] [-d datasets.yml]
```

Module entry point:

```bash
python -m neurodags.tui [pipeline.yml] [-d datasets.yml]
```

The `-d` path is relative to the directory you launch from (the 0.3.5 fallback to the pipeline
YAML's directory was removed in 0.4.0); see {doc}`usage` for the exact resolution rule. The
pipeline must declare `naming_version: 1`: an unmigrated 0.3.x configuration is refused before
the interface starts (see {doc}`migrating_to_0_4`).

## Interface Overview

The TUI is organized into several tabs, each focused on a specific part of the NeuroDAGs workflow.

### 1. Config Tab
This is where you load your pipeline configuration.
- **Pipeline YAML path**: Enter the path to your `pipeline.yml`.
- **Datasets YAML path (optional override)**: Enter the path to an independent `datasets.yml`. If provided, this overrides any `datasets` definition found inside the pipeline configuration. A relative path is resolved against the directory the TUI was started from (the same rule as `-d` on the command line, not the pipeline YAML's directory); the summary lists the datasets of that file, and every tab uses that same file.
- **Load**: Click to load the configuration. Once loaded, a summary of datasets and derivatives will be displayed.


### 2. DAG Tab
Visualize the structure of your pipeline.
- **Refresh**: Generate a [Mermaid](https://mermaid.js.org/) diagram of the loaded pipeline.
- **Open HTML in browser**: Render the DAG to an interactive HTML file and open it in your default web browser.

### 3. Dry Run Tab
Inspect what actions the pipeline would take without actually executing them.
- **Select derivative**: Choose which derivative you want to dry-run.
- **max files/dataset**: Limit the number of files processed per dataset (leave blank for all).
- **Run Dry Run**: Executes the dry run and displays the plan in a table, showing which files are already cached and which would be recomputed.

### 4. Run Pipeline Tab
Execute the pipeline to compute derivatives.
- **Select derivative**: Choose which derivative to compute.
- **max files/dataset**: Limit the number of files.
- **n_jobs**: Set the number of parallel jobs (leave blank for serial execution).
- **Run**: Starts the pipeline execution. Captured pipeline output is shown in the log area when the run completes.

### 5. DataFrame Tab
Assemble computation results into ML-ready DataFrames.
- **derivatives**: Derivatives to include, separated by spaces (for example `Alpha Beta`; leave blank for all). Commas are still accepted as separators (`Alpha,Beta`), the pre-0.4.0 syntax.
- **format**: Choose between `wide` or `long` format.
- **max files/dataset**: Limit the number of files.
- **Assemble**: Build the DataFrame and display a preview in the table.

### 6. Status Tab
Check per-derivative processing status for the loaded pipeline (the TUI equivalent of the `neurodags status` command).
- **Select derivative**: Choose a derivative to inspect, or leave on `All (DerivativeList)`.
- **Refresh**: Scan outputs and populate the table with *done* / *incomplete* / *missing* / *errored* counts per derivative, plus a summary line and any error paths. The states are those of {ref}`Unit states <naming-unit-states>`; use `neurodags status` for the `skipped` and `(migrated)` counts. A unit whose plan could not be computed (a configuration error) is counted as *errored* and listed with its planning error.

### 7. NC Viewer Tab
Launch the interactive Dash explorer for results.
- **.fif or .nc file path**: Path to a NetCDF or MNE FIF file.
- **Launch Dash Explorer**: Starts the Dash-Plotly explorer in the background. You can then open your browser at `http://127.0.0.1:8050` to interactively explore the data.

## Keyboard Shortcuts

- **q**: Quit the TUI.
- **Tab**: Navigate between fields and buttons.
- **Enter**: Press buttons or select items.
