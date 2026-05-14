# Usage

- `neurodags validate pipeline.yml` — load config and print dataset / derivative summary
- `neurodags run pipeline.yml --derivative MyDerivative` — execute one derivative
- `neurodags dry-run pipeline.yml --derivative MyDerivative` — inspect the execution plan
- `neurodags dataframe pipeline.yml --format wide` — assemble dataframe output
- `neurodags dag pipeline.yml --html pipeline_dag.html` — export Mermaid DAG HTML
- `neurodags view path/to/file.nc` — launch the Dash explorer
- `neurodags tui pipeline.yml` — launch the Textual TUI (`neurodags[tui]` extra required)

- {doc}`quickstart` — a minimal working example
- {doc}`tui` — manage and run pipelines from the terminal
- {doc}`pipeline_yaml` — all `pipeline.yml` keys and derivative flags
- {doc}`datasets_yaml` — all `datasets.yml` fields and mount points
- {doc}`custom_nodes` — write and register custom nodes
- {doc}`dataframe_assembly` — build ML-ready dataframes from derivatives
- {doc}`inspection` — dry-run, error markers, DAG visualization, file explorer
- {doc}`parallelism` — local n_jobs, subset execution, error handling, HPC pointer
