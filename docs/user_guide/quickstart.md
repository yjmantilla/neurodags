# Quickstart

See {doc}`/auto_examples/plot_quickstart_synthetic` — full synthetic pipeline, no real data required.

After creating `pipeline.yml` and `datasets.yml`, the equivalent CLI flow is:

```bash
neurodags validate pipeline.yml
neurodags dry-run pipeline.yml --derivative BasicPrep
neurodags run pipeline.yml
neurodags dataframe pipeline.yml --format wide --output derivative_dataframe.csv
```

## Template: preprocess your own recordings

`example_pipelines/` ships a ready-to-copy preprocessing template:
[`pipeline_quickstart_preprocessing.yml`](https://github.com/yjmantilla/neurodags/blob/main/example_pipelines/pipeline_quickstart_preprocessing.yml)
and
[`datasets_quickstart_preprocessing.yml`](https://github.com/yjmantilla/neurodags/blob/main/example_pipelines/datasets_quickstart_preprocessing.yml).
For each recording it band-passes 0.1–75 Hz, cuts 2 s epochs and resamples to 256 Hz with the
built-in `basic_preprocessing` node, and writes an MNE Report with the PSD after each step,
averaged across channels (`average_psd: true`). No custom code is needed.

Copy both files, edit the dataset's `source_root` / `file_pattern` / `derivatives_path`
(or set the environment variables they use), then. Relative paths in the datasets file are
resolved against that file's folder, and the pipeline file starts with `naming_version: 1`
(both required since 0.4.0):

```bash
export RAW_DATA=/path/to/rawdata OUTPUT_DIR=/path/to/output
neurodags validate pipeline_quickstart_preprocessing.yml   # prints the resolved paths
neurodags run pipeline_quickstart_preprocessing.yml
```

Outputs land in `$OUTPUT_DIR/derivatives/`, mirroring the folders below `$RAW_DATA`, as
`<file>@Preprocessed.fif` (the epochs) and `<file>@Preprocessed.report.html`, plus the
`<file>@Preprocessed.done` manifest that marks the file as done.
