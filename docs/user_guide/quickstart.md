# Quickstart

A minimal working pipeline from scratch.

## Project Layout

```
my_project/
├── datasets.yml
├── pipeline.yml
└── custom_nodes.py   # optional
```

## 1. Define Your Dataset

```yaml
# datasets.yml
my_dataset:
  name: MyDataset
  file_pattern:
    local: data/**/*.vhdr
  derivatives_path:
    local: outputs/
```

## 2. Define Your Pipeline

```yaml
# pipeline.yml
datasets: datasets.yml
mount_point: local

DerivativeDefinitions:
  CleanedEEG:
    nodes:
      - id: 0
        derivative: SourceFile
      - id: 1
        node: basic_preprocessing
        args:
          mne_object: id.0
          resample: 256
          filter_args:
            l_freq: 0.5
            h_freq: 110

  BandPower:
    for_dataframe: True
    nodes:
      - id: 0
        derivative: CleanedEEG.fif
      - id: 1
        node: bandpower
        args:
          psd_like: id.0
          relative: True
          bands:
            delta: [1.0, 4.0]
            theta: [4.0, 8.0]
            alpha: [8.0, 13.0]
            beta: [13.0, 30.0]

DerivativeList:
  - CleanedEEG
  - BandPower
```

## 3. Run the Pipeline

```python
from neurodags.loaders import load_configuration
from neurodags.orchestrators import iterate_derivative_pipeline

config = load_configuration("pipeline.yml")

# Run each derivative in order
for derivative in config["DerivativeList"]:
    iterate_derivative_pipeline(config, derivative)
```

## 4. Build a Dataframe

```python
from neurodags.orchestrators import build_derivative_dataframe

df = build_derivative_dataframe("pipeline.yml", output_format="wide")
print(df.head())
```

## 5. Inspect Before Running (Dry Run)

```python
plan = iterate_derivative_pipeline(config, "BandPower", dry_run=True)
print(plan)
```

## Output Naming

Derivatives are saved alongside their source files with a `@DerivativeName` suffix:

```
outputs/
  sub-1@CleanedEEG.fif
  sub-1@BandPower.nc
  sub-2@CleanedEEG.fif
  sub-2@BandPower.nc
```
