# Quickstart

See {doc}`/auto_examples/plot_quickstart_synthetic` — full synthetic pipeline, no real data required.

After creating `pipeline.yml` and `datasets.yml`, the equivalent CLI flow is:

```bash
neurodags validate pipeline.yml
neurodags dry-run pipeline.yml --derivative BasicPrep
neurodags run pipeline.yml
neurodags dataframe pipeline.yml --format wide --output derivative_dataframe.csv
```
