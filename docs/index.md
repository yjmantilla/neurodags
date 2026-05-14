---
html_theme_sidebar_secondary: false
---

# NeuroDAGs

[![PyPI](https://img.shields.io/pypi/v/neurodags)](https://pypi.org/project/neurodags/)

**An Extensible and Declarative DAG Framework for Reproducible Neuroscience Workflows**

M/EEG studies generate many interdependent intermediate derivatives. Recomputing full pipelines is wasteful; reusing valid intermediates is non-trivial. NeuroDAGs solves this with a declarative, graph-based framework for scalable, reproducible derivative computation.

**Core idea:** Pipelines are defined as a directed acyclic graph (DAG) of computation nodes. Each node outputs a reusable derivative. The DAG executes for each input file, skipping already-computed derivatives automatically.

```{toctree}
:maxdepth: 2
:hidden:

user_guide/index
api/index
api/src/neurodags/index
auto_examples/index
changelog
COMPARISON
```

## Poster

[View the NeuroDAGs conference poster](https://canva.link/c2b2fm0mk2wotwq) — overview of motivation, design, and format.

## Get Started

- {doc}`user_guide/installation` — install neurodags
- {doc}`user_guide/tui` — manage and run pipelines from the terminal
- {doc}`auto_examples/plot_quickstart_synthetic` — minimal working pipeline in minutes
- {doc}`user_guide/concepts` — understand DAGs, derivatives, nodes, and artifacts
- {doc}`user_guide/pipeline_yaml` — full `pipeline.yml` reference
- {doc}`user_guide/datasets_yaml` — full `datasets.yml` reference
- {doc}`user_guide/custom_nodes` — write and register your own nodes
- {doc}`user_guide/dataframe_assembly` — build ML-ready dataframes from derivatives
- {doc}`user_guide/inspection_parallelism` — dry-run, visualization, HPC parallelism
