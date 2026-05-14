# Comparison with Other Workflow Managers

NeuroDAGs is designed specifically for neuroscience signal processing (EEG/MEG/ECG). While general-purpose tools like Snakemake or Pydra can be used for these tasks, NeuroDAGs provides a more ergonomic and domain-specific experience.

## Quick Comparison Table

| Feature | **NeuroDAGs** | **Snakemake / Snakebids** | **Pydra** |
| :--- | :--- | :--- | :--- |
| **Philosophy** | Derivative-centric (Push) | File-centric (Pull) | Task-centric Dataflow |
| **Config Style** | YAML + MNE Nodes | Python-based DSL | Pure Python API |
| **Output Naming** | Automatic \`@Derivative\` | Manual Wildcards / \`bids()\` | Managed Caching (machine-readable) |
| **BIDS Relation** | BIDS-Preserving (Flexible) | BIDS-Aware (Strict) | BIDS-Agnostic (Glue code needed) |
| **Aggregation** | Built-in \`dataframe\` | Manual "Reduce" Rules | Manual "Combiners" |
| **Data Types** | Native \`mne\` / \`xarray\` | Generic Files | Generic Python Objects |

---

## Why Choose NeuroDAGs?

### 1. Convention Over Configuration
In general-purpose managers, you spend significant time managing file paths and wildcards (e.g., \`{subject}_{session}_{task}\`).
**NeuroDAGs** uses a "Path-Preserving Prefix" strategy. It automatically appends \`@DerivativeName\` to the original filename. This ensures that your output directory structure perfectly mirrors your input, regardless of how many BIDS entities (run, acquisition, etc.) your data contains, without you ever writing a regex.

### 2. "Push" vs. "Pull" Logic
Tools like **Snakemake** are "Pull-based": you define what *output* you want, and it works backward. This can be complex when dealing with irregular datasets.
**NeuroDAGs** is "Push-based": you define a pipeline and "push" your files through it. This is often more intuitive for standard preprocessing and feature extraction workflows.

### 3. Built-in Data Aggregation
The goal of most signal processing pipelines is a tidy dataframe for statistical analysis.
NeuroDAGs includes the \`build_derivative_dataframe\` utility which understands \`xarray\` coordinates and metadata. It can crawl your derivatives and assemble them into a CSV/Parquet file in a single step, a process that usually requires manual script-writing in other frameworks.

### 4. Human-Readable Caching
While **Pydra** and others use cryptographic hashes for caching (e.g., \`_task_f83e2b1c/\`), NeuroDAGs creates human-readable files (e.g., \`sub-01_task-rest@Preprocessing.nc\`). You can browse your derivatives folder and immediately know what is what.

## Comparison Examples

### Snakemake / Snakebids
Requires mastering a Domain Specific Language (DSL), writing explicit rules, and managing complex \`expand()\` functions. Best for large-scale, heterogeneous file transformations or when strict BIDS-App compliance is the primary goal.

### Pydra
Excellent for developers building complex, branching dataflows in pure Python. It offers industrial-strength caching but requires significant "glue code" to manage BIDS file structures and human-readable exports.

### NeuroDAGs
Best for researchers who want to focus on **signal processing logic** using MNE-Python and Xarray. It eliminates "orchestration overhead" by handling naming, parallelization, and aggregation automatically via a simple YAML interface.
