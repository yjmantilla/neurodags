# Comparison with Other Workflow Managers

NeuroDAGs is designed specifically for neuroscience signal processing (EEG/MEG/ECG). While general-purpose tools like Snakemake or Pydra can be used for these tasks, NeuroDAGs provides a more ergonomic and domain-specific experience.

## Quick Comparison Table

| Feature | **NeuroDAGs** | **Snakemake / Snakebids** | **Pydra** |
| :--- | :--- | :--- | :--- |
| **Philosophy** | Derivative-centric (Push) | File-centric (Pull) | Task-centric Dataflow |
| **Config Style** | YAML + Python nodes | Python-based DSL | Pure Python API |
| **Output Naming** | Automatic \`@Derivative\` | Manual Wildcards / \`bids()\` | Managed Caching (machine-readable) |
| **BIDS Relation** | BIDS-Preserving (Flexible) | BIDS-Aware (Strict) | BIDS-Agnostic (Glue code needed) |
| **Aggregation** | Built-in \`dataframe\` | Manual "Reduce" Rules | Manual "Combiners" |
| **Data Types** | MNE / xarray first-class; generic artifacts supported | Generic Files | Generic Python Objects |
| **Cluster Scheduling** | Manual (joblib parallelism within a job) | Native SLURM/SGE/PBS | Native SLURM/PBS via Dask/CF |
| **Provenance Tracking** | Human-readable `.error` markers; no full provenance graph | Full provenance via DAG | Hash-based provenance |
| **Maturity / Community** | Early-stage, small community | Mature, large community | Active, growing community |

---

## Why Choose NeuroDAGs?

### 1. Convention Over Configuration
In general-purpose managers, you spend significant time managing file paths and wildcards (e.g., \`{subject}_{session}_{task}\`).
**NeuroDAGs** uses a "Path-Preserving Prefix" strategy. It automatically appends \`@DerivativeName\` to the original filename. This ensures that your output directory structure perfectly mirrors your input, regardless of how many BIDS entities (run, acquisition, etc.) your data contains, without you ever writing a regex.

### 2. "Push" vs. "Pull" Logic
Tools like **Snakemake** are "Pull-based": you define what *output* you want, and it works backward. This is powerful for heterogeneous pipelines but can be complex for irregular datasets.
**NeuroDAGs** is "Push-based": you define a pipeline and "push" your files through it. Derivatives are automatically sorted by dependency order before execution. This is often more intuitive for standard preprocessing and feature extraction workflows.

### 3. Built-in Data Aggregation
The goal of most signal processing pipelines is a tidy dataframe for statistical analysis.
NeuroDAGs includes the \`build_derivative_dataframe\` utility which understands \`xarray\` coordinates and metadata. It can crawl your derivatives and assemble them into a CSV/Parquet file in a single step, a process that usually requires manual script-writing in other frameworks.

### 4. Human-Readable Caching
While **Pydra** and others use cryptographic hashes for caching (e.g., \`_task_f83e2b1c/\`), NeuroDAGs creates human-readable files (e.g., \`sub-01_task-rest@Preprocessing.nc\`). You can browse your derivatives folder and immediately know what is what. Failed runs write a \`.error\` marker alongside the expected output; successful retries clean it up automatically.

### 5. Custom Nodes in Plain Python
Nodes are plain Python functions decorated with \`@register_node\`. They can be defined in external files loaded via \`new_definitions\` in the YAML — no forking or subclassing required. MNE and xarray are first-class, but the artifact system accepts any writer function, so other file formats work too.

---

## When NOT to Use NeuroDAGs

- **Strict BIDS-App compliance required**: Snakebids has native BIDS validation and output layout enforcement. NeuroDAGs preserves input paths but does not validate against the BIDS spec.
- **Complex cluster scheduling**: If you need to submit per-derivative jobs to SLURM/SGE/PBS with dependency chaining, Snakemake or Pydra have native cluster backends. NeuroDAGs parallelises within a single job via joblib; multi-job orchestration is manual.
- **Full provenance tracking**: NeuroDAGs has no provenance graph — it cannot tell you which version of a node produced a given file. If audit trails matter, Pydra's hash-based caching or DVC are better fits.
- **Non-neuroscience / non-file-based pipelines**: NeuroDAGs assumes one input file → one set of derivative files. It is not designed for database records, streaming data, or large-scale generic ETL.
- **Large, mature community support**: NeuroDAGs is early-stage. Snakemake and Nipype have years of community plugins, tutorials, and battle-tested HPC configurations.

---

## Concrete BIDS Examples

The same task — smooth every `bold` file in a BIDS dataset — shown in each framework.

### Snakebids

Snakebids wraps Snakemake with a `generate_inputs()` call that parses the BIDS dataset via pybids. You then write a Snakemake rule using the `bids()` helper for output naming, and a `rule all` that calls `expand()` to iterate over every discovered subject/session/run combination.

```python
# config.yaml
pybids_inputs:
  bold:
    filters:
      suffix: bold
      extension: .nii.gz
      datatype: func
    wildcards:
      - subject
      - task
      - run
```

```python
# Snakefile
from snakebids import generate_inputs, bids

inputs = generate_inputs(bids_dir=config['bids_dir'],
                         pybids_inputs=config['pybids_inputs'])

rule all:
    input:
        inputs['bold'].expand(
            bids(root='results', fwhm='{fwhm}', suffix='bold.nii.gz',
                 **inputs['bold'].wildcards),
            fwhm=config['fwhm'],
        )

rule smooth:
    input:  inputs['bold'].path
    output: bids(root='results', fwhm='{fwhm}', suffix='bold.nii.gz',
                 **inputs['bold'].wildcards)
    shell:  'fslmaths {input} -s {params.sigma} {output}'
```

A `run.py` BIDS-App entry point is also required:

```python
from snakebids import bidsapp, plugins
app = bidsapp.app([plugins.SnakemakeBidsApp(Path(__file__).resolve().parent)])
```

**Verdict**: Excellent BIDS-App compliance and strict output validation. Requires learning Snakemake DSL, `expand()` mechanics, wildcard management, and pybids filter config. Best when strict BIDS-App compliance or heterogeneous file transformations are the primary goal.

---

### Pydra

Pydra uses Python decorators to define tasks. BIDS file discovery relies on external helpers (e.g. Nilearn's `first_level_from_bids`). The pipeline is a two-level workflow executed via a concurrent submitter. Results are cached under `~/.cache/pydra/` using cryptographic hashes.

```python
from pydra.mark import python

@python.define(outputs=["smoothed_path"])
def SmoothFile(bold_path: str, fwhm: float) -> str:
    import nibabel as nib
    from nilearn.image import smooth_img
    img = smooth_img(bold_path, fwhm=fwhm)
    out = bold_path.replace('.nii.gz', f'_smooth-{fwhm}.nii.gz')
    img.to_filename(out)
    return out

# BIDS file discovery requires an external call
from nilearn.glm.first_level import first_level_from_bids
models, run_imgs, events, confounds = first_level_from_bids(
    dataset_path=bids_dir, task_label='rest', space_label='MNI152')

# Build and run workflow
wf = pydra.Workflow(name="smooth_wf", input_spec=["bold_files"])
wf.add(SmoothFile(name="smooth", bold_path=wf.lzin.bold_files, fwhm=6.0))

with pydra.Submitter(worker='cf', n_procs=4) as sub:
    results = sub(wf)
```

Output files are saved under manually constructed paths — the framework manages caching internally but does not enforce human-readable derivative naming.

**Verdict**: Industrial-strength for complex branching dataflows in pure Python. Significant "glue code" needed to handle BIDS file discovery, human-readable output paths, and aggregation. Hash-based caching is robust but opaque.

---

### NeuroDAGs

File discovery, output naming, dependency ordering, and aggregation are handled by the framework. You write the signal processing logic; the YAML wires it up.

```yaml
# datasets.yml
my_dataset:
  name: MyStudy
  file_pattern: /data/bids/**/*.vhdr
  derivatives_path: /data/derivatives
```

```yaml
# pipeline.yml
DerivativeList:
  - Preprocessed
  - BandPower

DerivativeDefinitions:
  Preprocessed:
    nodes:
      - id: 0
        derivative: SourceFile
      - id: 1
        node: basic_preprocessing
        args:
          mne_object: id.0
          filter_args: {l_freq: 1.0, h_freq: 40.0}

  BandPower:
    for_dataframe: true
    nodes:
      - id: 0
        derivative: Preprocessed.fif   # depends on Preprocessed
      - id: 1
        node: bandpower
        args:
          psd_like: id.0
          bands: {alpha: [8, 13], beta: [13, 30]}
```

```python
from neurodags.loaders import load_configuration
from neurodags.orchestrators import run_pipeline

config = load_configuration("pipeline.yml")
run_pipeline(config)   # discovers files, sorts by dependency, runs all
```

Output files are named automatically: `sub-01_task-rest.vhdr@Preprocessed.fif`, `sub-01_task-rest.vhdr@BandPower.nc`. No wildcards, no `expand()`, no `bids()` helper needed.

**Verdict**: Lowest orchestration overhead for MNE-Python / xarray signal processing pipelines. Not a BIDS-App; does not validate BIDS compliance. No native cluster backend — parallelism is joblib within a single job.

---

### Nipype
The established standard for wrapping neuroimaging CLIs (FSL, FreeSurfer, ANTs, SPM). Best when you need to call existing command-line tools. NeuroDAGs targets Python-native signal processing rather than CLI wrapping.
