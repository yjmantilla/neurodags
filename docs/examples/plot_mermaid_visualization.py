"""
DAG Visualization with Mermaid
================================

NeuroDAGs can render any pipeline or derivative definition as an interactive
`Mermaid <https://mermaid.js.org/>`_ diagram saved to a standalone HTML file.

Two levels of detail are available:

- **Pipeline DAG** — high-level view: one node per derivative, edges showing
  inter-derivative dependencies.
- **Derivative DAG** — fine-grained view: every computation node and data
  reference inside a single derivative.
"""

# %%
# Setup
# -----
import yaml

# %%
# Sample pipeline
# ----------------
# We define a three-step pipeline in YAML — the same format used in a real
# ``pipeline.yml`` file.

PIPELINE_YAML = """\
DerivativeDefinitions:

  BasicPrep:
    nodes:
      - id: 0
        derivative: SourceFile
      - id: 1
        node: basic_preprocessing
        args:
          mne_object: id.0
          filter_args: {l_freq: 1.0, h_freq: 80.0}
          epoch_config: {duration: 2.0, overlap: 0.0}

  Spectrum:
    nodes:
      - id: 0
        derivative: BasicPrep.fif
      - id: 1
        node: mne_spectrum_array
        args:
          meeg: id.0
          method: welch

  BandPower:
    save: false
    for_dataframe: true
    nodes:
      - id: 0
        derivative: Spectrum.nc
      - id: 1
        node: extract_data_var
        args: {dataset_like: id.0, data_var: spectrum}
      - id: 2
        node: bandpower
        args:
          psd_like: id.1
          relative: true
          bands:
            delta: [1.0,  4.0]
            theta: [4.0,  8.0]
            alpha: [8.0, 13.0]
            beta:  [13.0, 30.0]
      - id: 3
        node: aggregate_across_dimension
        args: {xarray_data: id.2, dim: epochs, operation: mean}
"""

pipeline_config = yaml.safe_load(PIPELINE_YAML)

# %%
# Pipeline-level Mermaid diagram
# --------------------------------
# :func:`~neurodags.mermaid.pipeline_to_mermaid` returns the raw Mermaid string.
# :func:`~neurodags.mermaid.pipeline_to_html` saves it to a self-contained HTML
# file that renders in any browser.

from neurodags.mermaid import pipeline_to_html, pipeline_to_mermaid

mermaid_str = pipeline_to_mermaid(pipeline_config)
print("Pipeline Mermaid diagram:")
print(mermaid_str)

# %%
# Save to HTML (opens automatically when ``auto_open=True``).

import tempfile
from pathlib import Path

out_dir = Path(tempfile.mkdtemp(prefix="neurodags_mermaid_"))

pipeline_html = pipeline_to_html(
    pipeline_config,
    output_path=out_dir / "pipeline_dag.html",
    title="My Pipeline DAG",
    auto_open=False,  # set True to open in browser
)
print(f"Pipeline DAG saved to: {pipeline_html}")

# %%
# Derivative-level Mermaid diagram
# ----------------------------------
# Zoom into a single derivative to see every node and data reference.
#
# Node shapes:
#
# - **Circle** ``(((...)))`` — ``SourceFile`` (raw input).
# - **Cylinder** ``[(...)]`` — upstream derivative artifact (cached on disk).
# - **Rectangle** ``[...]`` — computation node.

from neurodags.mermaid import derivative_to_html, derivative_to_mermaid

deriv_name = "BandPower"
deriv_def = pipeline_config["DerivativeDefinitions"][deriv_name]

mermaid_str = derivative_to_mermaid(deriv_def, deriv_name)
print(f"\n{deriv_name} derivative Mermaid diagram:")
print(mermaid_str)

# %%
# Save the derivative diagram to HTML.

deriv_html = derivative_to_html(
    deriv_def,
    deriv_name,
    output_path=out_dir / f"{deriv_name}_dag.html",
    auto_open=False,
)
print(f"{deriv_name} DAG saved to: {deriv_html}")

# %%
# Using a real pipeline.yml
# --------------------------
# In a real project, load your config from disk and call the same functions:
#
# .. code-block:: python
#
#     import yaml
#     from neurodags.mermaid import pipeline_to_html, derivative_to_html
#
#     with open("pipeline.yml") as f:
#         config = yaml.safe_load(f)
#
#     # Full pipeline overview
#     pipeline_to_html(config, output_path="pipeline_dag.html", auto_open=True)
#
#     # Single derivative detail
#     derivative_to_html(
#         config["DerivativeDefinitions"]["BandPower"],
#         "BandPower",
#         output_path="bandpower_dag.html",
#         auto_open=True,
#     )
#
# CLI equivalents::
#
#     neurodags dag pipeline.yml
#     neurodags dag pipeline.yml --html pipeline_dag.html
#     neurodags dag pipeline.yml --derivative BandPower --html bandpower_dag.html

# %%
# Direct Mermaid string access
# -----------------------------
# Use the raw string functions when you need to embed diagrams in Jupyter
# notebooks or custom HTML templates.

from neurodags.mermaid import save_mermaid_html

custom_diagram = """    graph TD
      A["load_raw"] --> B["filter"]
      B --> C["epoch"]
      C --> D[("BasicPrep.fif")]"""

out = save_mermaid_html(
    custom_diagram,
    output_path=out_dir / "custom_dag.html",
    title="Custom DAG",
)
print(f"Custom DAG saved to: {out}")
