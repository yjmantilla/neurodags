"""Quick demo: generate mermaid HTML for the epilepsy pipeline."""

import yaml

from neurodags.mermaid import derivative_to_html, pipeline_to_html

with open("example_pipelines/pipeline_epilepsy.yml") as f:
    cfg = yaml.safe_load(f)

# Full pipeline inter-derivative graph
p = pipeline_to_html(cfg, output_path="pipeline_dag.html", auto_open=True)
print(f"Pipeline DAG: {p.resolve()}")

# One derivative in detail
name = "apSlopeMean"
d = derivative_to_html(
    cfg["DerivativeDefinitions"][name],
    name,
    output_path=f"{name}_dag.html",
    auto_open=True,
)
print(f"Derivative DAG: {d.resolve()}")
