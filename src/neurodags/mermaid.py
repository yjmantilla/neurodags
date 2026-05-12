"""Mermaid diagram generation for neurodags DAG visualization."""

from __future__ import annotations

import re
import webbrowser
from pathlib import Path
from typing import Any

_ID_REF = re.compile(r"^id\.(\d+)$")

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <style>
    body {{ font-family: sans-serif; margin: 2rem; background: #fafafa; }}
    h1 {{ color: #333; font-size: 1.4rem; }}
    .mermaid {{ background: white; padding: 1rem; border-radius: 4px; box-shadow: 0 1px 3px rgba(0,0,0,.15); }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <div class="mermaid">
{mermaid_content}
  </div>
  <script type="module">
    import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs';
    mermaid.initialize({{ startOnLoad: true, theme: 'default' }});
  </script>
</body>
</html>
"""


def _collect_id_refs(obj: Any) -> set[int]:
    refs: set[int] = set()
    if isinstance(obj, str):
        m = _ID_REF.match(obj)
        if m:
            refs.add(int(m.group(1)))
        return refs
    if isinstance(obj, dict):
        for v in obj.values():
            refs.update(_collect_id_refs(v))
        return refs
    if isinstance(obj, list | tuple | set):
        for v in obj:
            refs.update(_collect_id_refs(v))
        return refs
    return refs


def _safe_label(text: str) -> str:
    return text.replace('"', "'")


def derivative_to_mermaid(derivative_def: dict, derivative_name: str) -> str:
    """Generate a Mermaid graph string for a single derivative definition.

    Shapes:
      - SourceFile          → circle
      - derivative: Name    → cylinder (cached/stored artifact)
      - node: name          → rectangle (computation)
    """
    nodes = derivative_def.get("nodes", [])
    steps = {s["id"]: s for s in nodes}

    step_deps: dict[int, set[int]] = {}
    for sid, step in steps.items():
        declared = step.get("depends_on") or []
        deps = set(declared)
        if "args" in step:
            deps.update(_collect_id_refs(step.get("args")))
        deps.discard(sid)
        step_deps[sid] = deps

    lines = [f"    %% {derivative_name}", "    graph TD"]

    for sid, step in sorted(steps.items()):
        if "derivative" in step:
            name = _safe_label(step["derivative"])
            if name == "SourceFile":
                lines.append(f'      id{sid}((("{name}")))')
            else:
                lines.append(f'      id{sid}[("{name}")]')
        elif "node" in step:
            name = _safe_label(step["node"])
            lines.append(f'      id{sid}["{name}"]')

    for sid, deps in sorted(step_deps.items()):
        for dep in sorted(deps):
            lines.append(f"      id{dep} --> id{sid}")

    return "\n".join(lines)


def pipeline_to_mermaid(pipeline_config: dict) -> str:
    """Generate a Mermaid graph showing inter-derivative dependencies in a pipeline.

    Each derivative becomes a node. Edges drawn when one derivative references
    another via a ``derivative:`` step.
    """
    defs: dict[str, dict] = pipeline_config.get("DerivativeDefinitions", {}) or {}

    # Build edges: for each derivative, collect which derivatives it depends on
    dep_edges: list[tuple[str, str]] = []
    for deriv_name, deriv_def in defs.items():
        nodes = deriv_def.get("nodes", [])
        for step in nodes:
            if "derivative" in step:
                ref = step["derivative"]
                if ref == "SourceFile":
                    continue
                base = ref.split(".", 1)[0]
                if base in defs:
                    dep_edges.append((base, deriv_name))

    lines = ["    %% Pipeline DAG", "    graph TD"]

    for name in defs:
        label = _safe_label(name)
        lines.append(f'      {name}["{label}"]')

    for src, dst in dep_edges:
        lines.append(f"      {src} --> {dst}")

    return "\n".join(lines)


def save_mermaid_html(
    mermaid_content: str,
    output_path: str | Path | None = None,
    title: str = "DAG",
    auto_open: bool = False,
) -> Path:
    """Render *mermaid_content* into a standalone HTML file.

    Parameters
    ----------
    mermaid_content:
        Raw Mermaid diagram string (the content that goes inside the
        ``<div class="mermaid">`` block).
    output_path:
        Destination file. Defaults to ``{title}.html`` in the current directory.
    title:
        Page title and ``<h1>`` heading.
    auto_open:
        Open the file in the default browser after saving.
    """
    if output_path is None:
        safe_title = re.sub(r"[^\w\-.]", "_", title)
        output_path = Path(f"{safe_title}.html")
    output_path = Path(output_path)

    html = _HTML_TEMPLATE.format(
        title=title,
        mermaid_content=mermaid_content,
    )
    output_path.write_text(html, encoding="utf-8")

    if auto_open:
        webbrowser.open(output_path.resolve().as_uri())

    return output_path


def derivative_to_html(
    derivative_def: dict,
    derivative_name: str,
    output_path: str | Path | None = None,
    auto_open: bool = False,
) -> Path:
    """Convenience wrapper: generate Mermaid + save HTML for one derivative."""
    mermaid_str = derivative_to_mermaid(derivative_def, derivative_name)
    return save_mermaid_html(
        mermaid_str,
        output_path=output_path,
        title=derivative_name,
        auto_open=auto_open,
    )


def pipeline_to_html(
    pipeline_config: dict,
    output_path: str | Path | None = None,
    title: str = "Pipeline DAG",
    auto_open: bool = False,
) -> Path:
    """Convenience wrapper: generate Mermaid + save HTML for the full pipeline."""
    mermaid_str = pipeline_to_mermaid(pipeline_config)
    return save_mermaid_html(
        mermaid_str,
        output_path=output_path,
        title=title,
        auto_open=auto_open,
    )
