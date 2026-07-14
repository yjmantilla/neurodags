# Parser Nodes

A **parser node** takes a file path and returns a flat dict of *properties*
derived from the filename (subject, session, task, condition, …). Downstream
nodes then consume individual properties as regular `id.N.field` references,
instead of each node re-implementing `Path(path).stem.split(...)` internally.

Parser nodes are ordinary nodes — the framework needs no special type. The only
supporting feature is `id.N.field` [field access](pipeline_yaml.md) in the arg
resolver.

## Built-in: `bids_parse`

`bids_parse` extracts BIDS entities from a path using
`mne_bids.get_entities_from_fname`:

```yaml
CleanedPrep:
  nodes:
    - id: 0
      derivative: SourceFile          # e.g. sub-001_ses-01_task-rest_eeg.fif
    - id: 1
      node: bids_parse
      args:
        path: id.0
    - id: 2
      node: extract_condition_epochs
      args:
        raw:     id.0
        subject: id.1.subject          # "001"
        session: id.1.session          # "01"
        task:    id.1.task             # "rest"
```

`id.1` alone is the whole entity dict; `id.1.subject` pulls one field. Entities
absent from the filename are simply omitted from the dict (a missing-field
reference raises with the list of available fields), and non-BIDS names yield
few/no entities rather than erroring.

## Writing a custom parser

A parser is just a node that returns a `NodeResult` wrapping a flat dict:

```python
import json
from pathlib import Path
from neurodags.definitions import Artifact, NodeResult
from neurodags.nodes import register_node


@register_node(name="my_lab_parse")
def my_lab_parse(path):
    # /data/recordings/P001_EO_run2.edf -> {"subject","condition","run"}
    subject, condition, run = Path(path).stem.split("_")
    props = {"subject": subject, "condition": condition, "run": run}
    return NodeResult(
        artifacts={".json": Artifact(item=props, writer=lambda p: Path(p).write_text(json.dumps(props)))}
    )
```

```yaml
- id: 1
  node: my_lab_parse
  args: {path: id.0}
- id: 2
  node: my_processing_node
  args: {raw: id.0, condition: id.1.condition}   # "EO"
```

Point the pipeline at the module via `new_definitions:` (see
[custom nodes](custom_nodes.md)).

## The field-key constraint

```{important}
Because `.` is the reference separator in `id.N.field`, the dict **keys must be
plain identifiers** — letters, digits and underscores, not starting with a
digit. Dotted, dashed, or nested keys cannot be addressed and raise a clear
error (`id.1.a.b`, `id.1.some-key` are rejected).

- Keep parser output **flat** — no nested dicts. BIDS entities already comply.
- **Values are unconstrained** — a value may be any JSON-serializable type,
  including strings that contain dots (e.g. a path).
- Need the whole mapping? Reference `id.N` (no field) and index it in the node.
```

## Design contract

- Return a **flat, JSON-serializable dict** whose keys are plain identifiers.
- Do **no file I/O** — parse the path string only; keep the node pure and fast.
- Typically set `save: False` on a parser-only derivative (the dict is ephemeral
  and consumed in-memory). If saved, it is written as a readable `.json` sidecar.

## Unit-testing a parser

Because a parser is pure, it tests in isolation without a pipeline:

```python
from neurodags.nodes import get_node

def test_bids_parse():
    props = get_node("bids_parse")("sub-001_task-rest_eeg.fif").artifacts[".json"].item
    assert props["subject"] == "001"
    assert props["task"] == "rest"
    assert all(k.isidentifier() for k in props)   # referenceable as id.N.<key>
```
