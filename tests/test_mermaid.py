"""Tests for mermaid DAG visualization."""

from pathlib import Path

from neurodags.mermaid import (
    derivative_to_html,
    derivative_to_mermaid,
    pipeline_to_html,
    pipeline_to_mermaid,
    save_mermaid_html,
)

SIMPLE_DERIV = {
    "overwrite": False,
    "save": True,
    "nodes": [
        {"id": 0, "derivative": "SourceFile"},
        {"id": 1, "node": "my_node", "args": {"data": "id.0"}},
    ],
}

MULTI_NODE_DERIV = {
    "nodes": [
        {"id": 0, "derivative": "UpstreamDeriv.nc"},
        {"id": 1, "node": "extract_data_var", "args": {"dataset_like": "id.0", "data_var": "x"}},
        {"id": 2, "node": "aggregate", "args": {"data": "id.1"}},
    ]
}

PIPELINE_CONFIG = {
    "DerivativeDefinitions": {
        "StepA": {
            "nodes": [
                {"id": 0, "derivative": "SourceFile"},
                {"id": 1, "node": "node_a", "args": {"data": "id.0"}},
            ]
        },
        "StepB": {
            "nodes": [
                {"id": 0, "derivative": "StepA.nc"},
                {"id": 1, "node": "node_b", "args": {"data": "id.0"}},
            ]
        },
        "StepC": {
            "nodes": [
                {"id": 0, "derivative": "StepB.nc"},
                {"id": 1, "node": "node_c", "args": {"data": "id.0"}},
            ]
        },
    }
}


class TestDerivativeToMermaid:
    def test_source_file_is_circle(self):
        result = derivative_to_mermaid(SIMPLE_DERIV, "Simple")
        assert '((("SourceFile")))' in result

    def test_node_is_rectangle(self):
        result = derivative_to_mermaid(SIMPLE_DERIV, "Simple")
        assert '["my_node"]' in result

    def test_derivative_ref_is_cylinder(self):
        result = derivative_to_mermaid(MULTI_NODE_DERIV, "Multi")
        assert '[("UpstreamDeriv.nc")]' in result

    def test_edges_from_args_refs(self):
        result = derivative_to_mermaid(SIMPLE_DERIV, "Simple")
        assert "id0 --> id1" in result

    def test_multi_node_edges(self):
        result = derivative_to_mermaid(MULTI_NODE_DERIV, "Multi")
        assert "id0 --> id1" in result
        assert "id1 --> id2" in result

    def test_contains_graph_td(self):
        result = derivative_to_mermaid(SIMPLE_DERIV, "Simple")
        assert "graph TD" in result

    def test_comment_has_derivative_name(self):
        result = derivative_to_mermaid(SIMPLE_DERIV, "MyDeriv")
        assert "%% MyDeriv" in result

    def test_explicit_depends_on(self):
        deriv = {
            "nodes": [
                {"id": 0, "derivative": "SourceFile"},
                {"id": 1, "node": "node_a", "args": {"data": "id.0"}},
                {"id": 2, "node": "node_b", "depends_on": [0, 1]},
            ]
        }
        result = derivative_to_mermaid(deriv, "Test")
        assert "id0 --> id2" in result
        assert "id1 --> id2" in result


class TestPipelineToMermaid:
    def test_all_derivatives_present(self):
        result = pipeline_to_mermaid(PIPELINE_CONFIG)
        assert "StepA" in result
        assert "StepB" in result
        assert "StepC" in result

    def test_inter_derivative_edges(self):
        result = pipeline_to_mermaid(PIPELINE_CONFIG)
        assert "StepA --> StepB" in result
        assert "StepB --> StepC" in result

    def test_source_file_excluded_as_node(self):
        result = pipeline_to_mermaid(PIPELINE_CONFIG)
        assert "SourceFile" not in result

    def test_graph_td_present(self):
        result = pipeline_to_mermaid(PIPELINE_CONFIG)
        assert "graph TD" in result


class TestSaveMermaidHtml:
    def test_creates_file(self, tmp_path):
        out = tmp_path / "test.html"
        save_mermaid_html("graph TD\n  A --> B", output_path=out, title="Test")
        assert out.exists()

    def test_html_contains_mermaid_content(self, tmp_path):
        out = tmp_path / "test.html"
        save_mermaid_html("graph TD\n  A --> B", output_path=out)
        content = out.read_text()
        assert "graph TD" in content
        assert "A --> B" in content

    def test_html_contains_mermaid_script(self, tmp_path):
        out = tmp_path / "test.html"
        save_mermaid_html("graph TD", output_path=out)
        content = out.read_text()
        assert "mermaid" in content
        assert "cdn.jsdelivr.net" in content

    def test_default_output_path_uses_title(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = save_mermaid_html("graph TD", title="MyDAG")
        assert result == Path("MyDAG.html")
        assert result.exists()

    def test_returns_path(self, tmp_path):
        out = tmp_path / "out.html"
        result = save_mermaid_html("graph TD", output_path=out)
        assert isinstance(result, Path)
        assert result == out

    def test_title_in_html(self, tmp_path):
        out = tmp_path / "test.html"
        save_mermaid_html("graph TD", output_path=out, title="My Pipeline")
        content = out.read_text()
        assert "My Pipeline" in content


class TestConvenienceWrappers:
    def test_derivative_to_html(self, tmp_path):
        out = tmp_path / "deriv.html"
        result = derivative_to_html(SIMPLE_DERIV, "Simple", output_path=out)
        assert result.exists()
        content = result.read_text()
        assert "Simple" in content
        assert "graph TD" in content

    def test_pipeline_to_html(self, tmp_path):
        out = tmp_path / "pipeline.html"
        result = pipeline_to_html(PIPELINE_CONFIG, output_path=out)
        assert result.exists()
        content = result.read_text()
        assert "graph TD" in content
        assert "StepA" in content
