from pathlib import Path
from unittest.mock import patch

from neurodags.dag import _MissingPrecomputedArtifacts, collect_derivative_for_dataframe, run_derivative
from neurodags.definitions import NodeResult


def _build_derivative_def(*, save: bool | None = None) -> dict:
    definition: dict[str, object] = {
        "nodes": [
            {
                "id": 0,
                "node": "dummy",
                "args": {"param1": "foo"},
            }
        ]
    }
    if save is not None:
        definition["save"] = save
    return definition


def _run_dummy_derivative(tmp_path: Path, derivative_def: dict) -> tuple[NodeResult, Path]:
    reference_base = tmp_path / "subject" / "sample"
    reference_base.parent.mkdir(parents=True, exist_ok=True)
    result = run_derivative(
        derivative_def,
        derivative_name="dummy_derivative",
        file_path="input-file.vhdr",
        reference_base=reference_base,
    )
    output_path = Path(f"{reference_base.as_posix()}@DummyDerivative.message.txt")
    return result, output_path


def test_run_derivative_saves_artifacts_by_default(tmp_path):
    derivative_def = _build_derivative_def()
    result, output_path = _run_dummy_derivative(tmp_path, derivative_def)

    assert isinstance(result, NodeResult)
    assert output_path.exists()
    assert output_path.read_text() == "Dummy derivative extraction completed with param1=foo and param2=None"


def test_run_derivative_skips_saving_when_flag_false(tmp_path):
    derivative_def = _build_derivative_def(save=False)
    result, output_path = _run_dummy_derivative(tmp_path, derivative_def)

    assert isinstance(result, NodeResult)
    assert not output_path.exists()


def test_save_false_does_not_trigger_cached_shortcut(tmp_path):
    save_derivative = _build_derivative_def()
    _, output_path = _run_dummy_derivative(tmp_path, save_derivative)
    assert output_path.exists()

    nosave_derivative = _build_derivative_def(save=False)
    result, _ = _run_dummy_derivative(tmp_path, nosave_derivative)

    assert isinstance(result, NodeResult)


def test_collect_derivative_for_dataframe_missing_flag_does_not_require_precomputed(tmp_path):
    with patch("neurodags.dag._resolve_derivative_artifacts", return_value={}) as resolver:
        collect_derivative_for_dataframe(
            {"save": True},
            "Demo",
            "input.vhdr",
            reference_base=tmp_path / "subject" / "sample",
        )

    assert resolver.call_args.kwargs["precomputed_only"] is False


def test_collect_derivative_for_dataframe_missing_cached_returns_empty_when_opted_in(tmp_path):
    with patch(
        "neurodags.dag._resolve_derivative_artifacts",
        side_effect=_MissingPrecomputedArtifacts,
    ):
        result = collect_derivative_for_dataframe(
            {"save": True, "for_dataframe": True},
            "Demo",
            "input.vhdr",
            reference_base=tmp_path / "subject" / "sample",
        )

    assert result == {}
