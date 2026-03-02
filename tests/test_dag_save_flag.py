from pathlib import Path

from cocofeats.dag import run_derivative
from cocofeats.definitions import NodeResult


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
