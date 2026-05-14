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


def _error_path(tmp_path: Path) -> Path:
    reference_base = tmp_path / "subject" / "sample"
    return Path(f"{reference_base.as_posix()}@DummyDerivative.error")


def test_skip_errors_skips_file_with_error_marker(tmp_path):
    derivative_def = _build_derivative_def()
    reference_base = tmp_path / "subject" / "sample"
    reference_base.parent.mkdir(parents=True, exist_ok=True)
    error_file = _error_path(tmp_path)
    error_file.write_text("prior failure")

    result = run_derivative(
        derivative_def,
        derivative_name="dummy_derivative",
        file_path="input.vhdr",
        reference_base=reference_base,
        skip_errors=True,
    )

    assert result == {"skipped_error": str(error_file)}
    # output not produced
    assert not Path(f"{reference_base.as_posix()}@DummyDerivative.message.txt").exists()
    # error marker untouched
    assert error_file.exists()


def test_skip_errors_false_retries_file_with_error_marker(tmp_path):
    derivative_def = _build_derivative_def()
    reference_base = tmp_path / "subject" / "sample"
    reference_base.parent.mkdir(parents=True, exist_ok=True)
    error_file = _error_path(tmp_path)
    error_file.write_text("prior failure")

    result = run_derivative(
        derivative_def,
        derivative_name="dummy_derivative",
        file_path="input.vhdr",
        reference_base=reference_base,
        skip_errors=False,
    )

    assert isinstance(result, NodeResult)
    output = Path(f"{reference_base.as_posix()}@DummyDerivative.message.txt")
    assert output.exists()


def test_successful_run_removes_stale_error_marker(tmp_path):
    derivative_def = _build_derivative_def()
    reference_base = tmp_path / "subject" / "sample"
    reference_base.parent.mkdir(parents=True, exist_ok=True)
    error_file = _error_path(tmp_path)
    error_file.write_text("prior failure")

    run_derivative(
        derivative_def,
        derivative_name="dummy_derivative",
        file_path="input.vhdr",
        reference_base=reference_base,
        skip_errors=False,
    )

    assert not error_file.exists()
    output = Path(f"{reference_base.as_posix()}@DummyDerivative.message.txt")
    assert output.exists()


def test_dry_run_reports_error_marker(tmp_path):
    derivative_def = _build_derivative_def()
    reference_base = tmp_path / "subject" / "sample"
    reference_base.parent.mkdir(parents=True, exist_ok=True)
    error_file = _error_path(tmp_path)
    error_file.write_text("prior failure")

    result = run_derivative(
        derivative_def,
        derivative_name="dummy_derivative",
        file_path="input.vhdr",
        reference_base=reference_base,
        dry_run=True,
    )

    final_entry = next(e for e in result["plan"] if e.get("id") == "final")
    assert final_entry["has_error_marker"] is True
    assert final_entry["error_path"] is not None


def test_dry_run_no_error_marker_reports_false(tmp_path):
    derivative_def = _build_derivative_def()
    reference_base = tmp_path / "subject" / "sample"
    reference_base.parent.mkdir(parents=True, exist_ok=True)

    result = run_derivative(
        derivative_def,
        derivative_name="dummy_derivative",
        file_path="input.vhdr",
        reference_base=reference_base,
        dry_run=True,
    )

    final_entry = next(e for e in result["plan"] if e.get("id") == "final")
    assert final_entry["has_error_marker"] is False
    assert final_entry["error_path"] is None


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
