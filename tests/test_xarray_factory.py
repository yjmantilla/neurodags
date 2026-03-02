from __future__ import annotations

import json

import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_allclose

from neurodags.definitions import Artifact, NodeResult
from neurodags.nodes import get_node
from neurodags.nodes.factories import apply_1d

try:  # pragma: no cover - optional dependency during tests
    import mne
except Exception:  # pragma: no cover - surface import-time optionality
    mne = None  # type: ignore[assignment]


def _make_dataarray() -> xr.DataArray:
    data = np.arange(12, dtype=float).reshape(3, 4)
    return xr.DataArray(
        data,
        dims=("spaces", "times"),
        coords={
            "spaces": ["C3", "Cz", "C4"],
            "times": np.linspace(0.0, 0.3, 4),
        },
    )


def test_apply_1d_scalar_output() -> None:
    arr = _make_dataarray()
    result = apply_1d(arr, dim="times", pure_function=np.mean)

    assert isinstance(result, xr.DataArray)
    assert result.dims == ("spaces",)
    assert_allclose(result.values, arr.mean(dim="times").values)


def test_apply_1d_sequence_output_with_coords() -> None:
    arr = _make_dataarray()

    def stats(vector: np.ndarray) -> tuple[float, float]:
        return float(vector.mean()), float(vector.std())

    result = apply_1d(
        arr,
        dim="times",
        pure_function=stats,
        result_dim="stat",
        result_coords=("mean", "std"),
    )

    assert isinstance(result, xr.DataArray)
    assert result.dims == ("spaces", "stat")
    assert list(result.coords["stat"].values) == ["mean", "std"]
    assert_allclose(result.sel(stat="mean").values, arr.mean(dim="times").values)


def test_iterative_mode_matches_vectorized_and_reports_timings() -> None:
    arr = _make_dataarray()

    def stats(vector: np.ndarray) -> np.ndarray:
        return np.array([vector.sum(), vector.mean()])

    vector_result = apply_1d(
        arr,
        dim="times",
        pure_function=stats,
        result_dim="stat",
        result_coords=("sum", "mean"),
        mode="vectorized",
    )
    iterative_result = apply_1d(
        arr,
        dim="times",
        pure_function=stats,
        result_dim="stat",
        result_coords=("sum", "mean"),
        mode="iterative",
    )

    assert isinstance(vector_result, xr.DataArray)
    assert isinstance(iterative_result, xr.DataArray)
    assert_allclose(iterative_result.values, vector_result.values)

    metadata = json.loads(iterative_result.attrs["metadata"])
    assert metadata["mode"] == "iterative"
    assert metadata["per_slice_duration_unit"] == "seconds"

    timing_da = xr.DataArray.from_dict(metadata["per_slice_duration"])
    assert timing_da.dims == iterative_result.dims
    assert timing_da.shape == iterative_result.shape
    assert np.all(timing_da.values >= 0.0)


def test_apply_1d_accepts_noderesult_input() -> None:
    arr = _make_dataarray()
    node_input = NodeResult({".nc": Artifact(item=arr, writer=lambda path: arr.to_netcdf(path))})

    result = apply_1d(node_input, dim="times", pure_function=np.mean)

    assert isinstance(result, xr.DataArray)
    assert result.shape == (3,)
    assert_allclose(result.values, arr.mean(dim="times").values)


@pytest.mark.skipif(mne is None, reason="mne not available")
def test_apply_1d_handles_mne_raw() -> None:
    pytest.importorskip("scipy", reason="SciPy required for MNE test")

    sfreq = 100.0
    data = np.vstack([
        np.linspace(0.0, 1.0, 200),
        np.linspace(1.0, 2.0, 200),
    ])
    info = mne.create_info(ch_names=["C3", "C4"], sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose="error")

    result = apply_1d(raw, dim="times", pure_function=np.mean)

    assert isinstance(result, xr.DataArray)
    assert result.dims == ("spaces",)
    expected = raw.get_data().mean(axis=1)
    assert_allclose(result.values, expected)


def test_xarray_factory_registered_node_supports_dotted_path() -> None:
    node = get_node("xarray_factory")
    arr = _make_dataarray()

    result = node(arr, dim="times", pure_function="numpy.mean")
    out = result.artifacts[".nc"].item

    assert out.dims == ("spaces",)
    assert_allclose(out.values, arr.mean(dim="times").values)


def test_apply_1d_raises_on_missing_dimension() -> None:
    arr = _make_dataarray()

    with pytest.raises(ValueError):
        apply_1d(arr, dim="frequency", pure_function=np.mean)


def test_apply_1d_invalid_mode() -> None:
    arr = _make_dataarray()

    with pytest.raises(ValueError):
        apply_1d(arr, dim="times", pure_function=np.mean, mode="invalid")


def test_apply_1d_raises_on_bad_result_coords_length() -> None:
    arr = _make_dataarray()

    def stats(vector: np.ndarray) -> tuple[float, float]:
        return float(vector.mean()), float(vector.std())

    with pytest.raises(ValueError):
        apply_1d(
            arr,
            dim="times",
            pure_function=stats,
            result_dim="stat",
            result_coords=("mean",),
        )


def test_apply_1d_supports_per_slice_arguments() -> None:
    data = np.arange(2 * 3 * 5, dtype=float).reshape(2, 3, 5)
    arr = xr.DataArray(
        data,
        dims=("epochs", "spaces", "times"),
        coords={
            "epochs": [0, 1],
            "spaces": ["C3", "Cz", "C4"],
            "times": np.linspace(0.0, 1.0, 5),
        },
    )

    offsets = xr.DataArray(
        np.linspace(0.1, 0.6, 6).reshape(2, 3),
        dims=("epochs", "spaces"),
        coords={
            "epochs": arr.coords["epochs"],
            "spaces": arr.coords["spaces"],
        },
    )

    scales = xr.DataArray(
        np.linspace(1.0, 1.5, 3),
        dims=("spaces",),
        coords={"spaces": arr.coords["spaces"]},
    )

    def compute(vector: np.ndarray, offset: float, scale: float) -> float:
        return float(vector.mean() + offset * scale)

    result = apply_1d(
        arr,
        dim="times",
        pure_function=compute,
        args=(offsets,),
        kwargs={"scale": scales},
        mode="iterative",
    )

    assert result.dims == ("epochs", "spaces")

    expected = np.empty((2, 3), dtype=float)
    for epoch_index, epoch in enumerate(arr.coords["epochs"].values):
        for space_index, space in enumerate(arr.coords["spaces"].values):
            mean_val = data[epoch_index, space_index].mean()
            offset_val = offsets.sel(epochs=epoch, spaces=space).item()
            scale_val = scales.sel(spaces=space).item()
            expected[epoch_index, space_index] = mean_val + offset_val * scale_val

    assert_allclose(result.values, expected)

    metadata = json.loads(result.attrs["metadata"])
    per_slice = metadata.get("per_slice_arguments", {})
    assert per_slice
    assert per_slice["args"][0]["name"] == "arg_0"
    assert per_slice["kwargs"]["scale"]["name"] == "scale"


def test_apply_1d_per_slice_arguments_require_iterative_mode() -> None:
    data = np.arange(2 * 3 * 4, dtype=float).reshape(2, 3, 4)
    arr = xr.DataArray(
        data,
        dims=("epochs", "spaces", "times"),
        coords={
            "epochs": [0, 1],
            "spaces": ["C3", "Cz", "C4"],
            "times": np.linspace(0.0, 1.0, 4),
        },
    )

    offsets = xr.DataArray(
        np.zeros((2, 3)),
        dims=("epochs", "spaces"),
        coords={
            "epochs": arr.coords["epochs"],
            "spaces": arr.coords["spaces"],
        },
    )

    def add_offset(vector: np.ndarray, offset: float) -> float:
        return float(vector.mean() + offset)

    with pytest.raises(ValueError, match="Per-slice arguments require mode='iterative'"):
        apply_1d(
            arr,
            dim="times",
            pure_function=add_offset,
            args=(offsets,),
            mode="vectorized",
        )


def test_apply_1d_per_slice_argument_rejects_iteration_dimension() -> None:
    data = np.arange(2 * 3 * 4, dtype=float).reshape(2, 3, 4)
    arr = xr.DataArray(
        data,
        dims=("epochs", "spaces", "times"),
        coords={
            "epochs": [0, 1],
            "spaces": ["C3", "Cz", "C4"],
            "times": np.linspace(0.0, 1.0, 4),
        },
    )

    bad = xr.DataArray(np.arange(arr.sizes["times"]), dims=("times",))

    def add_offset(vector: np.ndarray, offset: float) -> float:
        return float(vector.mean() + offset)

    with pytest.raises(ValueError, match="times"):
        apply_1d(
            arr,
            dim="times",
            pure_function=add_offset,
            args=(bad,),
            mode="iterative",
        )

if __name__ == "__main__":
#    pytest.main([__file__])
    pytest.main(["-v", "-s", "-q", "--no-cov", "--pdb", __file__])
