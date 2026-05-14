"""Tests for the visualization entry point."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import xarray as xr

from neurodags.visualization import build_visualization_app, load_visualization_dataset, main


def test_load_visualization_dataset_rejects_unknown_suffix(tmp_path):
    bad = tmp_path / "bad.txt"
    bad.write_text("x")
    with pytest.raises(ValueError, match="Expected \\.fif or \\.nc"):
        load_visualization_dataset(bad)


def test_build_visualization_app_returns_dash_app():
    data = xr.DataArray([[1.0, 2.0]], dims=("channel", "freq"))
    app = build_visualization_app(data, "demo.nc")
    assert app is not None


def test_visualization_main_runs_dash(tmp_path):
    nc = tmp_path / "demo.nc"
    xr.DataArray([1.0, 2.0], dims=("freq",)).to_netcdf(nc)
    with patch("dash.Dash.run") as run:
        assert main([str(nc), "--port", "9999"]) == 0
        run.assert_called_once()
