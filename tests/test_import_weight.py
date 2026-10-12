"""Guard against heavy third-party imports at CLI / node-registry import time (#30).

Node discovery imports every module in ``neurodags.nodes`` on every CLI invocation
and in every worker process, so heavy dependencies must be imported lazily, when a
node that needs them actually runs.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

from neurodags.nodes import get_node, list_nodes
from neurodags.orchestrators import iterate_derivative_pipeline

# ``numba`` is deliberately absent: base ``mne`` imports it (cheap, no JIT).
HEAVY_MODULES = ("antropy", "neurokit2", "fooof", "matplotlib", "mne.export", "mne_bids")

LAZY_NODES = (
    "antropy_app_entropy",
    "antropy_hjorth_params",
    "neurokit_entropy_shannon",
    "neurokit_complexity_delay",
    "fooof",
    "fooof_scalars",
)


def _run_python(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
    )


def test_cli_import_does_not_load_heavy_stack():
    result = _run_python(f"""
        import sys
        import neurodags.cli
        heavy = [m for m in {HEAVY_MODULES!r} if m in sys.modules]
        print(",".join(heavy))
        """)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "", f"heavy modules imported by neurodags.cli: {result.stdout}"


def test_lazy_nodes_are_registered():
    nodes = set(list_nodes())
    for name in LAZY_NODES:
        assert name in nodes


def test_node_docstrings_name_wrapped_function():
    assert "antropy.app_entropy" in (get_node("antropy_app_entropy").__doc__ or "")
    assert "neurokit2.entropy_shannon" in (get_node("neurokit_entropy_shannon").__doc__ or "")


def test_missing_dependency_registers_node_and_raises_on_call():
    # Simulate antropy/neurokit2/fooof being uninstalled: the nodes must still be
    # registered and fail with an actionable ImportError only when called.
    result = _run_python("""
        import sys

        BLOCKED = {"antropy", "neurokit2", "fooof"}

        class _Blocker:
            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] in BLOCKED:
                    raise ModuleNotFoundError(f"No module named {name!r}", name=name)
                return None

        sys.meta_path.insert(0, _Blocker())

        import numpy as np
        import xarray as xr
        from neurodags.nodes import get_node, list_nodes

        nodes = set(list_nodes())
        for name in ("antropy_app_entropy", "neurokit_entropy_shannon", "fooof"):
            assert name in nodes, name

        da = xr.DataArray(np.random.default_rng(0).normal(size=(2, 64)), dims=("spaces", "times"))
        for name, pkg in (
            ("antropy_app_entropy", "antropy"),
            ("neurokit_entropy_shannon", "neurokit2"),
        ):
            try:
                get_node(name)(da, dim="times")
            except ImportError as exc:
                assert pkg in str(exc), str(exc)
            else:
                raise AssertionError(f"{name} did not raise ImportError")
        print("ok")
        """)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("ok")


def test_fooof_lazy_import_keeps_warning_filters_and_is_silent():
    # #16: importing fooof (now on first use) must not print its DeprecationWarning
    # nor leave ``warnings.simplefilter('always')`` installed process-wide.
    result = subprocess.run(
        [
            sys.executable,
            "-W",
            "always::DeprecationWarning",
            "-c",
            textwrap.dedent("""
                import warnings
                from neurodags.nodes.spectral import _fooof_cls
                before = list(warnings.filters)
                _fooof_cls()
                assert list(warnings.filters) == before
                """),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "fooof" not in result.stderr.lower()


@pytest.mark.filterwarnings("ignore")
def test_antropy_node_runs_in_parallel_workers(dummy_pipeline):
    # The antropy import now happens inside the node call, so it must also work in
    # fresh loky worker processes.
    cfg = dummy_pipeline["config"]
    out_dir = dummy_pipeline["out_dir"]
    cfg["DerivativeDefinitions"]["ZeroCross"] = {
        "overwrite": False,
        "nodes": [
            {"id": 0, "derivative": "BasicPrep.fif"},
            {
                "id": 1,
                "node": "antropy_num_zerocross",
                "args": {"data_like": "id.0", "dim": "times"},
            },
        ],
    }
    cfg["DerivativeList"].append("ZeroCross")

    iterate_derivative_pipeline(cfg, "ZeroCross", n_jobs=2, raise_on_error=True)

    outputs = list(out_dir.rglob("*@ZeroCross.nc"))
    sources = list(dummy_pipeline["data_dir"].rglob("*.vhdr"))
    assert len(outputs) == len(sources) > 0
