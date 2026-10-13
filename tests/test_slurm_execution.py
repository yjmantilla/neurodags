"""Execute the scripts emitted by ``neurodags slurm-script`` against real outputs.

Text checks cannot tell a selection that is honoured from one that only appears in a
comment, so these tests run the generated scripts unchanged with ``bash``. Small shims on
``PATH`` stand in for the cluster: ``sbatch`` runs every array task in turn (in submission
order, which respects ``--dependency=afterok``), ``neurodags`` and ``python`` run this
checkout, and ``activate`` is an empty file for ``source activate myenv``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

import neurodags
from neurodags.cli import main

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")

_SBATCH = """\
import os, re, subprocess, sys
args = sys.argv[1:]
script = args[-1]
env = dict(os.environ)
lo = hi = 0
for arg in args[:-1]:
    if arg.startswith("--array="):
        lo, hi = (int(x) for x in arg.split("=", 1)[1].split("-"))
    elif arg.startswith("--export="):
        key, value = arg.split("=", 1)[1].split("=", 1)
        env[key] = value
for task in range(lo, hi + 1):
    env["SLURM_ARRAY_TASK_ID"] = str(task)
    rc = subprocess.run(["bash", script], env=env).returncode
    if rc:
        sys.exit(rc)
if "--parsable" in args:
    print("1234")
"""


def _shims(bin_dir: Path) -> dict[str, str]:
    bin_dir.mkdir()
    (bin_dir / "activate").write_text("")
    for name, body in {
        "python": f'#!/bin/sh\nexec "{sys.executable}" "$@"\n',
        "neurodags": f'#!/bin/sh\nexec "{sys.executable}" -m neurodags.cli "$@"\n',
        "sbatch": f'#!/bin/sh\nexec "{sys.executable}" "{bin_dir / "sbatch.py"}" "$@"\n',
    }.items():
        (bin_dir / name).write_text(body)
        (bin_dir / name).chmod(0o755)
    (bin_dir / "sbatch.py").write_text(_SBATCH)
    src = str(Path(neurodags.__file__).resolve().parent.parent)
    return {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}",
        "PYTHONPATH": os.pathsep.join(filter(None, [src, os.environ.get("PYTHONPATH")])),
        "LOG_LEVEL": "ERROR",
    }


@pytest.fixture
def two_datasets(tmp_path: Path, monkeypatch) -> Path:
    """A pipeline on dataset ``a`` and a ``-d`` override on ``b`` with the same key and rel.

    A frozen list cannot tell them apart (same dataset key and ``rel``), so only the
    dataset the worker actually loads decides which tree gets written.
    """
    for name in ("a", "b"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "x.txt").write_text(name)

    def ds(name: str) -> dict:
        return {
            "d": {
                "name": "d",
                "source_root": str(tmp_path / name),
                "file_pattern": str(tmp_path / name / "*.txt"),
                "derivatives_path": str(tmp_path / (name + "-out")),
            }
        }

    defs = {
        n: {
            "nodes": [
                {"id": 0, "derivative": "SourceFile"},
                {"id": 1, "node": "dummy", "args": {"param1": "id.0"}},
            ]
        }
        for n in ("A", "B")
    }
    (tmp_path / "pipeline.yml").write_text(
        yaml.safe_dump(
            {
                "naming_version": 1,
                "datasets": ds("a"),
                "DerivativeDefinitions": defs,
                "DerivativeList": ["A", "B"],
            }
        )
    )
    (tmp_path / "override.yml").write_text(yaml.safe_dump(ds("b")))
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _artifacts(root: Path) -> list[str]:
    return sorted(str(f.relative_to(root)) for f in root.rglob("*.message.txt"))


def _bash(script: Path, env: dict, cwd: Path) -> subprocess.CompletedProcess:
    result = subprocess.run(["bash", str(script)], env=env, cwd=cwd, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def test_per_file_worker_honours_datasets_override_and_derivative(two_datasets: Path):
    p = two_datasets
    env = _shims(p / "bin")
    assert (
        main(
            [
                "--log-level",
                "ERROR",
                "slurm-script",
                "pipeline.yml",
                "-d",
                "override.yml",
                "--derivative",
                "B",
                "--output",
                "run_array.sh",
            ]
        )
        == 0
    )
    # The submission lines printed in the header freeze the override's inputs.
    header = [
        line[4:]
        for line in (p / "run_array.sh").read_text().splitlines()
        if line.startswith("#   ")
    ]
    (p / "submit.sh").write_text("set -e\n" + "\n".join(header) + "\n")
    _bash(p / "submit.sh", env, p)
    assert str(p / "b" / "x.txt") not in (p / "inputs.tsv").read_text()  # stores rel only
    assert _artifacts(p) == ["b-out/x.txt@B.message.txt"]


def test_flat_tasks_honour_datasets_override(two_datasets: Path):
    p = two_datasets
    env = _shims(p / "bin")
    argv = ["slurm-script", "pipeline.yml", "-d", "override.yml", "--derivative", "B"]
    assert main(["--log-level", "ERROR", *argv, "--pattern", "flat", "--output", "flat.sh"]) == 0
    header = [
        line[4:] for line in (p / "flat.sh").read_text().splitlines() if line.startswith("#   ")
    ]
    (p / "submit.sh").write_text("set -e\n" + "\n".join(header) + "\n")
    # The submit lines name the script run_array_per_deriv.sh.
    shutil.copy(p / "flat.sh", p / "run_array_per_deriv.sh")
    _bash(p / "submit.sh", env, p)
    assert _artifacts(p) == ["b-out/x.txt@B.message.txt"]


def test_chained_submission_honours_datasets_override(two_datasets: Path):
    p = two_datasets
    env = _shims(p / "bin")
    assert (
        main(
            [
                "--log-level",
                "ERROR",
                "slurm-script",
                "pipeline.yml",
                "-d",
                "override.yml",
                "--pattern",
                "chained",
                "--output",
                "submit_pipeline.sh",
            ]
        )
        == 0
    )
    _bash(p / "submit_pipeline.sh", env, p)
    assert _artifacts(p) == ["b-out/x.txt@A.message.txt", "b-out/x.txt@B.message.txt"]


def test_unquoted_heredoc_keeps_shell_metacharacters_in_datasets_path(two_datasets: Path):
    # flat/chained bodies are unquoted heredocs: backquotes, quotes and backslashes in the
    # path must reach Python unchanged (``$VARS`` are already expanded by -d itself).
    p = two_datasets
    odd = p / "odd `x` 'q' \\ dir"
    odd.mkdir()
    shutil.copy(p / "override.yml", odd / "override.yml")
    env = _shims(p / "bin")
    assert (
        main(
            [
                "--log-level",
                "ERROR",
                "slurm-script",
                "pipeline.yml",
                "-d",
                str(odd / "override.yml"),
                "--pattern",
                "chained",
                "--output",
                "submit_pipeline.sh",
            ]
        )
        == 0
    )
    _bash(p / "submit_pipeline.sh", env, p)
    assert _artifacts(p) == ["b-out/x.txt@A.message.txt", "b-out/x.txt@B.message.txt"]
