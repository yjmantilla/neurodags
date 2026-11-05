"""Execute the scripts emitted by ``neurodags slurm-script`` against real outputs.

Text checks cannot tell a selection that is honoured from one that only appears in a
comment, so these tests run the generated scripts unchanged with ``bash``. Small shims on
``PATH`` stand in for the cluster: ``sbatch`` runs every array task in turn (in submission
order, which respects ``--dependency=afterok``; ``--export`` follows SLURM: without
``ALL`` it replaces the task environment, with ``ALL`` inherited variables win), ``neurodags`` and ``python`` run this checkout,
and ``activate`` is an empty file for ``source activate myenv``. Both shims log their
arguments so the tests can check what was frozen, counted and submitted.
"""

from __future__ import annotations

import json
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
import json, os, subprocess, sys
args = sys.argv[1:]
# sbatch [options] script [script arguments]
first = next(i for i, a in enumerate(args) if not a.startswith("--"))
options, script, script_args = args[:first], args[first], args[first + 1:]
env = dict(os.environ)
lo = hi = 0
for arg in options:
    if arg.startswith("--array="):
        lo, hi = (int(x) for x in arg.split("=", 1)[1].split("-"))
    elif arg.startswith("--export="):
        # SLURM semantics (sbatch --export): without ALL, the listed variables *replace*
        # the environment (PATH and the activated environment are lost); with ALL, a
        # variable already in the user's environment takes precedence over the listed one.
        items = arg.split("=", 1)[1].split(",")
        if "ALL" not in items:
            env = {}
        for item in items:
            if "=" in item:
                key, value = item.split("=", 1)
                if "ALL" in items:
                    env.setdefault(key, value)
                else:
                    env[key] = value
with open(os.environ["SHIM_LOG"], "a") as log:
    log.write(json.dumps({"sbatch": args, "tasks": list(range(lo, hi + 1)),
                          "script_args": script_args}) + "\\n")
for task in range(lo, hi + 1):
    env["SLURM_ARRAY_TASK_ID"] = str(task)
    rc = subprocess.run(["bash", script, *script_args], env=env).returncode
    if rc:
        sys.exit(rc)
if "--parsable" in args:
    print("1234")
"""

_NEURODAGS = """\
import json, os, sys
with open(os.environ["SHIM_LOG"], "a") as log:
    log.write(json.dumps({"neurodags": sys.argv[1:]}) + "\\n")
from neurodags.cli import main
sys.exit(main(sys.argv[1:]))
"""


def _shims(bin_dir: Path) -> dict[str, str]:
    bin_dir.mkdir()
    (bin_dir / "activate").write_text("")
    (bin_dir / "sbatch.py").write_text(_SBATCH)
    (bin_dir / "neurodags_shim.py").write_text(_NEURODAGS)
    for name, body in {
        "python": f'#!/bin/sh\nexec "{sys.executable}" "$@"\n',
        "neurodags": f'#!/bin/sh\nexec "{sys.executable}" "{bin_dir / "neurodags_shim.py"}" "$@"\n',
        "sbatch": f'#!/bin/sh\nexec "{sys.executable}" "{bin_dir / "sbatch.py"}" "$@"\n',
    }.items():
        (bin_dir / name).write_text(body)
        (bin_dir / name).chmod(0o755)
    src = str(Path(neurodags.__file__).resolve().parent.parent)
    return {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}",
        "PYTHONPATH": os.pathsep.join(filter(None, [src, os.environ.get("PYTHONPATH")])),
        "LOG_LEVEL": "ERROR",
        "SHIM_LOG": str(bin_dir / "calls.jsonl"),
    }


def _calls(env: dict) -> list[dict]:
    path = Path(env["SHIM_LOG"])
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def _dataset(p: Path, key: str, folder: str) -> dict:
    return {
        key: {
            "name": key,
            "source_root": str(p / folder),
            "file_pattern": str(p / folder / "*.txt"),
            "derivatives_path": str(p / (folder + "-out")),
        }
    }


def _definitions(param2: str = "INTENDED") -> dict:
    return {
        n: {
            "nodes": [
                {"id": 0, "derivative": "SourceFile"},
                {"id": 1, "node": "dummy", "args": {"param1": "id.0", "param2": param2}},
            ]
        }
        for n in ("A", "B")
    }


def _write_pipeline(path: Path, datasets: dict, param2: str = "INTENDED") -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "naming_version": 1,
                "datasets": datasets,
                "DerivativeDefinitions": _definitions(param2),
                "DerivativeList": ["A", "B"],
            }
        )
    )


def _files(p: Path, folder: str, names: list[str]) -> None:
    (p / folder).mkdir()
    for name in names:
        (p / folder / name).write_text(name)


@pytest.fixture
def same_key(tmp_path: Path, monkeypatch) -> Path:
    """Pipeline on folder ``a``, ``-d`` override on folder ``b``: same key ``d``, same rel.

    The frozen list and count are identical for both, so only the dataset the *worker*
    loads decides which tree is written.
    """
    _files(tmp_path, "a", ["x.txt"])
    _files(tmp_path, "b", ["x.txt"])
    _write_pipeline(tmp_path / "pipeline.yml", _dataset(tmp_path, "d", "a"))
    (tmp_path / "override.yml").write_text(yaml.safe_dump(_dataset(tmp_path, "d", "b")))
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def other_population(tmp_path: Path, monkeypatch) -> Path:
    """Pipeline on dataset ``d`` (2 files), ``-d`` override on dataset ``e`` (3 files).

    Freezing or counting the wrong dataset gives a different list and array size.
    """
    _files(tmp_path, "a", ["a1.txt", "a2.txt"])
    _files(tmp_path, "b", ["z1.txt", "z2.txt", "z3.txt"])
    _write_pipeline(tmp_path / "pipeline.yml", _dataset(tmp_path, "d", "a"))
    (tmp_path / "override.yml").write_text(yaml.safe_dump(_dataset(tmp_path, "e", "b")))
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _artifacts(root: Path) -> list[str]:
    return sorted(str(f.relative_to(root)) for f in root.rglob("*.message.txt"))


def _bash(script: Path, env: dict, cwd: Path) -> None:
    result = subprocess.run(["bash", str(script)], env=env, cwd=cwd, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def _generate(*argv: str) -> None:
    assert main(["--log-level", "ERROR", "slurm-script", *argv]) == 0


def _submit_from_header(p: Path, script: str, env: dict) -> None:
    """Run the submission lines printed in a per-file/flat template's header comment."""
    header = [line[4:] for line in (p / script).read_text().splitlines() if line.startswith("#   ")]
    (p / "submit.sh").write_text("set -e\n" + "\n".join(header) + "\n")
    _bash(p / "submit.sh", env, p)


def _frozen(p: Path) -> list[list[str]]:
    lines = (p / "inputs.tsv").read_text().splitlines()
    return [line.split("\t") for line in lines if not line.startswith("#")]


# ---------------------------------------------------------------------------
# Worker selection (same key and rel in both datasets)
# ---------------------------------------------------------------------------


def test_per_file_worker_honours_datasets_override_and_derivative(same_key: Path):
    p = same_key
    env = _shims(p / "bin")
    _generate("pipeline.yml", "-d", "override.yml", "--derivative", "B", "--output", "run_array.sh")
    _submit_from_header(p, "run_array.sh", env)
    assert _artifacts(p) == ["b-out/x.txt@B.message.txt"]


def test_flat_tasks_honour_datasets_override(same_key: Path):
    p = same_key
    env = _shims(p / "bin")
    _generate("pipeline.yml", "-d", "override.yml", "--derivative", "B", "--pattern", "flat",
              "--output", "run_array_per_deriv.sh")  # fmt: skip
    _submit_from_header(p, "run_array_per_deriv.sh", env)
    assert _artifacts(p) == ["b-out/x.txt@B.message.txt"]


def test_chained_submission_honours_datasets_override(same_key: Path):
    p = same_key
    env = _shims(p / "bin")
    _generate("pipeline.yml", "-d", "override.yml", "--pattern", "chained",
              "--output", "submit_pipeline.sh")  # fmt: skip
    _bash(p / "submit_pipeline.sh", env, p)
    assert _artifacts(p) == ["b-out/x.txt@A.message.txt", "b-out/x.txt@B.message.txt"]


def test_chained_sbatch_keeps_the_job_environment(same_key: Path):
    # --export=DERIVATIVE=X alone would replace the environment (PATH, modules, venv);
    # the derivative is the worker's argument instead.
    p = same_key
    env = _shims(p / "bin")
    _generate("pipeline.yml", "-d", "override.yml", "--pattern", "chained",
              "--output", "submit_pipeline.sh")  # fmt: skip
    _bash(p / "submit_pipeline.sh", env, p)
    sbatch = [c["sbatch"] for c in _calls(env) if "sbatch" in c]
    assert [[a for a in argv if a.startswith("--export=")] for argv in sbatch] == [
        ["--export=ALL"],
        ["--export=ALL"],
    ]
    assert [argv[-2:] for argv in sbatch] == [
        ["run_one_derivative.sh", "A"],
        ["run_one_derivative.sh", "B"],
    ]


def test_chained_selection_ignores_an_inherited_derivative_variable(same_key: Path):
    # With --export=ALL an inherited DERIVATIVE would win over --export=ALL,DERIVATIVE=B.
    p = same_key
    env = {**_shims(p / "bin"), "DERIVATIVE": "A"}
    _generate("pipeline.yml", "-d", "override.yml", "--derivative", "B", "--pattern", "chained",
              "--output", "submit_pipeline.sh")  # fmt: skip
    _bash(p / "submit_pipeline.sh", env, p)
    assert _artifacts(p) == ["b-out/x.txt@B.message.txt"]


def test_chained_submission_honours_derivative_selection(same_key: Path):
    p = same_key
    env = _shims(p / "bin")
    _generate("pipeline.yml", "-d", "override.yml", "--derivative", "B", "--pattern", "chained",
              "--output", "submit_pipeline.sh")  # fmt: skip
    _bash(p / "submit_pipeline.sh", env, p)
    assert [c["script_args"] for c in _calls(env) if "sbatch" in c] == [["B"]]
    assert _artifacts(p) == ["b-out/x.txt@B.message.txt"]


# ---------------------------------------------------------------------------
# Submission selection (override has another dataset key and population)
# ---------------------------------------------------------------------------

_OVERRIDE_LIST = [["0", "e", "z1.txt"], ["1", "e", "z2.txt"], ["2", "e", "z3.txt"]]
_OVERRIDE_OUT = [f"b-out/z{i}.txt@{d}.message.txt" for i in (1, 2, 3) for d in ("A", "B")]


def _assert_submission_used_override(p: Path, env: dict, tasks: list[int]) -> None:
    assert _frozen(p) == _OVERRIDE_LIST
    override = str(p / "override.yml")
    calls = _calls(env)
    nd = [c["neurodags"] for c in calls if "neurodags" in c]
    assert [a[0] for a in nd] == ["inputs", "count-inputs"]
    for argv in nd:
        assert argv[argv.index("-d") + 1] == override
    assert [t for c in calls if "sbatch" in c for t in c["tasks"]] == tasks
    assert _artifacts(p) == sorted(_OVERRIDE_OUT)


def test_per_file_submission_freezes_and_counts_the_override(other_population: Path):
    p = other_population
    env = _shims(p / "bin")
    _generate("pipeline.yml", "-d", "override.yml", "--output", "run_array.sh")
    _submit_from_header(p, "run_array.sh", env)
    _assert_submission_used_override(p, env, [0, 1, 2])


def test_flat_submission_freezes_and_counts_the_override(other_population: Path):
    p = other_population
    env = _shims(p / "bin")
    _generate("pipeline.yml", "-d", "override.yml", "--pattern", "flat",
              "--output", "run_array_per_deriv.sh")  # fmt: skip
    _submit_from_header(p, "run_array_per_deriv.sh", env)
    _assert_submission_used_override(p, env, list(range(6)))


def test_chained_submission_freezes_and_counts_the_override(other_population: Path):
    p = other_population
    env = _shims(p / "bin")
    _generate("pipeline.yml", "-d", "override.yml", "--pattern", "chained",
              "--output", "submit_pipeline.sh")  # fmt: skip
    _bash(p / "submit_pipeline.sh", env, p)
    _assert_submission_used_override(p, env, [0, 1, 2, 0, 1, 2])


# ---------------------------------------------------------------------------
# Unusual pipeline and datasets paths reach the workers unchanged
# ---------------------------------------------------------------------------

# Spaces, $1 (removed by an unquoted shell), both quotes, backquotes (command
# substitution) and a backslash followed by "n" (a newline escape in Python source).
_HAZARDS = "p a$1 \"q\" 'r' `printf X` \\n"


@pytest.mark.parametrize("pattern", ["per-file", "flat", "chained"])
@pytest.mark.parametrize("name", [_HAZARDS + ".yml", "pipeline$1.yml"])
def test_unusual_pipeline_and_datasets_paths(tmp_path: Path, monkeypatch, pattern, name):
    p = tmp_path
    _files(p, "b", ["x.txt"])
    _files(p, "w", ["x.txt"])
    odd = p / ("ds " + _HAZARDS)
    odd.mkdir()
    (odd / "override.yml").write_text(yaml.safe_dump(_dataset(p, "d", "b")))
    _write_pipeline(p / name, _dataset(p, "d", "w"), param2="INTENDED")
    # Decoys a mangled path would silently load instead ($1 removed / backquotes run).
    for decoy in ("pipeline.yml", "p a \"q\" 'r' X \\n.yml"):
        if decoy != name:
            _write_pipeline(p / decoy, _dataset(p, "d", "w"), param2="WRONG_PIPELINE")
    monkeypatch.chdir(p)
    env = _shims(p / "bin")
    out = {"per-file": "run_array.sh", "flat": "run_array_per_deriv.sh"}.get(
        pattern, "submit_pipeline.sh"
    )
    _generate(name, "-d", str(odd / "override.yml"), "--pattern", pattern, "--output", out)
    if pattern == "chained":
        _bash(p / out, env, p)
    else:
        _submit_from_header(p, out, env)
    assert _artifacts(p) == ["b-out/x.txt@A.message.txt", "b-out/x.txt@B.message.txt"]
    for artifact in _artifacts(p):
        assert "param2=INTENDED" in (p / artifact).read_text()
    for argv in (c["neurodags"] for c in _calls(env) if "neurodags" in c):
        assert argv[1] == name
        assert argv[argv.index("-d") + 1] == str(odd / "override.yml")
