"""The shipped examples must keep working (run in subprocesses to keep the registry clean)."""

import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


@pytest.mark.parametrize(
    "script,args",
    [
        ("custom_lifter.py", []),
        ("quickstart.py", ["--lifter", "lift_splat", "--steps", "2"]),
        ("tpvformer_occupancy.py", ["--steps", "2"]),
    ],
)
def test_example_runs(script: str, args: list[str]) -> None:
    result = subprocess.run(
        [sys.executable, str(EXAMPLES / script), *args], capture_output=True, text=True, timeout=600
    )
    assert result.returncode == 0, result.stderr
