"""Check local-gate tier routing with command recorders, not real gate results."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from _pytest.mark.expression import Expression

CI_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "ci-local.sh"


@pytest.fixture
def gate_runner(tmp_path: Path):
    binary_dir = tmp_path / "bin"
    binary_dir.mkdir()
    trace = tmp_path / "commands.jsonl"
    recorder = (
        f"#!{sys.executable}\n"
        + """\
import json
import os
from pathlib import Path
import sys

with Path(os.environ["GATE_TRACE"]).open("a") as output:
    output.write(json.dumps({"program": Path(sys.argv[0]).name,
                             "args": sys.argv[1:],
                             "real": os.environ.get("SLOWER_WHISPER_TEST_REAL")}) + "\\n")
if os.environ.get("FAIL_UNIT") and sys.argv[1:3] == ["run", "pytest"]:
    sys.exit(17)
"""
    )
    for name in ("uv", "nix", "nix-clean"):
        executable = binary_dir / name
        executable.write_text(recorder, encoding="utf-8")
        executable.chmod(0o755)
    (tmp_path / ".venv").mkdir()
    snippets = tmp_path / "docs" / "_snippets"
    snippets.mkdir(parents=True)
    for name in ("schema_example.json", "schema_full_example.json"):
        (snippets / name).write_text("{}\n", encoding="utf-8")

    def run(mode: str, *, fail_unit: bool = False):
        env = dict(os.environ)
        env.update(PATH=str(binary_dir) + os.pathsep + env["PATH"], GATE_TRACE=str(trace))
        env.pop("SLOWER_WHISPER_TEST_REAL", None)
        if fail_unit:
            env["FAIL_UNIT"] = "1"
        result = subprocess.run(
            ["bash", str(CI_SCRIPT), mode],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )
        calls = [json.loads(line) for line in trace.read_text().splitlines()]
        return result, calls

    return run


@pytest.mark.parametrize("mode", ["fast", "full"])
def test_gate_keeps_unit_and_real_smoke_phases_separate(gate_runner, mode: str):
    result, calls = gate_runner(mode)
    assert result.returncode == 0, result.stdout + result.stderr
    tests = [row for row in calls if row["args"][:2] == ["run", "pytest"]]
    unit = tests[0]
    selector = Expression.compile(unit["args"][unit["args"].index("-m") + 1])
    assert selector.evaluate(lambda name: name == "unit")
    for tier in ("slow", "heavy", "smoke", "e2e"):
        assert not selector.evaluate(lambda name, selected=tier: name == selected), tier
    if mode == "fast":
        assert len(tests) == 1
    else:
        assert len(tests) == 2
        smoke = tests[1]
        assert smoke["args"][smoke["args"].index("-m") + 1] == "smoke"
        assert smoke["real"] == "1"
    assert any(
        row["args"] == ["run", "python", "scripts/verify_all.py", "--quick"] for row in calls
    )


def test_gate_stops_on_unit_failure_before_smoke_or_verification(gate_runner):
    result, calls = gate_runner("full", fail_unit=True)
    assert result.returncode != 0
    assert len([row for row in calls if row["args"][:2] == ["run", "pytest"]]) == 1
    assert not any("scripts/verify_all.py" in row["args"] for row in calls)
    assert "All local CI checks passed" not in result.stdout
