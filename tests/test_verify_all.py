"""Tests for the verification CLI.

These tests verify the verification script itself runs correctly
(meta, but important for ensuring the contract enforcement tooling works).
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    pass

ROOT = Path(__file__).parent.parent


def test_verify_help():
    """Test that --help flag works."""
    result = subprocess.run(
        ["uv", "run", "slower-whisper-verify", "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "verification" in result.stdout.lower()
    assert "--quick" in result.stdout


def test_import_verify_all():
    """Test that verify_all module can be imported."""
    from scripts.verify_all import main

    assert callable(main)


@pytest.mark.slow
def test_verify_quick_dry_run(monkeypatch: pytest.MonkeyPatch):
    """Test that --quick mode runs without errors (dry run with mocked subprocess)."""
    from scripts import verify_all

    # Track which commands would be run
    commands_run: list[list[str]] = []

    def mock_run(
        cmd: list[str],
        *,
        cwd: Path | None = None,
        allow_failure: bool = False,
    ) -> int:
        """Mock subprocess runner that just records commands."""
        commands_run.append(cmd)
        # Simulate success
        return 0

    # Mock the run function
    monkeypatch.setattr(verify_all, "run", mock_run)

    # Run verification in quick mode
    result = verify_all.main(["--quick"])

    # Should succeed
    assert result == 0

    # Should have run ruff, pytest (fast), and BDD scenarios
    assert len(commands_run) >= 3

    # Verify expected commands
    ruff_cmds = [c for c in commands_run if "ruff" in c]
    pytest_cmds = [c for c in commands_run if "pytest" in c]

    assert len(ruff_cmds) >= 1
    assert len(pytest_cmds) >= 2  # fast tests + BDD


@pytest.mark.slow
def test_verify_all_components_exist():
    """Test that all individual verification functions exist and are callable."""
    from scripts.verify_all import (
        check_ruff,
        docker_smoke,
        run_tests_fast,
        validate_k8s,
        verify_bdd,
    )

    # All components should be callable
    assert callable(check_ruff)
    assert callable(run_tests_fast)
    assert callable(verify_bdd)
    assert callable(docker_smoke)
    assert callable(validate_k8s)


def test_verify_cli_as_module():
    """Test that the script can be run as a Python module."""
    result = subprocess.run(
        ["uv", "run", "python", "-m", "scripts.verify_all", "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    # Should work as module
    assert result.returncode == 0 or "usage:" in result.stdout.lower()


def test_verify_cli_direct_script():
    """Test that the script can be run directly."""
    result = subprocess.run(
        ["uv", "run", "python", "scripts/verify_all.py", "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "usage:" in result.stdout.lower()
