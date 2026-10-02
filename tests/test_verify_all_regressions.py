"""Regression checks for repository-owned verification boundaries.

Docker subprocesses are intercepted deliberately: these tests verify command
construction and failure propagation, not a successful Docker image build.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from scripts import verify_all


def test_docker_uses_relocated_files_and_repository_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repository"
    config = root / "config"
    config.mkdir(parents=True)
    for name in ("Dockerfile", "Dockerfile.api"):
        (config / name).write_text("FROM scratch\n", encoding="utf-8")
    elsewhere = tmp_path / "unrelated-working-directory"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    monkeypatch.setattr(verify_all, "ROOT", root)
    monkeypatch.setattr(verify_all.shutil, "which", lambda _: "/usr/bin/docker")
    calls: list[list[str]] = []

    def fake_process(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        assert Path(kwargs["cwd"]) == root
        if cmd[1] == "build":
            dockerfile = root / cmd[cmd.index("-f") + 1]
            assert dockerfile.is_file(), f"Dockerfile does not exist: {dockerfile}"
            assert (root / cmd[-1]).resolve() == root
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(verify_all.subprocess, "run", fake_process)
    verify_all.docker_smoke()
    assert [cmd[1] for cmd in calls] == ["build", "run", "build"]
    assert [cmd[cmd.index("-f") + 1] for cmd in calls if cmd[1] == "build"] == [
        "config/Dockerfile",
        "config/Dockerfile.api",
    ]


def test_docker_unavailable_does_not_attempt_subprocess(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(verify_all.shutil, "which", lambda _: None)
    monkeypatch.setattr(verify_all, "run", lambda *a, **kw: pytest.fail("Docker is unavailable"))
    verify_all.docker_smoke()
    assert "skipping Docker smoke tests" in capsys.readouterr().out


def test_docker_build_failure_is_not_reported_as_success(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(verify_all.shutil, "which", lambda _: "/usr/bin/docker")
    calls: list[list[str]] = []

    def fail_build(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 17)

    monkeypatch.setattr(verify_all.subprocess, "run", fail_build)
    with pytest.raises(SystemExit) as exc:
        verify_all.docker_smoke()
    assert exc.value.code == 17
    assert len(calls) == 1
    assert "CPU image built" not in capsys.readouterr().out


@pytest.mark.parametrize("failure", [ModuleNotFoundError("pyannote"), ValueError("no spec")])
def test_optional_diarization_missing_parent_or_spec_is_a_skip(
    failure: Exception,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def unavailable(name: str) -> None:
        assert name == "pyannote.audio"
        raise failure

    monkeypatch.setattr(verify_all.importlib.util, "find_spec", unavailable)
    monkeypatch.setenv("HF_TOKEN", "test-not-a-real-token")
    monkeypatch.setattr(
        verify_all, "run", lambda *a, **kw: pytest.fail("No real backend is available")
    )
    verify_all.eval_diarization_real()
    assert "pyannote.audio not installed; skipping" in capsys.readouterr().out


def test_optional_diarization_no_backend_is_a_skip(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(verify_all.importlib.util, "find_spec", lambda _: None)
    monkeypatch.setenv("HF_TOKEN", "test-not-a-real-token")
    verify_all.eval_diarization_real()
    assert "pyannote.audio not installed; skipping" in capsys.readouterr().out


def test_optional_diarization_no_token_is_a_skip(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # A spec signals discoverability, without importing or running an ML backend.
    monkeypatch.setattr(
        verify_all.importlib.util,
        "find_spec",
        lambda _: importlib.util.spec_from_loader("pyannote.audio", loader=None),
    )
    monkeypatch.delenv("HF_TOKEN", raising=False)
    verify_all.eval_diarization_real()
    assert "HF_TOKEN not set; skipping" in capsys.readouterr().out


def test_help_runs_without_repo_working_directory(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(Path(verify_all.__file__).resolve()), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "--quick" in result.stdout
    assert "--eval-diarization" in result.stdout


def test_contradictory_api_flags_are_rejected() -> None:
    with pytest.raises(SystemExit) as exc:
        verify_all.main(["--api", "--skip-api", "--skip-sync"])
    assert exc.value.code == 2


@pytest.mark.parametrize("failure", [RuntimeError("unexpected probe error"), OSError("probe I/O")])
def test_unexpected_dependency_probe_failure_is_not_hidden(
    failure: Exception, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unexpected(name: str) -> None:
        assert name == "pyannote.audio"
        raise failure

    monkeypatch.setattr(verify_all.importlib.util, "find_spec", unexpected)
    with pytest.raises(type(failure), match=str(failure)):
        verify_all.eval_diarization_real()
