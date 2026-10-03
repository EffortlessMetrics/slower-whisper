"""Package-local build identity and generator determinism."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from transcription import _build_info
from transcription.build_info import (
    get_build_id,
    get_package_build_info,
    get_source_commit,
)
from transcription.receipt import get_git_commit


def test_source_checkout_build_identity_is_unknown_by_default() -> None:
    assert _build_info.SOURCE_COMMIT is None
    assert _build_info.BUILD_ID is None
    assert get_package_build_info().to_dict() == {}
    assert get_source_commit() is None
    assert get_build_id() is None
    assert get_git_commit() is None


def test_runtime_uses_only_validated_package_local_identity(monkeypatch) -> None:
    monkeypatch.setattr(_build_info, "SOURCE_COMMIT", "abcdef123456")
    monkeypatch.setattr(_build_info, "BUILD_ID", "github-12345-1")

    assert get_source_commit() == "abcdef123456"
    assert get_git_commit() == "abcdef123456"
    assert get_build_id() == "github-12345-1"
    assert get_package_build_info().to_dict() == {
        "source_commit": "abcdef123456",
        "build_id": "github-12345-1",
    }

    monkeypatch.setattr(_build_info, "SOURCE_COMMIT", "not-a-revision")
    monkeypatch.setattr(_build_info, "BUILD_ID", "contains a space")
    assert get_package_build_info().to_dict() == {}


def test_fake_git_and_unrelated_repository_cannot_change_identity(
    tmp_path: Path,
    monkeypatch,
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_git = fake_bin / "git"
    fake_git.write_text(
        "#!/bin/sh\necho deadbeefdead\n",
        encoding="utf-8",
    )
    fake_git.chmod(0o755)
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    (unrelated / ".git").mkdir()

    monkeypatch.setattr(_build_info, "SOURCE_COMMIT", "abcdef123456")
    monkeypatch.setenv("PATH", f"{fake_bin}{os.pathsep}{os.environ.get('PATH', '')}")
    monkeypatch.chdir(unrelated)

    assert get_git_commit() == "abcdef123456"


def test_build_info_generator_is_byte_deterministic(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "write_build_info.py"
    first = tmp_path / "first.py"
    second = tmp_path / "second.py"
    command = [
        sys.executable,
        str(script),
        "--source-commit",
        "ABCDEF1234567890ABCDEF1234567890ABCDEF12",
        "--build-id",
        "github-12345-1",
    ]

    subprocess.run([*command, "--output", str(first)], check=True)
    subprocess.run([*command, "--output", str(second)], check=True)

    assert first.read_bytes() == second.read_bytes()
    source = first.read_text(encoding="utf-8")
    assert 'SOURCE_COMMIT: str | None = "abcdef123456"' in source
    assert 'BUILD_ID: str | None = "github-12345-1"' in source


def test_build_info_generator_without_identity_renders_valid_python(
    tmp_path: Path,
) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "write_build_info.py"
    output = tmp_path / "unknown.py"
    subprocess.run(
        [sys.executable, str(script), "--output", str(output)],
        check=True,
    )
    source = output.read_text(encoding="utf-8")
    assert "SOURCE_COMMIT: str | None = None" in source
    assert "BUILD_ID: str | None = None" in source
    compile(source, str(output), "exec")


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--source-commit", "not-a-hash"], "source commit"),
        (["--build-id", "contains a space"], "build ID"),
    ],
)
def test_build_info_generator_rejects_invalid_inputs(
    tmp_path: Path,
    arguments: list[str],
    message: str,
) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "write_build_info.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            *arguments,
            "--output",
            str(tmp_path / "output.py"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert message in result.stderr
