"""Source policy for production ASR truth."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_ROOTS = (ROOT / "transcription", ROOT / "slower_whisper")
STREAMING_PROTOCOL_DEBT = ROOT / "transcription" / "streaming_ws.py"


def production_python_files() -> list[Path]:
    files: list[Path] = []
    for root in PRODUCTION_ROOTS:
        assert root.is_dir(), f"Production root is missing: {root}"
        files.extend(path for path in root.rglob("*.py") if "__pycache__" not in path.parts)
    assert files, "No production Python files were scanned"
    return sorted(files)


def test_production_asr_has_no_dummy_backend_or_dummy_transcript_literal() -> None:
    forbidden = ("DummyWhisperModel", "dummy segment")
    violations: list[str] = []
    for path in production_python_files():
        source = path.read_text(encoding="utf-8")
        for literal in forbidden:
            if literal in source:
                violations.append(f"{path.relative_to(ROOT)}: {literal}")
    assert violations == []


def test_streaming_placeholders_are_confined_to_tracked_protocol_debt() -> None:
    """Keep legacy streaming placeholders from spreading before #84 replaces them."""
    assert STREAMING_PROTOCOL_DEBT.is_file(), (
        f"Tracked protocol-debt file is missing: {STREAMING_PROTOCOL_DEBT}"
    )
    placeholders = ("[processing...]", "[final segment]")
    violations: list[str] = []
    for path in production_python_files():
        if path == STREAMING_PROTOCOL_DEBT:
            continue
        source = path.read_text(encoding="utf-8")
        for literal in placeholders:
            if literal in source:
                violations.append(f"{path.relative_to(ROOT)}: {literal}")
    assert violations == []
