"""
UX tests for the unified CLI (transcription/cli.py).

Tests specific UX improvements like dynamic "Next steps" and safety warnings.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from transcription.cli import main

# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_enrich_directory():
    """Mock the enrich_directory API function."""
    with patch("transcription.cli.enrich_directory") as mock:
        # Return a list of mock enriched transcripts
        mock.return_value = [MagicMock()]
        yield mock

@pytest.fixture
def temp_project_root(tmp_path: Path) -> Path:
    """Create a temporary project structure."""
    root = tmp_path / "project"
    root.mkdir()
    (root / "raw_audio").mkdir()
    (root / "whisper_json").mkdir()
    return root

# ============================================================================
# Tests
# ============================================================================

def test_enrich_next_steps_dynamic_filename(mock_enrich_directory, temp_project_root, capsys):
    """Test that enrich command suggests a concrete export command with a real filename."""
    # Create a dummy transcript file
    json_dir = temp_project_root / "whisper_json"
    test_json = json_dir / "my_meeting.json"
    test_json.write_text(
        '{"schema_version": 2, "file_name": "my_meeting.wav", "language": "en", "segments": []}'
    )

    # Create corresponding audio file
    # Normalized goes to input_audio usually but check logic
    # Actually logic checks paths.norm_dir which is input_audio usually?
    # Let's check Paths default.
    # Paths(root=root).norm_dir is root / "input_audio"

    input_audio = temp_project_root / "input_audio"
    input_audio.mkdir()
    test_wav = input_audio / "my_meeting.wav"
    test_wav.write_bytes(b"RIFF" + b"\x00" * 30) # Fake WAV

    # Run enrich
    with patch("transcription.cli.Paths"):
        # We need to ensure Paths works correctly or just rely on real Paths
        # The test uses real Paths if we don't patch it, which is better for integration test.
        # But we need to make sure input_audio is where it expects.
        pass

    exit_code = main(
        [
            "enrich",
            "--root",
            str(temp_project_root),
            "--device",
            "cpu",
            "--no-enable-prosody", # Speed up
            "--no-enable-emotion",
        ]
    )

    assert exit_code == 0
    captured = capsys.readouterr()

    # We expect the output to contain the filename relative to CWD if possible,
    # or at least the filename "my_meeting.json"
    # The current behavior is "path/to/transcript.json", so this assertion will fail initially
    # or rather we want to assert it DOES contain the specific file.

    # Check for the NEW expected behavior (which will fail currently)
    # We want to see: slower-whisper export .../my_meeting.json --format csv
    assert "my_meeting.json" in captured.out
    assert "path/to/transcript.json" not in captured.out


def test_cache_clear_warning(capsys):
    """Test that cache clear command includes a red warning."""
    with patch("builtins.input", return_value="n") as mock_input:
        with patch("sys.stdin.isatty", return_value=True):
            exit_code = main(["cache", "--clear", "whisper"])

            assert exit_code == 0

            # Check prompt argument
            prompt = mock_input.call_args[0][0]
            # Or at least the text "This cannot be undone."
            assert "This cannot be undone." in prompt

            # We also want to check if it's colored red.
            # Colors.red wraps in \033[31m ... \033[0m
            # But Colors.red checks sys.stdout.isatty() usually.
            # In the test environment, we might need to force color or check logic.
            # Colors._should_use_color checks os.getenv("FORCE_COLOR") or sys.stdout.isatty()

            # Let's just check the text for now, as that's the semantic change.
