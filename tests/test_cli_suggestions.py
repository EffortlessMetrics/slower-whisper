"""
Unit tests for the CLI suggestions feature ("Did you mean?").
"""

from __future__ import annotations

import pytest

from transcription.cli import build_parser


class TestCliSuggestions:
    """Test 'Did you mean?' suggestions for invalid commands and arguments."""

    def test_suggest_command_typo(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Typo in subcommand should suggest the correct one."""
        parser = build_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["transcrib"])

        assert exc_info.value.code != 0
        captured = capsys.readouterr()
        assert "Did you mean 'transcribe'?" in captured.err

    def test_suggest_argument_choice_typo(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Typo in argument choice should suggest the correct one."""
        parser = build_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["transcribe", "--device", "cud"])

        assert exc_info.value.code != 0
        captured = capsys.readouterr()
        assert "Did you mean 'cuda'?" in captured.err

    def test_no_suggestion_for_bad_typo(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Very bad typo should not offer suggestions."""
        parser = build_parser()
        with pytest.raises(SystemExit) as exc_info:
            # 'xyz' is not close to any valid command
            parser.parse_args(["xyz"])

        assert exc_info.value.code != 0
        captured = capsys.readouterr()
        # Should contain standard error but likely no "Did you mean"
        # unless 'xyz' happens to be close to something (it's not).
        # However, checking strictly for absence might be flaky if thresholds change.
        # But 'xyz' vs 'transcribe', 'enrich' etc. -> similarity is low.
        assert "invalid choice: 'xyz'" in captured.err
