"""Security tests for dogfood CLI subprocess invocations."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from slower_whisper.pipeline import dogfood


class TestSubprocessArgumentInjection:
    """Verify subprocess calls use '--' separator to prevent argument injection.

    When passing user-controlled paths to subprocess, filenames starting with '-'
    could be interpreted as flags by argparse-based scripts. The '--' separator
    tells argparse that everything after it is a positional argument.
    """

    @patch("subprocess.run")
    @patch("slower_whisper.pipeline.dogfood.print_diarization_stats")
    @patch("slower_whisper.pipeline.dogfood.compute_diarization_stats")
    @patch("slower_whisper.pipeline.dogfood.Path.exists", return_value=True)
    @patch("os.getenv", return_value="fake_api_key")
    def test_llm_subprocess_uses_separator(
        self,
        mock_getenv: MagicMock,
        mock_exists: MagicMock,
        mock_stats: MagicMock,
        mock_print_stats: MagicMock,
        mock_run: MagicMock,
    ) -> None:
        """subprocess.run for LLM integration includes '--' before the JSON path."""
        mock_run.return_value = MagicMock(stdout="LLM Output", returncode=0)

        # Run with a filename that starts with '-' to test the edge case
        dogfood.main(["--file=-test.wav", "--skip-transcribe", "--root", "."])

        # Find the subprocess call to the LLM script
        llm_calls = [
            call
            for call in mock_run.call_args_list
            if call.args and "summarize_with_diarization.py" in call.args[0][1]
        ]
        assert llm_calls, "LLM integration script was not invoked"

        # Verify the command structure includes '--' separator
        cmd = llm_calls[0][0][0]  # First positional arg of first matching call
        assert cmd[2] == "--", (
            f"Missing '--' separator before filename argument. Command was: {cmd}"
        )
        assert cmd[3].endswith("-test.json"), f"Expected json path, got: {cmd[3]}"
