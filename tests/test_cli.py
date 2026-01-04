"""
Unit tests for the slower-whisper CLI (transcription/cli.py).

This module provides comprehensive unit tests for the CLI, complementing
the integration tests in test_cli_integration.py. Focus areas:
- Version flag
- Cache subcommand
- Samples subcommand
- Benchmark subcommand
- Configuration file loading
- Environment variable handling
- Error exit codes
- Help text validation

Tests use mocking to avoid actual transcription/enrichment and focus on
CLI argument parsing and routing.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from transcription import __version__
from transcription.cli import (
    _config_from_enrich_args,
    _config_from_transcribe_args,
    _format_size,
    _get_cache_size,
    build_parser,
    main,
)

# ============================================================================
# Version Flag Tests
# ============================================================================


class TestVersionFlag:
    """Test --version flag behavior."""

    def test_version_flag_shows_version(self, capsys: pytest.CaptureFixture[str]) -> None:
        """The --version flag should display the package version and exit."""
        with pytest.raises(SystemExit) as exc_info:
            main(["--version"])

        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert __version__ in captured.out
        assert "slower-whisper" in captured.out

    def test_version_flag_short_form_not_available(self) -> None:
        """Verify -v is not a short form for --version (argparse default)."""
        parser = build_parser()
        # -v by itself should fail since there's no short version flag
        with pytest.raises(SystemExit):
            parser.parse_args(["-v"])


# ============================================================================
# Cache Subcommand Tests
# ============================================================================


class TestCacheSubcommand:
    """Test cache subcommand parsing and behavior."""

    def test_cache_requires_flag(self) -> None:
        """Cache subcommand requires either --show or --clear."""
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["cache"])

    def test_cache_show_flag(self) -> None:
        """Cache --show flag is parsed correctly."""
        parser = build_parser()
        args = parser.parse_args(["cache", "--show"])

        assert args.command == "cache"
        assert args.show is True
        assert args.clear is None

    def test_cache_clear_all_flag(self) -> None:
        """Cache --clear all is parsed correctly."""
        parser = build_parser()
        args = parser.parse_args(["cache", "--clear", "all"])

        assert args.command == "cache"
        assert args.show is False
        assert args.clear == "all"

    @pytest.mark.parametrize(
        "cache_type",
        ["whisper", "emotion", "diarization", "hf", "torch", "samples"],
    )
    def test_cache_clear_individual_types(self, cache_type: str) -> None:
        """Cache --clear accepts all valid cache types."""
        parser = build_parser()
        args = parser.parse_args(["cache", "--clear", cache_type])

        assert args.clear == cache_type

    def test_cache_clear_invalid_type(self) -> None:
        """Cache --clear rejects invalid cache types."""
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["cache", "--clear", "invalid"])

    def test_cache_mutually_exclusive(self) -> None:
        """Cache --show and --clear are mutually exclusive."""
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["cache", "--show", "--clear", "all"])

    def test_cache_show_displays_info(
        self,
        capsys: pytest.CaptureFixture[str],
        tmp_path: Path,
    ) -> None:
        """Cache --show displays cache information."""
        # Set up mock paths
        mock_paths = MagicMock()
        mock_paths.root = tmp_path / "cache"
        mock_paths.hf_home = tmp_path / "hf"
        mock_paths.torch_home = tmp_path / "torch"
        mock_paths.whisper_root = tmp_path / "whisper"
        mock_paths.emotion_root = tmp_path / "emotion"
        mock_paths.diarization_root = tmp_path / "diarization"
        mock_paths.ensure_dirs.return_value = mock_paths

        # Create some test directories with files
        (tmp_path / "hf").mkdir()
        (tmp_path / "hf" / "test.bin").write_bytes(b"x" * 1024)

        # Patch at the locations where the functions are imported/used
        with (
            patch("transcription.cache.CachePaths.from_env", return_value=mock_paths),
            patch("transcription.samples.get_samples_cache_dir", return_value=tmp_path / "samples"),
        ):
            exit_code = main(["cache", "--show"])

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "slower-whisper cache locations:" in captured.out
        assert "HF_HOME:" in captured.out
        assert "Whisper:" in captured.out
        assert "Total:" in captured.out


# ============================================================================
# Samples Subcommand Tests
# ============================================================================


class TestSamplesSubcommand:
    """Test samples subcommand parsing and behavior."""

    def test_samples_requires_action(self) -> None:
        """Samples subcommand requires an action (list, download, copy, generate)."""
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["samples"])

    def test_samples_list_parsing(self) -> None:
        """Samples list action is parsed correctly."""
        parser = build_parser()
        args = parser.parse_args(["samples", "list"])

        assert args.command == "samples"
        assert args.samples_action == "list"

    def test_samples_download_parsing(self) -> None:
        """Samples download action is parsed correctly."""
        parser = build_parser()
        args = parser.parse_args(["samples", "download", "mini_diarization"])

        assert args.command == "samples"
        assert args.samples_action == "download"
        assert args.dataset == "mini_diarization"
        assert args.force is False

    def test_samples_download_force_flag(self) -> None:
        """Samples download --force flag is parsed correctly."""
        parser = build_parser()
        args = parser.parse_args(["samples", "download", "test_dataset", "--force"])

        assert args.force is True

    def test_samples_copy_parsing(self) -> None:
        """Samples copy action is parsed correctly."""
        parser = build_parser()
        args = parser.parse_args(["samples", "copy", "mini_diarization"])

        assert args.command == "samples"
        assert args.samples_action == "copy"
        assert args.dataset == "mini_diarization"
        assert args.root == Path.cwd()

    def test_samples_copy_with_root(self, tmp_path: Path) -> None:
        """Samples copy with --root is parsed correctly."""
        parser = build_parser()
        args = parser.parse_args(["samples", "copy", "mini_diarization", "--root", str(tmp_path)])

        assert args.root == tmp_path

    def test_samples_generate_parsing(self) -> None:
        """Samples generate action is parsed correctly."""
        parser = build_parser()
        args = parser.parse_args(["samples", "generate"])

        assert args.command == "samples"
        assert args.samples_action == "generate"
        assert args.output is None
        assert args.speakers == 2

    def test_samples_generate_with_options(self, tmp_path: Path) -> None:
        """Samples generate with options is parsed correctly."""
        parser = build_parser()
        args = parser.parse_args(
            ["samples", "generate", "--output", str(tmp_path), "--speakers", "3"]
        )

        assert args.output == tmp_path
        assert args.speakers == 3

    def test_samples_generate_invalid_speakers(self) -> None:
        """Samples generate rejects invalid speaker counts."""
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["samples", "generate", "--speakers", "5"])

    def test_samples_list_displays_datasets(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Samples list displays available datasets."""
        mock_datasets = {
            "test_dataset": {
                "description": "Test dataset for testing",
                "source_url": "https://example.com",
                "license": "MIT",
                "test_files": ["test.wav"],
            }
        }

        # Patch at the module where the function is defined
        with patch("transcription.samples.list_sample_datasets", return_value=mock_datasets):
            exit_code = main(["samples", "list"])

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "Available sample datasets:" in captured.out
        assert "test_dataset" in captured.out


# ============================================================================
# Benchmark Subcommand Tests
# ============================================================================


class TestBenchmarkSubcommand:
    """Test benchmark subcommand parsing and behavior."""

    def test_benchmark_default_shows_list(self) -> None:
        """Benchmark without action defaults to list."""
        parser = build_parser()
        args = parser.parse_args(["benchmark"])

        assert args.command == "benchmark"
        assert args.benchmark_action is None

    def test_benchmark_list_parsing(self) -> None:
        """Benchmark list action is parsed correctly."""
        parser = build_parser()
        args = parser.parse_args(["benchmark", "list"])

        assert args.command == "benchmark"
        assert args.benchmark_action == "list"

    def test_benchmark_status_parsing(self) -> None:
        """Benchmark status action is parsed correctly."""
        parser = build_parser()
        args = parser.parse_args(["benchmark", "status"])

        assert args.command == "benchmark"
        assert args.benchmark_action == "status"

    def test_benchmark_run_parsing(self) -> None:
        """Benchmark run action is parsed correctly."""
        parser = build_parser()
        args = parser.parse_args(["benchmark", "run", "--track", "asr"])

        assert args.command == "benchmark"
        assert args.benchmark_action == "run"
        assert args.track == "asr"
        assert args.dataset is None
        assert args.split == "test"
        assert args.limit is None
        assert args.output is None
        assert args.verbose is False

    def test_benchmark_run_all_options(self, tmp_path: Path) -> None:
        """Benchmark run with all options is parsed correctly."""
        output_file = tmp_path / "results.json"
        parser = build_parser()
        args = parser.parse_args(
            [
                "benchmark",
                "run",
                "--track",
                "diarization",
                "--dataset",
                "ami",
                "--split",
                "dev",
                "--limit",
                "10",
                "--output",
                str(output_file),
                "--verbose",
            ]
        )

        assert args.track == "diarization"
        assert args.dataset == "ami"
        assert args.split == "dev"
        assert args.limit == 10
        assert args.output == output_file
        assert args.verbose is True

    def test_benchmark_run_requires_track(self) -> None:
        """Benchmark run requires --track flag."""
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["benchmark", "run"])

    @pytest.mark.parametrize("track", ["asr", "diarization", "streaming", "semantic", "emotion"])
    def test_benchmark_run_valid_tracks(self, track: str) -> None:
        """Benchmark run accepts all valid tracks."""
        parser = build_parser()
        args = parser.parse_args(["benchmark", "run", "--track", track])

        assert args.track == track

    def test_benchmark_run_invalid_track(self) -> None:
        """Benchmark run rejects invalid tracks."""
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["benchmark", "run", "--track", "invalid"])

    @patch("transcription.cli.handle_benchmark_command")
    def test_benchmark_list_invokes_handler(self, mock_handler: MagicMock) -> None:
        """Benchmark list invokes the benchmark handler."""
        mock_handler.return_value = 0

        exit_code = main(["benchmark", "list"])

        assert exit_code == 0
        mock_handler.assert_called_once()


# ============================================================================
# Configuration File Loading Tests
# ============================================================================


class TestConfigFileLoading:
    """Test configuration file loading via CLI."""

    def test_transcribe_with_config_file(self, tmp_path: Path) -> None:
        """Transcribe --config loads settings from file."""
        config_file = tmp_path / "config.json"
        config_file.write_text(
            json.dumps(
                {
                    "model": "small",
                    "device": "cpu",
                    "language": "en",
                }
            )
        )

        parser = build_parser()
        args = parser.parse_args(["transcribe", "--config", str(config_file)])

        assert args.config == config_file

        # Verify config is loaded correctly
        config = _config_from_transcribe_args(args)
        assert config.model == "small"
        assert config.device == "cpu"
        assert config.language == "en"

    def test_transcribe_cli_overrides_config_file(self, tmp_path: Path) -> None:
        """CLI flags take precedence over config file settings."""
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({"model": "small", "device": "cuda"}))

        parser = build_parser()
        args = parser.parse_args(
            ["transcribe", "--config", str(config_file), "--model", "base", "--device", "cpu"]
        )

        config = _config_from_transcribe_args(args)
        # CLI flags should override config file
        assert config.model == "base"
        assert config.device == "cpu"

    def test_enrich_with_config_file(self, tmp_path: Path) -> None:
        """Enrich --config loads settings from file."""
        config_file = tmp_path / "enrich_config.json"
        config_file.write_text(
            json.dumps(
                {
                    "enable_prosody": False,
                    "enable_emotion": True,
                    "device": "cuda",
                }
            )
        )

        parser = build_parser()
        args = parser.parse_args(["enrich", "--config", str(config_file)])

        assert args.enrich_config == config_file

        config = _config_from_enrich_args(args)
        assert config.enable_prosody is False
        assert config.enable_emotion is True
        assert config.device == "cuda"

    def test_transcribe_nonexistent_config_file_raises_error(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Transcribe with nonexistent config file raises error."""
        nonexistent = tmp_path / "does_not_exist.json"

        # The error should be raised when building the config
        parser = build_parser()
        args = parser.parse_args(["transcribe", "--config", str(nonexistent)])

        with pytest.raises(FileNotFoundError):
            _config_from_transcribe_args(args)

    def test_transcribe_invalid_json_config_raises_error(self, tmp_path: Path) -> None:
        """Transcribe with invalid JSON config file raises error."""
        config_file = tmp_path / "invalid.json"
        config_file.write_text("{ invalid json }")

        parser = build_parser()
        args = parser.parse_args(["transcribe", "--config", str(config_file)])

        with pytest.raises(json.JSONDecodeError):
            _config_from_transcribe_args(args)


# ============================================================================
# Environment Variable Handling Tests
# ============================================================================


class TestEnvironmentVariableHandling:
    """Test environment variable handling in CLI."""

    def test_transcribe_env_vars_applied(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Environment variables are applied when no CLI flags given."""
        monkeypatch.setenv("SLOWER_WHISPER_MODEL", "medium")
        monkeypatch.setenv("SLOWER_WHISPER_DEVICE", "cpu")
        monkeypatch.setenv("SLOWER_WHISPER_LANGUAGE", "fr")

        parser = build_parser()
        args = parser.parse_args(["transcribe"])

        config = _config_from_transcribe_args(args)
        assert config.model == "medium"
        assert config.device == "cpu"
        assert config.language == "fr"

    def test_transcribe_cli_overrides_env_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """CLI flags take precedence over environment variables."""
        monkeypatch.setenv("SLOWER_WHISPER_MODEL", "medium")
        monkeypatch.setenv("SLOWER_WHISPER_DEVICE", "cuda")

        parser = build_parser()
        args = parser.parse_args(["transcribe", "--model", "tiny", "--device", "cpu"])

        config = _config_from_transcribe_args(args)
        assert config.model == "tiny"
        assert config.device == "cpu"

    def test_enrich_env_vars_applied(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Enrich environment variables are applied."""
        monkeypatch.setenv("SLOWER_WHISPER_ENRICH_DEVICE", "cuda")
        monkeypatch.setenv("SLOWER_WHISPER_ENRICH_ENABLE_PROSODY", "false")

        parser = build_parser()
        args = parser.parse_args(["enrich"])

        config = _config_from_enrich_args(args)
        assert config.device == "cuda"
        assert config.enable_prosody is False

    def test_diarization_env_var_triggers_warning(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Diarization enabled via env var triggers experimental warning."""
        monkeypatch.setenv("SLOWER_WHISPER_ENABLE_DIARIZATION", "true")

        from transcription.pipeline import PipelineBatchResult

        mock_result = PipelineBatchResult(
            total_files=0,
            processed=0,
            skipped=0,
            diarized_only=0,
            failed=0,
            total_audio_seconds=0.0,
            total_time_seconds=0.0,
        )

        # Patch where run_pipeline is actually used (inside _handle_transcribe_command)
        with patch.object(
            __import__("transcription.pipeline", fromlist=["run_pipeline"]),
            "run_pipeline",
            return_value=mock_result,
        ):
            main(["transcribe"])

        captured = capsys.readouterr()
        assert "Speaker diarization is EXPERIMENTAL" in captured.err


# ============================================================================
# Exit Code Tests
# ============================================================================


class TestExitCodes:
    """Test CLI exit codes for various scenarios."""

    def test_success_exit_code_zero(self) -> None:
        """Successful commands return exit code 0."""
        from transcription.pipeline import PipelineBatchResult

        mock_result = PipelineBatchResult(
            total_files=1,
            processed=1,
            skipped=0,
            diarized_only=0,
            failed=0,
            total_audio_seconds=10.0,
            total_time_seconds=5.0,
        )

        # Patch at the module level where it's imported dynamically
        with patch.object(
            __import__("transcription.pipeline", fromlist=["run_pipeline"]),
            "run_pipeline",
            return_value=mock_result,
        ):
            exit_code = main(["transcribe"])

        assert exit_code == 0

    def test_partial_failure_exit_code_one(self) -> None:
        """Partial failures return exit code 1."""
        from transcription.pipeline import PipelineBatchResult

        mock_result = PipelineBatchResult(
            total_files=2,
            processed=1,
            skipped=0,
            diarized_only=0,
            failed=1,
            total_audio_seconds=10.0,
            total_time_seconds=5.0,
        )

        # Patch at the module level where it's imported dynamically
        with patch.object(
            __import__("transcription.pipeline", fromlist=["run_pipeline"]),
            "run_pipeline",
            return_value=mock_result,
        ):
            exit_code = main(["transcribe"])

        assert exit_code == 1

    def test_unexpected_error_exit_code_two(self) -> None:
        """Unexpected errors return exit code 2."""
        # Patch at the module level where it's imported dynamically
        with patch.object(
            __import__("transcription.pipeline", fromlist=["run_pipeline"]),
            "run_pipeline",
            side_effect=RuntimeError("Unexpected error"),
        ):
            exit_code = main(["transcribe"])

        assert exit_code == 2

    def test_configuration_error_exit_code_one(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Configuration errors return exit code 1."""
        exit_code = main(["enrich", "--root", "/nonexistent/path"])

        assert exit_code == 1
        captured = capsys.readouterr()
        assert "Error:" in captured.err

    def test_missing_subcommand_exit_code_two(self) -> None:
        """Missing subcommand returns exit code 2."""
        with pytest.raises(SystemExit) as exc_info:
            main([])

        assert exc_info.value.code == 2


# ============================================================================
# Help Text Tests
# ============================================================================


class TestHelpText:
    """Test help text generation and content."""

    def test_main_help_shows_all_subcommands(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Main help shows all available subcommands."""
        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])

        assert exc_info.value.code == 0
        captured = capsys.readouterr()

        # Check all subcommands are listed
        assert "transcribe" in captured.out
        assert "enrich" in captured.out
        assert "cache" in captured.out
        assert "samples" in captured.out
        assert "export" in captured.out
        assert "validate" in captured.out
        assert "benchmark" in captured.out

    def test_transcribe_help_shows_options(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Transcribe help shows all options."""
        with pytest.raises(SystemExit) as exc_info:
            main(["transcribe", "--help"])

        assert exc_info.value.code == 0
        captured = capsys.readouterr()

        # Check key options are documented
        assert "--model" in captured.out
        assert "--device" in captured.out
        assert "--language" in captured.out
        assert "--enable-diarization" in captured.out
        assert "--word-timestamps" in captured.out
        assert "--enable-chunking" in captured.out

    def test_enrich_help_shows_options(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Enrich help shows all options."""
        with pytest.raises(SystemExit) as exc_info:
            main(["enrich", "--help"])

        assert exc_info.value.code == 0
        captured = capsys.readouterr()

        # Check key options are documented
        assert "--enable-prosody" in captured.out
        assert "--enable-emotion" in captured.out
        assert "--enable-categorical-emotion" in captured.out
        assert "--enable-semantics" in captured.out
        assert "--enable-speaker-analytics" in captured.out
        assert "--pause-threshold" in captured.out

    def test_cache_help_shows_options(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Cache help shows options."""
        with pytest.raises(SystemExit) as exc_info:
            main(["cache", "--help"])

        assert exc_info.value.code == 0
        captured = capsys.readouterr()

        assert "--show" in captured.out
        assert "--clear" in captured.out

    def test_samples_help_shows_actions(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Samples help shows available actions."""
        with pytest.raises(SystemExit) as exc_info:
            main(["samples", "--help"])

        assert exc_info.value.code == 0
        captured = capsys.readouterr()

        assert "list" in captured.out
        assert "download" in captured.out
        assert "copy" in captured.out
        assert "generate" in captured.out

    def test_export_help_shows_formats(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Export help shows available formats."""
        with pytest.raises(SystemExit) as exc_info:
            main(["export", "--help"])

        assert exc_info.value.code == 0
        captured = capsys.readouterr()

        assert "csv" in captured.out
        assert "html" in captured.out
        assert "vtt" in captured.out
        assert "textgrid" in captured.out

    def test_benchmark_help_shows_tracks(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Benchmark help shows available tracks."""
        with pytest.raises(SystemExit) as exc_info:
            main(["benchmark", "run", "--help"])

        assert exc_info.value.code == 0
        captured = capsys.readouterr()

        assert "asr" in captured.out
        assert "diarization" in captured.out
        assert "streaming" in captured.out
        assert "semantic" in captured.out
        assert "emotion" in captured.out


# ============================================================================
# Utility Function Tests
# ============================================================================


class TestUtilityFunctions:
    """Test CLI utility functions."""

    def test_format_size_bytes(self) -> None:
        """Format size handles bytes correctly."""
        assert _format_size(0) == "0.0 B"
        assert _format_size(512) == "512.0 B"
        assert _format_size(1023) == "1023.0 B"

    def test_format_size_kilobytes(self) -> None:
        """Format size handles kilobytes correctly."""
        assert _format_size(1024) == "1.0 KB"
        assert _format_size(1536) == "1.5 KB"

    def test_format_size_megabytes(self) -> None:
        """Format size handles megabytes correctly."""
        assert _format_size(1024 * 1024) == "1.0 MB"
        assert _format_size(1024 * 1024 * 2) == "2.0 MB"

    def test_format_size_gigabytes(self) -> None:
        """Format size handles gigabytes correctly."""
        assert _format_size(1024 * 1024 * 1024) == "1.0 GB"
        assert _format_size(1024 * 1024 * 1024 * 4) == "4.0 GB"

    def test_get_cache_size_empty_dir(self, tmp_path: Path) -> None:
        """Get cache size for empty directory returns 0."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        size = _get_cache_size(empty_dir)
        assert size == 0

    def test_get_cache_size_with_files(self, tmp_path: Path) -> None:
        """Get cache size counts all files."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        (cache_dir / "file1.bin").write_bytes(b"x" * 100)
        (cache_dir / "file2.bin").write_bytes(b"y" * 200)

        size = _get_cache_size(cache_dir)
        assert size == 300

    def test_get_cache_size_nested_dirs(self, tmp_path: Path) -> None:
        """Get cache size includes nested directories."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        nested = cache_dir / "nested" / "deep"
        nested.mkdir(parents=True)
        (nested / "file.bin").write_bytes(b"z" * 50)

        size = _get_cache_size(cache_dir)
        assert size == 50

    def test_get_cache_size_nonexistent_dir(self, tmp_path: Path) -> None:
        """Get cache size for nonexistent directory returns 0."""
        nonexistent = tmp_path / "does_not_exist"

        size = _get_cache_size(nonexistent)
        assert size == 0


# ============================================================================
# Export Subcommand Tests
# ============================================================================


class TestExportSubcommand:
    """Test export subcommand parsing and behavior."""

    def test_export_requires_transcript(self) -> None:
        """Export requires transcript path argument."""
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["export", "--format", "csv"])

    def test_export_requires_format(self, tmp_path: Path) -> None:
        """Export requires --format flag."""
        transcript = tmp_path / "test.json"
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["export", str(transcript)])

    @pytest.mark.parametrize("fmt", ["csv", "html", "vtt", "textgrid"])
    def test_export_valid_formats(self, fmt: str, tmp_path: Path) -> None:
        """Export accepts all valid formats."""
        transcript = tmp_path / "test.json"
        parser = build_parser()
        args = parser.parse_args(["export", str(transcript), "--format", fmt])

        assert args.format == fmt

    def test_export_invalid_format(self, tmp_path: Path) -> None:
        """Export rejects invalid formats."""
        transcript = tmp_path / "test.json"
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["export", str(transcript), "--format", "invalid"])

    def test_export_unit_options(self, tmp_path: Path) -> None:
        """Export --unit accepts segments and turns."""
        transcript = tmp_path / "test.json"
        parser = build_parser()

        args_seg = parser.parse_args(
            ["export", str(transcript), "--format", "csv", "--unit", "segments"]
        )
        assert args_seg.unit == "segments"

        args_turn = parser.parse_args(
            ["export", str(transcript), "--format", "csv", "--unit", "turns"]
        )
        assert args_turn.unit == "turns"

    def test_export_output_path(self, tmp_path: Path) -> None:
        """Export --output sets output path."""
        transcript = tmp_path / "test.json"
        output = tmp_path / "output.csv"
        parser = build_parser()
        args = parser.parse_args(
            ["export", str(transcript), "--format", "csv", "--output", str(output)]
        )

        assert args.output == output


# ============================================================================
# Validate Subcommand Tests
# ============================================================================


class TestValidateSubcommand:
    """Test validate subcommand parsing and behavior."""

    def test_validate_requires_transcripts(self) -> None:
        """Validate requires at least one transcript path."""
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["validate"])

    def test_validate_single_transcript(self, tmp_path: Path) -> None:
        """Validate accepts single transcript."""
        transcript = tmp_path / "test.json"
        parser = build_parser()
        args = parser.parse_args(["validate", str(transcript)])

        assert len(args.transcripts) == 1
        assert args.transcripts[0] == transcript

    def test_validate_multiple_transcripts(self, tmp_path: Path) -> None:
        """Validate accepts multiple transcripts."""
        t1 = tmp_path / "test1.json"
        t2 = tmp_path / "test2.json"
        parser = build_parser()
        args = parser.parse_args(["validate", str(t1), str(t2)])

        assert len(args.transcripts) == 2

    def test_validate_custom_schema(self, tmp_path: Path) -> None:
        """Validate --schema sets custom schema path."""
        transcript = tmp_path / "test.json"
        schema = tmp_path / "custom-schema.json"
        parser = build_parser()
        args = parser.parse_args(["validate", str(transcript), "--schema", str(schema)])

        assert args.schema == schema


# ============================================================================
# Word Timestamps Tests
# ============================================================================


class TestWordTimestamps:
    """Test word-timestamps flag parsing (v1.8+ feature)."""

    def test_word_timestamps_default_none(self) -> None:
        """Word timestamps default to None (not explicitly set)."""
        parser = build_parser()
        args = parser.parse_args(["transcribe"])

        assert args.word_timestamps is None

    def test_word_timestamps_enabled(self) -> None:
        """--word-timestamps enables word-level timestamps."""
        parser = build_parser()
        args = parser.parse_args(["transcribe", "--word-timestamps"])

        assert args.word_timestamps is True

    def test_word_timestamps_disabled(self) -> None:
        """--no-word-timestamps disables word-level timestamps."""
        parser = build_parser()
        args = parser.parse_args(["transcribe", "--no-word-timestamps"])

        assert args.word_timestamps is False


# ============================================================================
# Chunking Options Tests
# ============================================================================


class TestChunkingOptions:
    """Test chunking-related CLI options."""

    def test_enable_chunking_flag(self) -> None:
        """--enable-chunking enables chunking."""
        parser = build_parser()
        args = parser.parse_args(["transcribe", "--enable-chunking"])

        assert args.enable_chunking is True

    def test_disable_chunking_flag(self) -> None:
        """--no-enable-chunking disables chunking."""
        parser = build_parser()
        args = parser.parse_args(["transcribe", "--no-enable-chunking"])

        assert args.enable_chunking is False

    def test_chunk_target_duration(self) -> None:
        """--chunk-target-duration-s accepts float values."""
        parser = build_parser()
        args = parser.parse_args(
            ["transcribe", "--enable-chunking", "--chunk-target-duration-s", "45.5"]
        )

        assert args.chunk_target_duration_s == 45.5

    def test_chunk_max_duration(self) -> None:
        """--chunk-max-duration-s accepts float values."""
        parser = build_parser()
        args = parser.parse_args(
            ["transcribe", "--enable-chunking", "--chunk-max-duration-s", "60.0"]
        )

        assert args.chunk_max_duration_s == 60.0

    def test_chunk_target_tokens(self) -> None:
        """--chunk-target-tokens accepts integer values."""
        parser = build_parser()
        args = parser.parse_args(
            ["transcribe", "--enable-chunking", "--chunk-target-tokens", "500"]
        )

        assert args.chunk_target_tokens == 500

    def test_chunk_pause_threshold(self) -> None:
        """--chunk-pause-split-threshold-s accepts float values."""
        parser = build_parser()
        args = parser.parse_args(
            [
                "transcribe",
                "--enable-chunking",
                "--chunk-pause-split-threshold-s",
                "2.0",
            ]
        )

        assert args.chunk_pause_split_threshold_s == 2.0


# ============================================================================
# Skip Existing Alias Tests
# ============================================================================


class TestSkipExistingAlias:
    """Test --skip-existing alias for transcribe command."""

    def test_skip_existing_alias_works(self) -> None:
        """--skip-existing is alias for --skip-existing-json."""
        parser = build_parser()

        args1 = parser.parse_args(["transcribe", "--skip-existing"])
        args2 = parser.parse_args(["transcribe", "--skip-existing-json"])

        assert args1.skip_existing_json is True
        assert args2.skip_existing_json is True

    def test_no_skip_existing_alias_works(self) -> None:
        """--no-skip-existing is alias for --no-skip-existing-json."""
        parser = build_parser()

        args1 = parser.parse_args(["transcribe", "--no-skip-existing"])
        args2 = parser.parse_args(["transcribe", "--no-skip-existing-json"])

        assert args1.skip_existing_json is False
        assert args2.skip_existing_json is False


# ============================================================================
# Progress Flag Tests
# ============================================================================


class TestProgressFlag:
    """Test --progress flag behavior."""

    def test_transcribe_progress_default_false(self) -> None:
        """Transcribe --progress defaults to False."""
        parser = build_parser()
        args = parser.parse_args(["transcribe"])

        assert args.progress is False

    def test_transcribe_progress_enabled(self) -> None:
        """Transcribe --progress can be enabled."""
        parser = build_parser()
        args = parser.parse_args(["transcribe", "--progress"])

        assert args.progress is True

    def test_enrich_progress_default_false(self) -> None:
        """Enrich --progress defaults to False."""
        parser = build_parser()
        args = parser.parse_args(["enrich"])

        assert args.progress is False

    def test_enrich_progress_enabled(self) -> None:
        """Enrich --progress can be enabled."""
        parser = build_parser()
        args = parser.parse_args(["enrich", "--progress"])

        assert args.progress is True


# ============================================================================
# Enrich Pause Threshold Tests
# ============================================================================


class TestEnrichPauseThreshold:
    """Test enrich --pause-threshold option."""

    def test_pause_threshold_default_none(self) -> None:
        """Pause threshold defaults to None."""
        parser = build_parser()
        args = parser.parse_args(["enrich"])

        assert getattr(args, "pause_threshold", None) is None

    def test_pause_threshold_accepts_float(self) -> None:
        """Pause threshold accepts float values."""
        parser = build_parser()
        args = parser.parse_args(["enrich", "--pause-threshold", "2.5"])

        assert args.pause_threshold == 2.5

    def test_pause_threshold_propagates_to_config(self) -> None:
        """Pause threshold is propagated to EnrichmentConfig."""
        parser = build_parser()
        args = parser.parse_args(["enrich", "--pause-threshold", "3.0"])

        config = _config_from_enrich_args(args)
        assert config.pause_threshold == 3.0
