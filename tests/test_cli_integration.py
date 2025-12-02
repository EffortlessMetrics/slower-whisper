"""
Integration tests for the unified CLI (transcription/cli.py).

Tests the slower-whisper command-line interface including:
- Main entry point and subcommand routing
- transcribe subcommand with various options
- enrich subcommand with various options
- Error handling and validation
- Help text generation
- Configuration parsing and conversion
- Integration with the API layer (mocked)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from transcription.cli import (
    _config_from_enrich_args,
    _config_from_transcribe_args,
    build_parser,
    main,
)
from transcription.config import EnrichmentConfig, TranscriptionConfig

pytestmark = pytest.mark.integration

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_transcribe_directory():
    """Mock the transcribe_directory API function and run_pipeline."""
    from transcription.pipeline import PipelineBatchResult

    # Create a mock PipelineBatchResult for run_pipeline
    mock_batch_result = PipelineBatchResult(
        total_files=3,
        processed=3,
        skipped=0,
        diarized_only=0,
        failed=0,
        total_audio_seconds=10.0,
        total_time_seconds=5.0,
    )

    # Patch both run_pipeline (at source) and transcribe_directory for compatibility
    with (
        patch("transcription.pipeline.run_pipeline") as mock_pipeline,
        patch("transcription.cli.transcribe_directory") as mock_transcribe,
    ):
        mock_pipeline.return_value = mock_batch_result
        mock_transcribe.return_value = [MagicMock(), MagicMock(), MagicMock()]
        yield mock_transcribe


@pytest.fixture
def mock_enrich_directory():
    """Mock the enrich_directory API function."""
    with patch("transcription.cli.enrich_directory") as mock:
        # Return a list of mock enriched transcripts
        mock.return_value = [MagicMock(), MagicMock()]
        yield mock


@pytest.fixture
def temp_project_root(tmp_path):
    """Create a temporary project structure."""
    root = tmp_path / "project"
    root.mkdir()
    (root / "raw_audio").mkdir()
    (root / "input_audio").mkdir()
    (root / "whisper_json").mkdir()
    (root / "transcripts").mkdir()
    return root


@pytest.fixture
def sample_transcript_file(tmp_path: Path) -> Path:
    """Write a minimal, schema-valid transcript JSON for export/validate commands."""
    path = tmp_path / "sample_transcript.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "file": "sample.wav",
                "language": "en",
                "segments": [
                    {
                        "id": 0,
                        "start": 0.0,
                        "end": 1.2,
                        "text": "Hello there",
                        "speaker": {"id": "spk_0", "confidence": 0.98},
                    },
                    {
                        "id": 1,
                        "start": 1.3,
                        "end": 2.1,
                        "text": "Testing export flow",
                        "speaker": {"id": "spk_1", "confidence": 0.97},
                    },
                ],
                "turns": [
                    {
                        "id": "turn_0",
                        "speaker_id": "spk_0",
                        "segment_ids": [0],
                        "start": 0.0,
                        "end": 1.2,
                        "text": "Hello there",
                    },
                    {
                        "id": "turn_1",
                        "speaker_id": "spk_1",
                        "segment_ids": [1],
                        "start": 1.3,
                        "end": 2.1,
                        "text": "Testing export flow",
                    },
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


# ============================================================================
# Parser Tests
# ============================================================================


def test_build_parser():
    """Test that build_parser creates a valid ArgumentParser."""
    parser = build_parser()

    assert isinstance(parser, argparse.ArgumentParser)
    assert parser.prog == "slower-whisper"
    assert "slower-whisper" in parser.format_help().lower()


def test_parser_requires_subcommand():
    """Test that parser requires a subcommand."""
    parser = build_parser()

    # Should raise SystemExit when no subcommand is provided
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_parser_transcribe_subcommand_exists():
    """Test that transcribe subcommand is available."""
    parser = build_parser()

    # Should parse without error
    args = parser.parse_args(["transcribe"])
    assert args.command == "transcribe"


def test_parser_enrich_subcommand_exists():
    """Test that enrich subcommand is available."""
    parser = build_parser()

    # Should parse without error
    args = parser.parse_args(["enrich"])
    assert args.command == "enrich"


def test_parser_invalid_subcommand():
    """Test that invalid subcommand raises error."""
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["invalid-command"])


# ============================================================================
# Transcribe Subcommand Tests
# ============================================================================


def test_transcribe_default_args():
    """Test transcribe subcommand with default arguments.

    Note: CLI args default to None, actual defaults are applied in _config_from_transcribe_args().
    """
    parser = build_parser()
    args = parser.parse_args(["transcribe"])

    assert args.command == "transcribe"
    assert args.root == Path(".")
    # CLI args default to None - actual defaults applied by TranscriptionConfig
    assert args.model is None
    assert args.device is None
    assert args.compute_type is None
    assert args.language is None
    assert args.task is None
    assert args.vad_min_silence_ms is None
    assert args.beam_size is None
    assert args.skip_existing_json is None
    assert args.enable_diarization is None
    assert args.diarization_device is None
    assert args.min_speakers is None
    assert args.max_speakers is None
    assert args.overlap_threshold is None


def test_transcribe_custom_root():
    """Test transcribe with custom root directory."""
    parser = build_parser()
    args = parser.parse_args(["transcribe", "--root", "/path/to/project"])

    assert args.root == Path("/path/to/project")


def test_transcribe_custom_model():
    """Test transcribe with custom model."""
    parser = build_parser()
    args = parser.parse_args(["transcribe", "--model", "base"])

    assert args.model == "base"


def test_transcribe_cpu_device():
    """Test transcribe with CPU device."""
    parser = build_parser()
    args = parser.parse_args(["transcribe", "--device", "cpu"])

    assert args.device == "cpu"


def test_transcribe_language_option():
    """Test transcribe with language option."""
    parser = build_parser()
    args = parser.parse_args(["transcribe", "--language", "en"])

    assert args.language == "en"


def test_transcribe_translate_task():
    """Test transcribe with translate task."""
    parser = build_parser()
    args = parser.parse_args(["transcribe", "--task", "translate"])

    assert args.task == "translate"


def test_transcribe_custom_vad_settings():
    """Test transcribe with custom VAD settings."""
    parser = build_parser()
    args = parser.parse_args(["transcribe", "--vad-min-silence-ms", "1000"])

    assert args.vad_min_silence_ms == 1000


def test_transcribe_custom_beam_size():
    """Test transcribe with custom beam size."""
    parser = build_parser()
    args = parser.parse_args(["transcribe", "--beam-size", "10"])

    assert args.beam_size == 10


def test_transcribe_skip_existing_flag():
    """Test transcribe with skip existing flag variations."""
    parser = build_parser()

    # Test --skip-existing-json
    args = parser.parse_args(["transcribe", "--skip-existing-json"])
    assert args.skip_existing_json is True

    # Test --no-skip-existing-json
    args = parser.parse_args(["transcribe", "--no-skip-existing-json"])
    assert args.skip_existing_json is False


def test_transcribe_all_options():
    """Test transcribe with all options specified."""
    parser = build_parser()
    args = parser.parse_args(
        [
            "transcribe",
            "--root",
            "/data/project",
            "--model",
            "medium",
            "--device",
            "cpu",
            "--compute-type",
            "int8",
            "--language",
            "fr",
            "--task",
            "translate",
            "--vad-min-silence-ms",
            "700",
            "--beam-size",
            "8",
            "--no-skip-existing-json",
        ]
    )

    assert args.root == Path("/data/project")
    assert args.model == "medium"
    assert args.device == "cpu"
    assert args.compute_type == "int8"
    assert args.language == "fr"
    assert args.task == "translate"
    assert args.vad_min_silence_ms == 700
    assert args.beam_size == 8
    assert args.skip_existing_json is False
    assert args.enable_diarization is None
    assert args.diarization_device is None
    assert args.min_speakers is None
    assert args.max_speakers is None
    assert args.overlap_threshold is None


def test_transcribe_diarization_options():
    """Test diarization-related arguments are parsed."""
    parser = build_parser()
    args = parser.parse_args(
        [
            "transcribe",
            "--enable-diarization",
            "--diarization-device",
            "cpu",
            "--min-speakers",
            "2",
            "--max-speakers",
            "4",
            "--overlap-threshold",
            "0.45",
        ]
    )

    assert args.enable_diarization is True
    assert args.diarization_device == "cpu"
    assert args.min_speakers == 2
    assert args.max_speakers == 4
    assert args.overlap_threshold == 0.45


# ============================================================================
# Enrich Subcommand Tests
# ============================================================================


def test_enrich_default_args():
    """Test enrich subcommand with default arguments.

    Note: CLI args default to None, actual defaults are applied in _config_from_enrich_args().
    """
    parser = build_parser()
    args = parser.parse_args(["enrich"])

    assert args.command == "enrich"
    assert args.root == Path(".")
    # CLI args default to None - actual defaults applied by EnrichmentConfig
    assert args.skip_existing is None
    assert args.enable_prosody is None
    assert args.enable_emotion is None
    assert args.enable_categorical_emotion is None
    assert args.enable_semantic_annotator is None
    assert args.device is None


def test_enrich_custom_root():
    """Test enrich with custom root directory."""
    parser = build_parser()
    args = parser.parse_args(["enrich", "--root", "/path/to/project"])

    assert args.root == Path("/path/to/project")


def test_enrich_disable_prosody():
    """Test enrich with prosody disabled."""
    parser = build_parser()
    args = parser.parse_args(["enrich", "--no-enable-prosody"])

    assert args.enable_prosody is False


def test_enrich_disable_emotion():
    """Test enrich with emotion disabled."""
    parser = build_parser()
    args = parser.parse_args(["enrich", "--no-enable-emotion"])

    assert args.enable_emotion is False


def test_enrich_enable_categorical_emotion():
    """Test enrich with categorical emotion enabled."""
    parser = build_parser()
    args = parser.parse_args(["enrich", "--enable-categorical-emotion"])

    assert args.enable_categorical_emotion is True


def test_enrich_cuda_device():
    """Test enrich with CUDA device."""
    parser = build_parser()
    args = parser.parse_args(["enrich", "--device", "cuda"])

    assert args.device == "cuda"


def test_enrich_skip_existing_flag():
    """Test enrich with skip existing flag variations."""
    parser = build_parser()

    # Test --skip-existing
    args = parser.parse_args(["enrich", "--skip-existing"])
    assert args.skip_existing is True

    # Test --no-skip-existing
    args = parser.parse_args(["enrich", "--no-skip-existing"])
    assert args.skip_existing is False


def test_enrich_all_options():
    """Test enrich with all options specified."""
    parser = build_parser()
    args = parser.parse_args(
        [
            "enrich",
            "--root",
            "/data/project",
            "--no-skip-existing",
            "--no-enable-prosody",
            "--no-enable-emotion",
            "--enable-categorical-emotion",
            "--device",
            "cuda",
        ]
    )

    assert args.root == Path("/data/project")
    assert args.skip_existing is False
    assert args.enable_prosody is False
    assert args.enable_emotion is False
    assert args.enable_categorical_emotion is True
    assert args.device == "cuda"


# ============================================================================
# Configuration Conversion Tests
# ============================================================================


def test_config_from_transcribe_args_defaults():
    """Test conversion from transcribe args to TranscriptionConfig with defaults."""
    parser = build_parser()
    args = parser.parse_args(["transcribe"])

    config = _config_from_transcribe_args(args)

    assert isinstance(config, TranscriptionConfig)
    assert config.model == "large-v3"
    assert config.device == "cuda"
    assert config.compute_type == "float16"
    assert config.language is None
    assert config.task == "transcribe"
    assert config.vad_min_silence_ms == 500
    assert config.beam_size == 5
    assert config.skip_existing_json is True
    assert config.enable_diarization is False
    assert config.diarization_device == "auto"
    assert config.min_speakers is None
    assert config.max_speakers is None
    assert config.overlap_threshold == 0.3


def test_config_from_transcribe_args_custom():
    """Test conversion from transcribe args to TranscriptionConfig with custom values."""
    parser = build_parser()
    args = parser.parse_args(
        [
            "transcribe",
            "--model",
            "base",
            "--device",
            "cpu",
            "--compute-type",
            "int8",
            "--language",
            "es",
            "--task",
            "translate",
            "--vad-min-silence-ms",
            "1000",
            "--beam-size",
            "3",
            "--no-skip-existing-json",
        ]
    )

    config = _config_from_transcribe_args(args)

    assert config.model == "base"
    assert config.device == "cpu"
    assert config.compute_type == "int8"
    assert config.language == "es"
    assert config.task == "translate"
    assert config.vad_min_silence_ms == 1000
    assert config.beam_size == 3
    assert config.skip_existing_json is False
    assert config.enable_diarization is False
    assert config.diarization_device == "auto"
    assert config.min_speakers is None
    assert config.max_speakers is None
    assert config.overlap_threshold == 0.3


def test_config_from_transcribe_args_diarization_fields():
    """Test conversion from transcribe args includes diarization settings."""
    parser = build_parser()
    args = parser.parse_args(
        [
            "transcribe",
            "--enable-diarization",
            "--diarization-device",
            "cpu",
            "--min-speakers",
            "2",
            "--max-speakers",
            "4",
            "--overlap-threshold",
            "0.45",
        ]
    )

    config = _config_from_transcribe_args(args)

    assert config.enable_diarization is True
    assert config.diarization_device == "cpu"
    assert config.min_speakers == 2
    assert config.max_speakers == 4
    assert config.overlap_threshold == 0.45


def test_config_from_enrich_args_defaults():
    """Test conversion from enrich args to EnrichmentConfig with defaults."""
    parser = build_parser()
    args = parser.parse_args(["enrich"])

    config = _config_from_enrich_args(args)

    assert isinstance(config, EnrichmentConfig)
    assert config.skip_existing is True
    assert config.enable_prosody is True
    assert config.enable_emotion is True
    assert config.enable_categorical_emotion is False
    assert config.enable_turn_metadata is True
    assert config.enable_speaker_stats is True
    assert config.enable_semantic_annotator is False
    assert config.device == "cpu"


def test_config_from_enrich_args_custom():
    """Test conversion from enrich args to EnrichmentConfig with custom values."""
    parser = build_parser()
    args = parser.parse_args(
        [
            "enrich",
            "--no-skip-existing",
            "--no-enable-prosody",
            "--no-enable-emotion",
            "--enable-categorical-emotion",
            "--no-enable-turn-metadata",
            "--no-enable-speaker-stats",
            "--device",
            "cuda",
        ]
    )

    config = _config_from_enrich_args(args)

    assert config.skip_existing is False
    assert config.enable_prosody is False
    assert config.enable_emotion is False
    assert config.enable_categorical_emotion is True
    assert config.enable_turn_metadata is False
    assert config.enable_speaker_stats is False
    assert config.enable_semantic_annotator is False
    assert config.device == "cuda"


def test_config_from_enrich_args_speaker_analytics_toggle():
    """Speaker analytics convenience flag should set both underlying toggles."""
    parser = build_parser()
    args = parser.parse_args(["enrich", "--no-enable-speaker-analytics"])

    config = _config_from_enrich_args(args)

    assert config.enable_turn_metadata is False
    assert config.enable_speaker_stats is False


def test_enrich_semantic_annotator_flag():
    """Semantic annotator flag should propagate into EnrichmentConfig."""
    parser = build_parser()
    args = parser.parse_args(["enrich", "--enable-semantics"])

    config = _config_from_enrich_args(args)
    assert config.enable_semantic_annotator is True


# ============================================================================
# Main Entry Point Tests
# ============================================================================


def test_main_transcribe_integration(mock_transcribe_directory, temp_project_root, capsys):
    """Test main() with transcribe subcommand calls API correctly."""
    # Call main with transcribe arguments
    main(
        [
            "transcribe",
            "--root",
            str(temp_project_root),
            "--model",
            "base",
            "--device",
            "cpu",
        ]
    )

    # Verify API was called
    assert mock_transcribe_directory.called
    call_args = mock_transcribe_directory.call_args

    # Check root path
    assert call_args[0][0] == temp_project_root

    # Check config
    config = call_args[1]["config"]
    assert isinstance(config, TranscriptionConfig)
    assert config.model == "base"
    assert config.device == "cpu"

    # Check output
    captured = capsys.readouterr()
    assert "[done] Transcribed 3 files" in captured.out


def test_main_transcribe_warns_when_diarization_enabled_cli(
    mock_transcribe_directory, temp_project_root, capsys
):
    """Experimental diarization warning should print when enabled via CLI flag."""
    main(
        [
            "transcribe",
            "--root",
            str(temp_project_root),
            "--enable-diarization",
        ]
    )

    captured = capsys.readouterr()
    assert "Speaker diarization is EXPERIMENTAL" in captured.err


def test_main_transcribe_warns_when_diarization_enabled_env(
    mock_transcribe_directory, temp_project_root, capsys, monkeypatch
):
    """Experimental diarization warning should also print when enabled via env/config."""
    monkeypatch.setenv("SLOWER_WHISPER_ENABLE_DIARIZATION", "true")

    main(
        [
            "transcribe",
            "--root",
            str(temp_project_root),
        ]
    )

    captured = capsys.readouterr()
    assert "Speaker diarization is EXPERIMENTAL" in captured.err


def test_main_enrich_integration(mock_enrich_directory, temp_project_root, capsys):
    """Test main() with enrich subcommand calls API correctly."""
    # Call main with enrich arguments
    main(
        [
            "enrich",
            "--root",
            str(temp_project_root),
            "--device",
            "cuda",
            "--enable-categorical-emotion",
        ]
    )

    # Verify API was called
    assert mock_enrich_directory.called
    call_args = mock_enrich_directory.call_args

    # Check root path
    assert call_args[0][0] == temp_project_root

    # Check config
    config = call_args[1]["config"]
    assert isinstance(config, EnrichmentConfig)
    assert config.device == "cuda"
    assert config.enable_categorical_emotion is True

    # Check output
    captured = capsys.readouterr()
    assert "[done] Enriched 2 transcripts" in captured.out


def test_main_no_args_shows_help(capsys):
    """Test main() with no arguments shows help and exits."""
    with pytest.raises(SystemExit) as exc_info:
        main([])

    # Should exit with error code
    assert exc_info.value.code == 2


def test_main_invalid_command_shows_error(capsys):
    """Test main() with invalid command shows error."""
    with pytest.raises(SystemExit):
        main(["invalid-command"])


def test_main_transcribe_help(capsys):
    """Test that transcribe --help shows help text."""
    with pytest.raises(SystemExit) as exc_info:
        main(["transcribe", "--help"])

    # Help should exit with code 0
    assert exc_info.value.code == 0

    captured = capsys.readouterr()
    assert "transcribe" in captured.out.lower()
    assert "--model" in captured.out
    assert "--device" in captured.out


def test_main_enrich_help(capsys):
    """Test that enrich --help shows help text."""
    with pytest.raises(SystemExit) as exc_info:
        main(["enrich", "--help"])

    # Help should exit with code 0
    assert exc_info.value.code == 0

    captured = capsys.readouterr()
    assert "enrich" in captured.out.lower()
    assert "--enable-prosody" in captured.out
    assert "--enable-emotion" in captured.out


def test_main_global_help(capsys):
    """Test that --help shows global help text."""
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])

    # Help should exit with code 0
    assert exc_info.value.code == 0

    captured = capsys.readouterr()
    help_text = captured.out.lower()
    assert "slower-whisper" in help_text
    assert "transcribe" in help_text
    assert "enrich" in help_text


# ============================================================================
# Export / Validate Subcommand Tests
# ============================================================================


def test_export_cli_writes_csv(sample_transcript_file: Path, tmp_path: Path, capsys):
    """Export command should write CSV using turn rows when requested."""
    csv_path = tmp_path / "out.csv"
    exit_code = main(
        [
            "export",
            str(sample_transcript_file),
            "--format",
            "csv",
            "--unit",
            "turns",
            "--output",
            str(csv_path),
        ]
    )

    assert exit_code == 0
    csv_text = csv_path.read_text(encoding="utf-8")
    assert "id,start,end,speaker,text" in csv_text
    assert "turn_0" in csv_text and "turn_1" in csv_text
    captured = capsys.readouterr()
    assert "[done] Wrote csv" in captured.out


def test_validate_cli_reports_failures(sample_transcript_file: Path, tmp_path: Path, capsys):
    """Validate command should succeed on valid files and report schema errors."""
    pytest.importorskip("jsonschema")
    ok_exit = main(["validate", str(sample_transcript_file)])
    ok_output = capsys.readouterr()
    assert ok_exit == 0
    assert "[ok] 1 transcript(s) valid" in ok_output.out

    invalid = tmp_path / "bad_transcript.json"
    invalid.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "file": "bad.wav",
                "segments": [{"id": 0, "start": 0.0, "end": 1.0}],
                "turns": [],
            }
        ),
        encoding="utf-8",
    )

    fail_exit = main(["validate", str(invalid)])
    fail_output = capsys.readouterr()
    assert fail_exit == 1
    assert "Validation failed" in fail_output.out
    assert "text" in fail_output.out or "language" in fail_output.out


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================


def test_transcribe_with_minimal_args(mock_transcribe_directory, capsys):
    """Test transcribe with only required args uses defaults."""
    main(["transcribe"])

    assert mock_transcribe_directory.called
    config = mock_transcribe_directory.call_args[1]["config"]

    # Should use default values
    assert config.model == "large-v3"
    assert config.device == "cuda"


def test_enrich_with_minimal_args(mock_enrich_directory, capsys):
    """Test enrich with only required args uses defaults."""
    main(["enrich"])

    assert mock_enrich_directory.called
    config = mock_enrich_directory.call_args[1]["config"]

    # Should use default values
    assert config.enable_prosody is True
    assert config.enable_emotion is True
    assert config.enable_turn_metadata is True
    assert config.enable_speaker_stats is True
    assert config.device == "cpu"


def test_transcribe_invalid_task_choice():
    """Test that invalid task choice raises error."""
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["transcribe", "--task", "invalid"])


def test_enrich_invalid_device_choice():
    """Test that invalid device choice for enrich raises error."""
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["enrich", "--device", "tpu"])


def test_transcribe_invalid_beam_size():
    """Test that invalid beam size (non-integer) raises error."""
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["transcribe", "--beam-size", "invalid"])


def test_transcribe_invalid_vad_silence():
    """Test that invalid VAD silence (non-integer) raises error."""
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["transcribe", "--vad-min-silence-ms", "not-a-number"])


# ============================================================================
# Real-world Scenario Tests
# ============================================================================


def test_transcribe_realistic_workflow(mock_transcribe_directory, temp_project_root, capsys):
    """Test a realistic transcribe workflow with common options."""
    main(
        [
            "transcribe",
            "--root",
            str(temp_project_root),
            "--model",
            "large-v3",
            "--device",
            "cuda",
            "--language",
            "en",
            "--beam-size",
            "5",
            "--skip-existing-json",
        ]
    )

    assert mock_transcribe_directory.called
    config = mock_transcribe_directory.call_args[1]["config"]

    assert config.model == "large-v3"
    assert config.device == "cuda"
    assert config.language == "en"
    assert config.beam_size == 5
    assert config.skip_existing_json is True


def test_enrich_realistic_workflow(mock_enrich_directory, temp_project_root, capsys):
    """Test a realistic enrich workflow with common options."""
    main(
        [
            "enrich",
            "--root",
            str(temp_project_root),
            "--enable-prosody",
            "--enable-emotion",
            "--no-enable-categorical-emotion",
            "--device",
            "cpu",
            "--skip-existing",
        ]
    )

    assert mock_enrich_directory.called
    config = mock_enrich_directory.call_args[1]["config"]

    assert config.enable_prosody is True
    assert config.enable_emotion is True
    assert config.enable_categorical_emotion is False
    assert config.device == "cpu"
    assert config.skip_existing is True


def test_sequential_transcribe_then_enrich(
    mock_transcribe_directory,
    mock_enrich_directory,
    temp_project_root,
    capsys,
):
    """Test sequential workflow: transcribe then enrich."""
    # Step 1: Transcribe
    main(["transcribe", "--root", str(temp_project_root), "--model", "base"])

    assert mock_transcribe_directory.called
    transcribe_output = capsys.readouterr()
    assert "[done] Transcribed 3 files" in transcribe_output.out

    # Step 2: Enrich
    main(["enrich", "--root", str(temp_project_root)])

    assert mock_enrich_directory.called
    enrich_output = capsys.readouterr()
    assert "[done] Enriched 2 transcripts" in enrich_output.out


def test_transcribe_cpu_mode_for_compatibility(mock_transcribe_directory, capsys):
    """Test transcribe in CPU mode for systems without CUDA."""
    main(
        [
            "transcribe",
            "--device",
            "cpu",
            "--compute-type",
            "int8",
            "--model",
            "base",
        ]
    )

    config = mock_transcribe_directory.call_args[1]["config"]

    assert config.device == "cpu"
    assert config.compute_type == "int8"
    assert config.model == "base"


def test_enrich_prosody_only(mock_enrich_directory, capsys):
    """Test enrichment with only prosody enabled (lighter dependencies)."""
    main(
        [
            "enrich",
            "--enable-prosody",
            "--no-enable-emotion",
            "--no-enable-categorical-emotion",
        ]
    )

    config = mock_enrich_directory.call_args[1]["config"]

    assert config.enable_prosody is True
    assert config.enable_emotion is False
    assert config.enable_categorical_emotion is False


def test_enrich_emotion_only(mock_enrich_directory, capsys):
    """Test enrichment with only emotion enabled."""
    main(
        [
            "enrich",
            "--no-enable-prosody",
            "--enable-emotion",
            "--enable-categorical-emotion",
        ]
    )

    config = mock_enrich_directory.call_args[1]["config"]

    assert config.enable_prosody is False
    assert config.enable_emotion is True
    assert config.enable_categorical_emotion is True


# ============================================================================
# Integration with sys.argv
# ============================================================================


def test_main_called_without_args_uses_sys_argv(mock_transcribe_directory):
    """Test that main() without arguments uses sys.argv."""
    original_argv = sys.argv
    try:
        sys.argv = ["slower-whisper", "transcribe", "--model", "base"]
        main()

        assert mock_transcribe_directory.called
        config = mock_transcribe_directory.call_args[1]["config"]
        assert config.model == "base"
    finally:
        sys.argv = original_argv


# ============================================================================
# Path Handling Tests
# ============================================================================


def test_transcribe_relative_path(mock_transcribe_directory, capsys):
    """Test transcribe with relative path."""
    main(["transcribe", "--root", "./my_project"])

    assert mock_transcribe_directory.called
    root_arg = mock_transcribe_directory.call_args[0][0]
    assert root_arg == Path("./my_project")


def test_transcribe_absolute_path(mock_transcribe_directory, capsys):
    """Test transcribe with absolute path."""
    main(["transcribe", "--root", "/absolute/path/to/project"])

    assert mock_transcribe_directory.called
    root_arg = mock_transcribe_directory.call_args[0][0]
    assert root_arg == Path("/absolute/path/to/project")


def test_enrich_path_expansion(mock_enrich_directory, capsys):
    """Test that enrich handles various path formats."""
    main(["enrich", "--root", "~/Documents/project"])

    assert mock_enrich_directory.called
    root_arg = mock_enrich_directory.call_args[0][0]
    # Path should be created as-is (expansion happens later if needed)
    assert root_arg == Path("~/Documents/project")
