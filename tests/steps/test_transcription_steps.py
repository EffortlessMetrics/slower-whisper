"""
Step definitions for transcription BDD tests.

This module implements Gherkin steps for testing the transcription pipeline
using pytest-bdd. Steps use the public API from transcription.api.
"""

import shutil
from pathlib import Path

import numpy as np
import pytest
from pytest_bdd import given, parsers, scenarios, then, when

from transcription import TranscriptionConfig, transcribe_directory, transcribe_file

# Check if ffmpeg is available
FFMPEG_AVAILABLE = shutil.which("ffmpeg") is not None

# Load all scenarios from the feature file
# Mark all scenarios as xfail if ffmpeg is not available since they require audio normalization
pytestmark = pytest.mark.xfail(
    not FFMPEG_AVAILABLE, reason="Requires ffmpeg for audio normalization", strict=False
)

scenarios("../features/transcription.feature")


# ============================================================================
# Fixtures for test state
# ============================================================================


@pytest.fixture
def test_state():
    """Shared state dictionary for passing data between steps."""
    return {
        "project_root": None,
        "audio_files": [],
        "transcripts": [],
        "config": None,
        "original_json": None,
    }


# ============================================================================
# Helper functions
# ============================================================================


def create_test_wav(path: Path, duration: float = 1.0, sr: int = 16000):
    """Create a simple test WAV file with silence."""
    try:
        import soundfile as sf

        # Generate silent audio
        samples = int(duration * sr)
        audio = np.zeros(samples, dtype=np.float32)

        # Write WAV file
        sf.write(str(path), audio, sr)
    except ImportError:
        # Fallback: create an empty file if soundfile not available
        path.touch()


# ============================================================================
# Given steps (setup)
# ============================================================================


@given("a clean test project directory")
def clean_project_directory(test_state, tmp_path):
    """Create a clean temporary project directory."""
    project_root = tmp_path / "test_project"
    project_root.mkdir(exist_ok=True)

    # Create required subdirectories
    (project_root / "raw_audio").mkdir(exist_ok=True)
    (project_root / "input_audio").mkdir(exist_ok=True)
    (project_root / "whisper_json").mkdir(exist_ok=True)
    (project_root / "transcripts").mkdir(exist_ok=True)

    test_state["project_root"] = project_root


@given(parsers.parse('a project with a mono WAV file named "{filename}"'))
def project_with_wav_file(test_state, filename):
    """Create a test WAV file in the project."""
    project_root = test_state["project_root"]
    raw_audio_dir = project_root / "raw_audio"

    wav_path = raw_audio_dir / filename
    create_test_wav(wav_path)

    test_state["audio_files"].append(filename)


@given(parsers.parse('an audio file "{filename}"'))
def standalone_audio_file(test_state, tmp_path, filename):
    """Create a standalone audio file."""
    audio_path = tmp_path / filename
    create_test_wav(audio_path)

    test_state["audio_file"] = audio_path
    test_state["project_root"] = tmp_path / "project"
    test_state["project_root"].mkdir(exist_ok=True)


@given(parsers.parse('the file "{filename}" has already been transcribed'))
def already_transcribed_file(test_state, filename):
    """Create an existing transcript for a file."""
    project_root = test_state["project_root"]

    # First create the audio file
    raw_audio_dir = project_root / "raw_audio"
    wav_path = raw_audio_dir / filename
    create_test_wav(wav_path)

    # Transcribe it
    config = TranscriptionConfig(model="base", device="cpu")
    _ = transcribe_directory(project_root, config)

    # Store the original JSON content for comparison
    stem = Path(filename).stem
    json_path = project_root / "whisper_json" / f"{stem}.json"
    test_state["original_json"] = json_path.read_text()


@given("a project with audio files")
def project_with_multiple_files(test_state, datatable):
    """Create multiple audio files in the project."""
    project_root = test_state["project_root"]
    raw_audio_dir = project_root / "raw_audio"

    for row in datatable:
        filename = row["filename"]
        wav_path = raw_audio_dir / filename
        create_test_wav(wav_path)
        test_state["audio_files"].append(filename)


# ============================================================================
# When steps (actions)
# ============================================================================


@when("I transcribe the project with default settings")
def transcribe_with_defaults(test_state):
    """Transcribe the project using default configuration."""
    project_root = test_state["project_root"]
    config = TranscriptionConfig(model="base", device="cpu")

    test_state["config"] = config
    test_state["transcripts"] = transcribe_directory(project_root, config)


@when(parsers.parse('I transcribe the project with model "{model}" and language "{language}"'))
def transcribe_with_custom_config(test_state, model, language):
    """Transcribe with custom model and language settings."""
    project_root = test_state["project_root"]
    config = TranscriptionConfig(model=model, language=language, device="cpu")

    test_state["config"] = config
    test_state["transcripts"] = transcribe_directory(project_root, config)


@when("I transcribe the single file with default settings")
def transcribe_single_file(test_state):
    """Transcribe a single audio file."""
    audio_path = test_state["audio_file"]
    project_root = test_state["project_root"]
    config = TranscriptionConfig(model="base", device="cpu")

    test_state["config"] = config
    transcript = transcribe_file(audio_path, project_root, config)
    test_state["transcripts"] = [transcript]


@when("I transcribe the project with skip_existing_json enabled")
def transcribe_with_skip_existing(test_state):
    """Transcribe with skip_existing_json enabled."""
    project_root = test_state["project_root"]
    config = TranscriptionConfig(model="base", device="cpu", skip_existing_json=True)

    test_state["config"] = config
    test_state["transcripts"] = transcribe_directory(project_root, config)


# ============================================================================
# Then steps (assertions)
# ============================================================================


@then(parsers.parse('a transcript JSON exists for "{filename}"'))
def transcript_json_exists(test_state, filename):
    """Verify that a JSON transcript file exists."""
    project_root = test_state["project_root"]
    stem = Path(filename).stem
    json_path = project_root / "whisper_json" / f"{stem}.json"

    assert json_path.exists(), f"JSON file not found: {json_path}"


@then("the transcript contains at least one segment")
def transcript_has_segments(test_state):
    """Verify that the transcript has segments."""
    transcripts = test_state["transcripts"]
    assert len(transcripts) > 0, "No transcripts found"

    for transcript in transcripts:
        assert len(transcript.segments) > 0, f"No segments in {transcript.file_name}"


@then("the JSON file has schema version 2")
def json_has_schema_version(test_state):
    """Verify the JSON schema version."""
    project_root = test_state["project_root"]
    json_files = list((project_root / "whisper_json").glob("*.json"))

    assert len(json_files) > 0, "No JSON files found"

    import json

    for json_path in json_files:
        with open(json_path) as f:
            data = json.load(f)
        assert data.get("schema_version") == 2, f"Wrong schema version in {json_path.name}"


@then(parsers.parse('the transcript language is "{language}"'))
def transcript_has_language(test_state, language):
    """Verify the transcript language."""
    transcripts = test_state["transcripts"]

    for transcript in transcripts:
        assert transcript.language == language, (
            f"Expected language {language}, got {transcript.language}"
        )


@then(parsers.parse('the transcript metadata contains model "{model}"'))
def transcript_metadata_has_model(test_state, model):
    """Verify the transcript metadata contains the model name."""
    transcripts = test_state["transcripts"]

    for transcript in transcripts:
        assert transcript.meta is not None, "Transcript has no metadata"
        assert model in transcript.meta.get("model_name", ""), f"Model {model} not in metadata"


@then(parsers.parse('a transcript TXT exists for "{filename}"'))
def transcript_txt_exists(test_state, filename):
    """Verify that a TXT transcript file exists."""
    project_root = test_state["project_root"]
    stem = Path(filename).stem
    txt_path = project_root / "transcripts" / f"{stem}.txt"

    assert txt_path.exists(), f"TXT file not found: {txt_path}"


@then(parsers.parse('a transcript SRT exists for "{filename}"'))
def transcript_srt_exists(test_state, filename):
    """Verify that an SRT transcript file exists."""
    project_root = test_state["project_root"]
    stem = Path(filename).stem
    srt_path = project_root / "transcripts" / f"{stem}.srt"

    assert srt_path.exists(), f"SRT file not found: {srt_path}"


@then("the transcript has segments")
def transcript_object_has_segments(test_state):
    """Verify the transcript object has segments."""
    transcripts = test_state["transcripts"]
    assert len(transcripts) > 0, "No transcripts"
    assert len(transcripts[0].segments) > 0, "No segments in transcript"


@then(parsers.parse('the transcript file name is "{filename}"'))
def transcript_has_filename(test_state, filename):
    """Verify the transcript file name."""
    transcripts = test_state["transcripts"]
    assert len(transcripts) > 0, "No transcripts"
    assert transcripts[0].file_name == filename, (
        f"Expected {filename}, got {transcripts[0].file_name}"
    )


@then(parsers.parse('the transcript JSON for "{filename}" is unchanged'))
def transcript_json_unchanged(test_state, filename):
    """Verify that the JSON file was not modified."""
    project_root = test_state["project_root"]
    stem = Path(filename).stem
    json_path = project_root / "whisper_json" / f"{stem}.json"

    current_json = json_path.read_text()
    original_json = test_state["original_json"]

    assert current_json == original_json, f"JSON file was modified: {json_path}"


@then("transcript JSONs exist for all files")
def all_transcript_jsons_exist(test_state):
    """Verify JSON files exist for all audio files."""
    project_root = test_state["project_root"]
    audio_files = test_state["audio_files"]

    for filename in audio_files:
        stem = Path(filename).stem
        json_path = project_root / "whisper_json" / f"{stem}.json"
        assert json_path.exists(), f"JSON not found for {filename}"


@then("each transcript contains at least one segment")
def each_transcript_has_segments(test_state):
    """Verify that each transcript has at least one segment."""
    transcripts = test_state["transcripts"]

    for transcript in transcripts:
        assert len(transcript.segments) > 0, f"No segments in transcript for {transcript.file_name}"
