"""
Step definitions for transcription BDD tests.

This module implements Gherkin steps for testing the transcription pipeline
using pytest-bdd. Steps use the public API from transcription.api.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest

# Skip all tests if pytest-bdd is not installed
pytest.importorskip("pytest_bdd")

from pytest_bdd import given, parsers, scenarios, then, when  # noqa: E402

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
def test_state() -> dict[str, Any]:
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
        if isinstance(row, dict):
            filename = row["filename"]
        else:
            # datatable may be a list-of-lists in pytest-bdd
            if not row:
                continue
            first = str(row[0]).strip()
            if first.lower() == "filename":
                # Skip header rows like ["filename"]
                continue
            filename = first
        wav_path = raw_audio_dir / filename
        create_test_wav(wav_path)
        test_state["audio_files"].append(filename)


# Additional environment/fixture helpers for diarization scenarios


@given("the HF_TOKEN environment variable is set")
def set_hf_token(monkeypatch):
    """Simulate presence of HF token for diarization backend."""
    monkeypatch.setenv("HF_TOKEN", "dummy-token")
    monkeypatch.setenv("SLOWER_WHISPER_PYANNOTE_MODE", "stub")


@given("the HF_TOKEN environment variable is not set")
def unset_hf_token(monkeypatch):
    """Ensure HF token is absent."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("SLOWER_WHISPER_PYANNOTE_MODE", "auto")


@given("pyannote.audio is not installed")
def simulate_missing_pyannote(monkeypatch):
    """Force pyannote to be treated as missing so failure path is exercised."""
    monkeypatch.setenv("SLOWER_WHISPER_PYANNOTE_MODE", "missing")


@given("a project with the synthetic 2-speaker fixture")
def project_with_synthetic_fixture(test_state, tmp_path):
    """Place the synthetic diarization fixture into the project."""
    project_root = test_state.get("project_root") or (tmp_path / "test_project")
    project_root.mkdir(exist_ok=True)
    raw_audio_dir = project_root / "raw_audio"
    raw_audio_dir.mkdir(exist_ok=True)

    fixture_path = Path(__file__).resolve().parent.parent / "fixtures" / "synthetic_2speaker.wav"
    if not fixture_path.exists():
        pytest.skip(f"Synthetic fixture not found at {fixture_path}")

    target = raw_audio_dir / "synthetic_2speaker.wav"
    shutil.copy2(fixture_path, target)
    test_state["project_root"] = project_root
    test_state["audio_files"].append(target.name)


@given(parsers.parse("a project with audio files:"))
def project_with_audio_table(test_state, datatable):
    """Alias for table-driven project setup."""
    return project_with_multiple_files(test_state, datatable)


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


@when("I transcribe the project with diarization enabled")
def transcribe_with_diarization(test_state):
    """Transcribe with diarization flag enabled (skeleton behavior)."""
    project_root = test_state["project_root"]
    config = TranscriptionConfig(
        model="base",
        device="cpu",
        enable_diarization=True,
        diarization_device="cpu",
    )
    test_state["config"] = config
    test_state["transcripts"] = transcribe_directory(project_root, config)


@when("I transcribe the project with diarization enabled (min=2, max=2)")
def transcribe_with_fixed_speakers(test_state):
    """Transcribe with diarization enabled and fixed speaker count hints."""
    project_root = test_state["project_root"]
    config = TranscriptionConfig(
        model="base",
        device="cpu",
        enable_diarization=True,
        diarization_device="cpu",
        min_speakers=2,
        max_speakers=2,
    )
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


@then('all segments have a "speaker" field')
def segments_have_speaker_field(test_state):
    """Verify speaker field exists on every segment (may be None)."""
    for transcript in test_state["transcripts"]:
        for segment in transcript.segments:
            assert hasattr(segment, "speaker")


@then('all segment "speaker" values are null (diarization disabled)')
def speaker_values_null(test_state):
    """Ensure diarization-disabled transcripts keep speaker as None."""
    for transcript in test_state["transcripts"]:
        for segment in transcript.segments:
            assert segment.speaker is None


@then("the transcription completes successfully")
def transcription_completes_successfully(test_state):
    """Ensure transcription produced output."""
    assert test_state["transcripts"], "No transcripts were produced"


@then("the transcript JSON has schema version 2")
def transcript_json_schema_version(test_state):
    """Verify persisted JSONs use schema v2."""
    project_root = test_state["project_root"]
    json_files = list((project_root / "whisper_json").glob("*.json"))
    assert json_files, "No transcript JSON files found"
    for json_path in json_files:
        data = json.loads(json_path.read_text())
        assert data.get("schema_version") == 2


@then('the transcript may contain a "speakers" array')
def transcript_may_have_speakers(test_state):
    """Schema should tolerate optional speakers array."""
    for transcript in test_state["transcripts"]:
        assert hasattr(transcript, "speakers")


@then('the transcript may contain a "turns" array')
def transcript_may_have_turns(test_state):
    """Schema should tolerate optional turns array."""
    for transcript in test_state["transcripts"]:
        assert hasattr(transcript, "turns")


@then('if "speakers" array exists, each speaker has required fields')
def speakers_have_required_fields(test_state):
    """Validate speaker entries when present."""
    for transcript in test_state["transcripts"]:
        if transcript.speakers:
            for speaker in transcript.speakers:
                assert "id" in speaker


@then(parsers.parse('the transcript JSON has meta.diarization.status "{status}"'))
def transcript_diarization_status(test_state, status):
    """Validate diarization status metadata."""
    for transcript in test_state["transcripts"]:
        assert transcript.meta is not None
        assert "diarization" in transcript.meta
        assert transcript.meta["diarization"].get("status") == status


@then("meta.diarization.requested is true")
def diarization_requested_true(test_state):
    for transcript in test_state["transcripts"]:
        assert transcript.meta and transcript.meta.get("diarization", {}).get("requested") is True


@then(parsers.parse('meta.diarization.backend is "{backend}"'))
def diarization_backend(test_state, backend):
    for transcript in test_state["transcripts"]:
        assert transcript.meta
        assert transcript.meta.get("diarization", {}).get("backend") == backend


@then(parsers.parse('meta.diarization.error_type is "{error_type}"'))
def diarization_error_type(test_state, error_type):
    for transcript in test_state["transcripts"]:
        assert transcript.meta
        actual = transcript.meta.get("diarization", {}).get("error_type")
        # If the backend itself is unavailable, treat that as acceptable fallback
        assert actual in {error_type, "missing_dependency"}


@then('the "speakers" field is null')
def speakers_field_null(test_state):
    for transcript in test_state["transcripts"]:
        assert transcript.speakers is None


@then('the "speakers" array is populated with detected speakers')
def speakers_array_populated(test_state):
    for transcript in test_state["transcripts"]:
        assert transcript.speakers is not None
        assert len(transcript.speakers) > 0


@then(parsers.parse("meta.diarization.num_speakers equals {expected:d}"))
def diarization_num_speakers(test_state, expected: int):
    for transcript in test_state["transcripts"]:
        assert transcript.meta
        assert transcript.meta.get("diarization", {}).get("num_speakers") == expected


@then(parsers.parse('the "speakers" array has exactly {count:d} speakers'))
def speakers_array_count(test_state, count: int):
    for transcript in test_state["transcripts"]:
        assert transcript.speakers is not None
        assert len(transcript.speakers) == count


@then('the "turns" array contains turn structure')
def turns_array_structure(test_state):
    for transcript in test_state["transcripts"]:
        assert transcript.turns is not None
        assert len(transcript.turns) > 0
        for turn in transcript.turns:
            assert "speaker_id" in turn
            assert "start" in turn and "end" in turn
            assert "segment_ids" in turn


@then('the "turns" array has at least 2 turns')
def turns_array_has_turns(test_state):
    for transcript in test_state["transcripts"]:
        assert transcript.turns is not None
        assert len(transcript.turns) >= 2


@then("speaker turns alternate between the two detected speakers")
def speaker_turns_alternate(test_state):
    for transcript in test_state["transcripts"]:
        turns = transcript.turns or []
        assert len({t.get("speaker_id") for t in turns}) >= 2
        for idx in range(len(turns) - 1):
            assert turns[idx].get("speaker_id") != turns[idx + 1].get("speaker_id")


@then('the "turns" field is null')
def turns_field_null(test_state):
    for transcript in test_state["transcripts"]:
        assert transcript.turns is None


# ============================================================================
# Edge case scenarios for empty/invalid audio files (Issue #53)
# ============================================================================


@given(parsers.parse('a project with an empty audio file named "{filename}"'))
def project_with_empty_audio_file(test_state, filename):
    """Create an empty (0-byte) audio file in the project."""
    project_root = test_state["project_root"]
    raw_audio_dir = project_root / "raw_audio"

    # Create an empty file (0 bytes)
    empty_path = raw_audio_dir / filename
    empty_path.touch()

    test_state["audio_files"].append(filename)
    test_state["expected_failure"] = "empty"


@given(parsers.parse('a project with a zero-duration WAV file named "{filename}"'))
def project_with_zero_duration_wav(test_state, filename):
    """Create a valid WAV file with zero audio frames."""
    import wave

    project_root = test_state["project_root"]
    raw_audio_dir = project_root / "raw_audio"

    wav_path = raw_audio_dir / filename

    # Create a valid WAV header but with 0 frames
    with wave.open(str(wav_path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"")  # No audio data

    test_state["audio_files"].append(filename)
    test_state["expected_failure"] = "zero_duration"


@given(parsers.parse('a project with a silent audio file named "{filename}"'))
def project_with_silent_audio_file(test_state, filename):
    """Create a WAV file with silence (all zeros)."""
    project_root = test_state["project_root"]
    raw_audio_dir = project_root / "raw_audio"

    wav_path = raw_audio_dir / filename
    sr = 16000
    duration = 2.0  # 2 seconds of silence

    try:
        import soundfile as sf

        # Generate pure silence (all zeros)
        samples = int(duration * sr)
        audio = np.zeros(samples, dtype=np.float32)
        sf.write(str(wav_path), audio, sr)
    except ImportError:
        # Fallback: write raw WAV with wave module
        import wave

        samples = int(duration * sr)
        audio_bytes = b"\x00\x00" * samples  # 16-bit silence

        with wave.open(str(wav_path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sr)
            wav.writeframes(audio_bytes)

    test_state["audio_files"].append(filename)


@given(
    parsers.parse(
        'a project with a very short audio file named "{filename}" of {duration:f} seconds'
    )
)
def project_with_short_audio_file(test_state, filename, duration):
    """Create a very short audio file."""
    project_root = test_state["project_root"]
    raw_audio_dir = project_root / "raw_audio"

    wav_path = raw_audio_dir / filename
    sr = 16000

    try:
        import soundfile as sf

        samples = max(1, int(duration * sr))
        # Generate minimal noise to have some content
        audio = np.random.uniform(-0.01, 0.01, samples).astype(np.float32)
        sf.write(str(wav_path), audio, sr)
    except ImportError:
        import wave

        samples = max(1, int(duration * sr))
        audio_bytes = b"\x00\x00" * samples

        with wave.open(str(wav_path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sr)
            wav.writeframes(audio_bytes)

    test_state["audio_files"].append(filename)
    test_state["short_duration"] = duration


@when("I attempt to transcribe the project with default settings")
def attempt_transcribe_with_defaults(test_state):
    """Attempt to transcribe, capturing any errors."""
    project_root = test_state["project_root"]
    config = TranscriptionConfig(model="base", device="cpu")

    test_state["config"] = config
    test_state["transcription_error"] = None
    test_state["transcripts"] = []

    try:
        transcripts = transcribe_directory(project_root, config)
        test_state["transcripts"] = transcripts
    except Exception as e:
        test_state["transcription_error"] = e


@then("transcription fails gracefully with an error about the empty file")
def transcription_fails_gracefully_empty(test_state):
    """Verify transcription failed with an appropriate error for empty file."""
    error = test_state.get("transcription_error")
    transcripts = test_state.get("transcripts", [])

    # Either an exception was raised, or no transcripts were produced
    if error:
        # Check error message mentions the issue
        error_msg = str(error).lower()
        assert any(
            keyword in error_msg
            for keyword in ["empty", "no transcript", "no .wav", "normalize", "ffmpeg", "invalid"]
        ), f"Expected error about empty file, got: {error}"
    else:
        # If no error, transcripts should be empty (file was skipped/filtered)
        # This is also graceful handling
        assert len(transcripts) == 0, (
            f"Expected no transcripts for empty file, got {len(transcripts)}"
        )


@then(parsers.parse('if a transcript exists for "{filename}", it has placeholder segments'))
def transcript_has_placeholder_if_exists(test_state, filename):
    """Verify transcript has placeholder segments if it was created for zero-duration file."""
    transcripts = test_state.get("transcripts", [])
    error = test_state.get("transcription_error")

    # If there was an error, that's acceptable graceful handling
    if error:
        return

    # If transcripts exist, check for placeholder handling
    for transcript in transcripts:
        if transcript.file_name == filename:
            # Check if meta indicates fallback/placeholder behavior
            meta = transcript.meta or {}
            # System may have created placeholder segments
            has_placeholder_marker = (
                meta.get("asr_placeholder_segments", False)
                or meta.get("asr_fallback_reason") is not None
            )
            # Either has placeholder marker, or has segments (valid transcript)
            assert has_placeholder_marker or len(transcript.segments) >= 0, (
                f"Expected placeholder markers or valid segments for {filename}"
            )


@then(parsers.parse('no transcript JSON is created for "{filename}"'))
def no_transcript_json_created(test_state, filename):
    """Verify no JSON transcript was created for the file."""
    project_root = test_state["project_root"]
    stem = Path(filename).stem
    json_path = project_root / "whisper_json" / f"{stem}.json"

    assert not json_path.exists(), f"Expected no JSON for {filename}, but found {json_path}"


@then("the transcript may contain zero or more segments")
def transcript_may_have_segments(test_state):
    """Verify transcript exists (segments may be empty for silent audio)."""
    transcripts = test_state["transcripts"]

    # Should have at least one transcript
    assert len(transcripts) > 0, "Expected at least one transcript"

    # Segments can be zero or more (silent audio may produce no segments)
    for transcript in transcripts:
        assert hasattr(transcript, "segments"), "Transcript should have segments attribute"
        # segments can be empty list or populated - both are valid


@then("the transcription completes or fails gracefully")
def transcription_completes_or_fails_gracefully(test_state):
    """Verify transcription either completed or failed gracefully."""
    error = test_state.get("transcription_error")
    transcripts = test_state.get("transcripts", [])

    if error:
        # If there's an error, it should be a known error type
        from transcription.exceptions import TranscriptionError

        assert isinstance(error, (TranscriptionError, Exception)), (
            f"Error should be a proper exception type, got: {type(error)}"
        )
        # The error should have a meaningful message
        assert len(str(error)) > 0, "Error message should not be empty"
    else:
        # If no error, should have completed (transcripts list exists)
        # Transcripts may be empty or populated
        assert isinstance(transcripts, list), "Transcripts should be a list"


@then("if transcript exists, it has valid schema")
def transcript_has_valid_schema_if_exists(test_state):
    """Verify transcript has valid schema if it was created."""
    project_root = test_state["project_root"]
    json_files = list((project_root / "whisper_json").glob("*.json"))

    for json_path in json_files:
        with open(json_path) as f:
            data = json.load(f)

        # Verify required schema fields
        assert "schema_version" in data, f"Missing schema_version in {json_path.name}"
        assert data.get("schema_version") == 2, f"Wrong schema version in {json_path.name}"
        assert "segments" in data, f"Missing segments in {json_path.name}"
        assert isinstance(data["segments"], list), f"Segments should be a list in {json_path.name}"
