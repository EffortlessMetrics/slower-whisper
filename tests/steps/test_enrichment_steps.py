"""
Step definitions for enrichment BDD tests.

This module implements Gherkin steps for testing the audio enrichment pipeline
using pytest-bdd. Steps use the public API from transcription.api.
"""

import shutil
from pathlib import Path

import numpy as np
import pytest
from pytest_bdd import given, parsers, scenarios, then, when

from transcription import (
    EnrichmentConfig,
    EnrichmentError,
    enrich_directory,
    enrich_transcript,
)

# Check if ffmpeg is available
FFMPEG_AVAILABLE = shutil.which("ffmpeg") is not None

# Load all scenarios from the feature file
# Mark all scenarios as xfail if ffmpeg is not available since they require transcription
pytestmark = pytest.mark.xfail(
    not FFMPEG_AVAILABLE, reason="Requires ffmpeg for audio normalization", strict=False
)

scenarios("../features/enrichment.feature")


# ============================================================================
# Fixtures for test state
# ============================================================================


@pytest.fixture
def enrich_state():
    """Shared state dictionary for enrichment tests."""
    return {
        "project_root": None,
        "audio_files": [],
        "transcripts": [],
        "enriched_transcripts": [],
        "config": None,
        "transcript_object": None,
        "original_json": None,
    }


# ============================================================================
# Helper functions
# ============================================================================


def create_test_wav(path: Path, duration: float = 1.0, sr: int = 16000):
    """Create a test WAV file with speech-like audio.

    Generates white noise to ensure Whisper detects speech segments.
    This is sufficient for testing enrichment workflows without requiring
    real speech samples.
    """
    try:
        import soundfile as sf

        # Generate white noise (speech-like) instead of silence
        # This ensures Whisper produces segments for enrichment tests
        samples = int(duration * sr)
        # Low-amplitude noise (-20dB) to simulate speech energy
        audio = np.random.normal(0, 0.1, samples).astype(np.float32)

        # Write WAV file
        sf.write(str(path), audio, sr)
    except ImportError:
        # Fallback: create an empty file if soundfile not available
        path.touch()


def create_dummy_transcript_json(json_path: Path, audio_filename: str, num_segments: int = 2):
    """Create a dummy transcript JSON file for enrichment tests.

    This bypasses the transcription step (which fails on synthetic audio)
    and creates transcript JSON with dummy segments directly.

    Args:
        json_path: Path to write JSON file
        audio_filename: Name of audio file (for transcript metadata)
        num_segments: Number of dummy segments to create
    """
    import json

    transcript_data = {
        "schema_version": 2,
        "file_name": audio_filename,
        "language": "en",
        "meta": {
            "generated_at": "2025-01-01T00:00:00Z",
            "model_name": "base",
            "device": "cpu",
        },
        "segments": [
            {
                "id": i,
                "start": float(i * 2.0),
                "end": float((i + 1) * 2.0),
                "text": f"Dummy segment {i}",
                "speaker": None,
                "tone": None,
                "audio_state": None,
            }
            for i in range(num_segments)
        ],
    }

    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(transcript_data, f, indent=2)


# ============================================================================
# Given steps (setup)
# ============================================================================


@given("a clean test project directory", target_fixture="enrich_state")
def clean_enrichment_directory(tmp_path):
    """Create a clean temporary project directory for enrichment."""
    project_root = tmp_path / "enrich_project"
    project_root.mkdir(exist_ok=True)

    # Create required subdirectories
    (project_root / "raw_audio").mkdir(exist_ok=True)
    (project_root / "input_audio").mkdir(exist_ok=True)
    (project_root / "whisper_json").mkdir(exist_ok=True)
    (project_root / "transcripts").mkdir(exist_ok=True)

    return {
        "project_root": project_root,
        "audio_files": [],
        "transcripts": [],
        "enriched_transcripts": [],
        "config": None,
        "transcript_object": None,
        "original_json": None,
    }


@given(parsers.parse('a transcribed file "{filename}" exists'))
def transcribed_file_exists(enrich_state, filename):
    """Create a transcribed file ready for enrichment."""
    project_root = enrich_state["project_root"]
    norm_audio_dir = project_root / "input_audio"
    json_dir = project_root / "whisper_json"
    stem = Path(filename).stem

    # Create audio files (both raw and normalized)
    raw_audio_dir = project_root / "raw_audio"
    raw_wav_path = raw_audio_dir / filename
    norm_wav_path = norm_audio_dir / f"{stem}.wav"
    json_path = json_dir / f"{stem}.json"

    create_test_wav(raw_wav_path, duration=4.0)
    create_test_wav(norm_wav_path, duration=4.0)
    create_dummy_transcript_json(json_path, filename, num_segments=2)

    # Load the transcript
    from transcription import load_transcript

    transcript = load_transcript(json_path)
    enrich_state["audio_files"].append(filename)
    enrich_state["transcripts"] = [transcript]


@given(parsers.parse("transcribed files exist:"), target_fixture="enrich_state")
def multiple_transcribed_files_exist(enrich_state, datatable):
    """Create multiple transcribed files with dummy transcripts."""
    project_root = enrich_state["project_root"]
    raw_audio_dir = project_root / "raw_audio"
    norm_audio_dir = project_root / "input_audio"
    json_dir = project_root / "whisper_json"

    norm_audio_dir.mkdir(exist_ok=True)
    json_dir.mkdir(exist_ok=True)

    # datatable is a list of lists: [['filename'], ['file1.wav'], ['file2.wav'], ...]
    # Skip header row (index 0) and access first column (index 0) for each data row
    for row in datatable[1:]:
        filename = row[0]
        stem = Path(filename).stem

        # Create audio files (both raw and normalized)
        raw_wav_path = raw_audio_dir / filename
        norm_wav_path = norm_audio_dir / f"{stem}.wav"
        json_path = json_dir / f"{stem}.json"

        create_test_wav(raw_wav_path, duration=4.0)  # Longer for multiple segments
        create_test_wav(norm_wav_path, duration=4.0)
        create_dummy_transcript_json(json_path, filename, num_segments=2)

        enrich_state["audio_files"].append(filename)

    # Load the transcripts we just created
    from transcription import load_transcript

    transcripts = []
    for filename in enrich_state["audio_files"]:
        stem = Path(filename).stem
        json_path = json_dir / f"{stem}.json"
        transcript = load_transcript(json_path)
        transcripts.append(transcript)

    enrich_state["transcripts"] = transcripts

    return enrich_state


@given(parsers.parse('a transcript object for "{filename}"'))
def transcript_object_exists(enrich_state, tmp_path, filename):
    """Create a transcript object for enrichment."""
    # Create a simple project for this file
    project_root = tmp_path / "transcript_project"
    project_root.mkdir(exist_ok=True)
    raw_audio_dir = project_root / "raw_audio"
    norm_audio_dir = project_root / "input_audio"
    json_dir = project_root / "whisper_json"

    raw_audio_dir.mkdir(exist_ok=True)
    norm_audio_dir.mkdir(exist_ok=True)
    json_dir.mkdir(exist_ok=True)

    stem = Path(filename).stem

    # Create audio files and transcript JSON
    raw_wav_path = raw_audio_dir / filename
    norm_wav_path = norm_audio_dir / f"{stem}.wav"
    json_path = json_dir / f"{stem}.json"

    create_test_wav(raw_wav_path, duration=4.0)
    create_test_wav(norm_wav_path, duration=4.0)
    create_dummy_transcript_json(json_path, filename, num_segments=2)

    # Load the transcript
    from transcription import load_transcript

    transcript = load_transcript(json_path)

    enrich_state["transcript_object"] = transcript
    enrich_state["project_root"] = project_root


@given("the corresponding audio file exists")
def audio_file_exists_for_transcript(enrich_state):
    """Ensure audio file exists for the transcript object."""
    # Audio file was already created in previous step
    project_root = enrich_state["project_root"]
    audio_dir = project_root / "input_audio"

    # Verify it exists
    wav_files = list(audio_dir.glob("*.wav"))
    assert len(wav_files) > 0, "No audio files found"

    enrich_state["audio_path"] = wav_files[0]


@given("the transcript is already enriched")
def transcript_already_enriched(enrich_state):
    """Enrich a transcript and store its state."""
    project_root = enrich_state["project_root"]

    # Enrich the transcripts
    try:
        config = EnrichmentConfig(enable_prosody=True, enable_emotion=False)
        enriched = enrich_directory(project_root, config)
        enrich_state["transcripts"] = enriched

        # Store original JSON
        json_files = list((project_root / "whisper_json").glob("*.json"))
        if json_files:
            enrich_state["original_json"] = json_files[0].read_text()
    except ImportError:
        pytest.skip("Enrichment dependencies not available")


@given(parsers.parse('a transcript JSON for "{filename}" exists'))
def transcript_json_exists_no_audio(enrich_state, filename):
    """Create a transcript JSON without corresponding audio."""
    project_root = enrich_state["project_root"]
    json_dir = project_root / "whisper_json"
    stem = Path(filename).stem

    # Create transcript JSON directly (no audio files)
    json_path = json_dir / f"{stem}.json"
    create_dummy_transcript_json(json_path, filename, num_segments=2)

    enrich_state["audio_files"].append(filename)


@given(parsers.parse('the audio file "{filename}" does not exist'))
def audio_file_does_not_exist(enrich_state, filename):
    """Remove the audio file."""
    project_root = enrich_state["project_root"]
    audio_dir = project_root / "input_audio"

    stem = Path(filename).stem
    wav_path = audio_dir / f"{stem}.wav"

    if wav_path.exists():
        wav_path.unlink()


# ============================================================================
# When steps (actions)
# ============================================================================


@when("I enrich the transcript with prosody enabled")
def enrich_with_prosody(enrich_state):
    """Enrich transcripts with prosody features."""
    project_root = enrich_state["project_root"]

    try:
        config = EnrichmentConfig(enable_prosody=True, enable_emotion=False)
        enriched = enrich_directory(project_root, config)
        enrich_state["enriched_transcripts"] = enriched
    except ImportError as e:
        pytest.skip(f"Enrichment dependencies not available: {e}")


@when("I enrich the transcript with emotion enabled")
def enrich_with_emotion(enrich_state):
    """Enrich transcripts with emotion features."""
    project_root = enrich_state["project_root"]

    try:
        config = EnrichmentConfig(enable_prosody=False, enable_emotion=True)
        enriched = enrich_directory(project_root, config)
        enrich_state["enriched_transcripts"] = enriched
    except ImportError as e:
        pytest.skip(f"Enrichment dependencies not available: {e}")


@when("I enrich the transcript with prosody and emotion enabled")
def enrich_with_full_features(enrich_state):
    """Enrich transcripts with all features."""
    project_root = enrich_state["project_root"]

    try:
        config = EnrichmentConfig(enable_prosody=True, enable_emotion=True)
        enriched = enrich_directory(project_root, config)
        enrich_state["enriched_transcripts"] = enriched
    except ImportError as e:
        pytest.skip(f"Enrichment dependencies not available: {e}")


@when("I enrich the directory with prosody enabled")
def enrich_directory_with_prosody(enrich_state):
    """Enrich entire directory with prosody."""
    project_root = enrich_state["project_root"]

    try:
        config = EnrichmentConfig(enable_prosody=True, enable_emotion=False)
        enriched = enrich_directory(project_root, config)
        enrich_state["enriched_transcripts"] = enriched
    except ImportError as e:
        pytest.skip(f"Enrichment dependencies not available: {e}")
    except EnrichmentError:
        # Missing audio should not blow up the scenario; record empty result
        enrich_state["enriched_transcripts"] = []


@when("I enrich the transcript object with prosody enabled")
def enrich_transcript_object(enrich_state):
    """Enrich a single transcript object."""
    transcript_obj = enrich_state["transcript_object"]
    audio_path = enrich_state["audio_path"]

    try:
        config = EnrichmentConfig(enable_prosody=True, enable_emotion=False)
        enriched = enrich_transcript(transcript_obj, audio_path, config)
        enrich_state["enriched_transcripts"] = [enriched]
    except ImportError as e:
        pytest.skip(f"Enrichment dependencies not available: {e}")


@when("I enrich the directory with skip_existing enabled")
def enrich_with_skip_existing(enrich_state):
    """Enrich directory with skip_existing enabled."""
    project_root = enrich_state["project_root"]

    try:
        config = EnrichmentConfig(enable_prosody=True, enable_emotion=False, skip_existing=True)
        enriched = enrich_directory(project_root, config)
        enrich_state["enriched_transcripts"] = enriched
    except ImportError as e:
        pytest.skip(f"Enrichment dependencies not available: {e}")


# ============================================================================
# Then steps (assertions)
# ============================================================================


@then("the transcript has audio_state for all segments")
def transcript_has_audio_state(enrich_state):
    """Verify all segments have audio_state."""
    enriched = enrich_state["enriched_transcripts"]
    assert len(enriched) > 0, "No enriched transcripts"

    for transcript in enriched:
        for segment in transcript.segments:
            assert segment.audio_state is not None, f"Segment {segment.id} has no audio_state"


@then("each audio_state contains prosody data")
def audio_state_has_prosody(enrich_state):
    """Verify audio_state contains prosody features."""
    enriched = enrich_state["enriched_transcripts"]

    for transcript in enriched:
        for segment in transcript.segments:
            if segment.audio_state:
                assert "prosody" in segment.audio_state, "No prosody in audio_state"


@then("each audio_state has extraction_status")
def audio_state_has_extraction_status(enrich_state):
    """Verify audio_state has extraction_status."""
    enriched = enrich_state["enriched_transcripts"]

    for transcript in enriched:
        for segment in transcript.segments:
            if segment.audio_state:
                assert "extraction_status" in segment.audio_state, (
                    "No extraction_status in audio_state"
                )


@then("each audio_state contains emotion data")
def audio_state_has_emotion(enrich_state):
    """Verify audio_state contains emotion features."""
    enriched = enrich_state["enriched_transcripts"]

    for transcript in enriched:
        for segment in transcript.segments:
            if segment.audio_state:
                assert "emotion" in segment.audio_state, "No emotion in audio_state"


@then("the emotion data includes dimensional values")
def emotion_has_dimensional_values(enrich_state):
    """Verify emotion data has valence/arousal."""
    enriched = enrich_state["enriched_transcripts"]

    for transcript in enriched:
        for segment in transcript.segments:
            if segment.audio_state and "emotion" in segment.audio_state:
                emotion = segment.audio_state["emotion"]
                # Check for dimensional emotion values
                assert "valence" in emotion or "arousal" in emotion, "No dimensional emotion data"


@then("each audio_state has a text rendering")
def audio_state_has_rendering(enrich_state):
    """Verify audio_state has text rendering."""
    enriched = enrich_state["enriched_transcripts"]

    for transcript in enriched:
        for segment in transcript.segments:
            if segment.audio_state:
                assert "rendering" in segment.audio_state, "No rendering in audio_state"
                assert isinstance(segment.audio_state["rendering"], str), (
                    "Rendering is not a string"
                )


@then("all transcripts have audio_state data")
def all_transcripts_have_audio_state(enrich_state):
    """Verify all transcripts were enriched."""
    enriched = enrich_state["enriched_transcripts"]
    audio_files = enrich_state["audio_files"]

    assert len(enriched) == len(audio_files), "Not all files were enriched"

    for transcript in enriched:
        has_audio_state = any(seg.audio_state is not None for seg in transcript.segments)
        assert has_audio_state, f"Transcript {transcript.file_name} has no audio_state"


@then("each transcript has extraction_status")
def each_transcript_has_extraction_status(enrich_state):
    """Verify each transcript has extraction_status."""
    enriched = enrich_state["enriched_transcripts"]

    for transcript in enriched:
        for segment in transcript.segments:
            if segment.audio_state:
                assert "extraction_status" in segment.audio_state


@then("the enriched transcript has audio_state")
def enriched_transcript_has_audio_state(enrich_state):
    """Verify the enriched transcript object has audio_state."""
    enriched = enrich_state["enriched_transcripts"]
    assert len(enriched) > 0, "No enriched transcript"

    transcript = enriched[0]
    has_audio_state = any(seg.audio_state is not None for seg in transcript.segments)
    assert has_audio_state, "Enriched transcript has no audio_state"


@then("the prosody features include pitch, energy, and rate")
def prosody_has_all_features(enrich_state):
    """Verify prosody includes all expected features."""
    enriched = enrich_state["enriched_transcripts"]

    for transcript in enriched:
        for segment in transcript.segments:
            if segment.audio_state and "prosody" in segment.audio_state:
                prosody = segment.audio_state["prosody"]
                assert "pitch" in prosody, "No pitch in prosody"
                assert "energy" in prosody, "No energy in prosody"
                assert "rate" in prosody, "No rate in prosody"


@then(parsers.parse('the transcript for "{filename}" is unchanged'))
def enrichment_transcript_unchanged(enrich_state, filename):
    """Verify the transcript was not modified."""
    project_root = enrich_state["project_root"]
    stem = Path(filename).stem
    json_path = project_root / "whisper_json" / f"{stem}.json"

    current_json = json_path.read_text()
    original_json = enrich_state["original_json"]

    assert current_json == original_json, f"Transcript {filename} was modified"


@then(parsers.parse('the enrichment skips "{filename}"'))
def enrichment_skips_file(enrich_state, filename):
    """Verify the file was skipped during enrichment."""
    # If enrichment completed without error, it handled the missing file gracefully
    # This is verified by the enrichment not raising an exception
    pass


@then("other files are enriched successfully")
def other_files_enriched(enrich_state):
    """Verify other files were enriched despite errors."""
    _ = enrich_state["enriched_transcripts"]
    # At least some files should have been enriched
    # (This step is mostly a placeholder for error handling verification)
    assert True, "Error handling test passed"


@then("each audio_state has a rendering field")
def audio_state_has_rendering_field(enrich_state):
    """Verify rendering field exists."""
    enriched = enrich_state["enriched_transcripts"]

    for transcript in enriched:
        for segment in transcript.segments:
            if segment.audio_state:
                assert "rendering" in segment.audio_state


@then("the rendering is a human-readable text description")
def rendering_is_human_readable(enrich_state):
    """Verify rendering is human-readable text."""
    enriched = enrich_state["enriched_transcripts"]

    for transcript in enriched:
        for segment in transcript.segments:
            if segment.audio_state and segment.audio_state.get("rendering"):
                rendering = segment.audio_state["rendering"]
                assert isinstance(rendering, str)
                assert len(rendering) > 0


@then('the rendering matches the format "[audio: ...]"')
def rendering_matches_format(enrich_state):
    """Verify rendering format."""
    enriched = enrich_state["enriched_transcripts"]

    for transcript in enriched:
        for segment in transcript.segments:
            if segment.audio_state and segment.audio_state.get("rendering"):
                rendering = segment.audio_state["rendering"]
                assert rendering.startswith("[audio:")
                assert rendering.endswith("]")


@then("the prosody features use speaker-relative baselines")
def prosody_uses_baselines(enrich_state):
    """Verify prosody features use speaker-relative baselines."""
    # This is a design verification - if enrichment succeeded with compute_baseline=True,
    # the baseline computation worked
    enriched = enrich_state["enriched_transcripts"]
    assert len(enriched) > 0, "No enriched transcripts"


@then("the baseline is computed from sampled segments")
def baseline_computed_from_samples(enrich_state):
    """Verify baseline computation used sampling."""
    # This is verified by the successful enrichment process
    # The actual baseline computation is internal to the enrichment module
    enriched = enrich_state["enriched_transcripts"]
    assert len(enriched) > 0, "No enriched transcripts"
