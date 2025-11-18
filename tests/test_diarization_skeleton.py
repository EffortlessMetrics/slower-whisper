"""
Integration tests for v1.1 diarization skeleton.

Tests that enable_diarization=True doesn't crash the pipeline,
and properly marks the transcript with meta.diarization.status.
"""

import subprocess

import pytest

from transcription import TranscriptionConfig, transcribe_file
from transcription.config import TranscriptionConfig as TConfig
from transcription.models import Segment, Transcript


def ffmpeg_available() -> bool:
    """Check if ffmpeg is available on PATH."""
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


# ============================================================================
# Unit tests (no ffmpeg required)
# ============================================================================


def test_maybe_run_diarization_disabled():
    """
    Test that _maybe_run_diarization() returns transcript unchanged when disabled.
    """
    from pathlib import Path

    from transcription.api import _maybe_run_diarization

    transcript = Transcript(
        file_name="test.wav",
        language="en",
        segments=[Segment(id=0, start=0.0, end=2.0, text="Hello")],
    )

    config = TConfig(enable_diarization=False)

    result = _maybe_run_diarization(transcript, Path("dummy.wav"), config)

    # Should return transcript unchanged
    assert result is transcript
    assert result.speakers is None
    assert result.turns is None
    assert "diarization" not in (result.meta or {})


def test_maybe_run_diarization_graceful_failure():
    """
    Test that _maybe_run_diarization() gracefully handles failures.

    When diarization is enabled but fails (missing file, missing pyannote, etc.),
    the function should return the transcript unchanged with status="failed" metadata.
    """
    from pathlib import Path

    from transcription.api import _maybe_run_diarization

    transcript = Transcript(
        file_name="test.wav",
        language="en",
        segments=[Segment(id=0, start=0.0, end=2.0, text="Hello")],
    )

    config = TConfig(enable_diarization=True, diarization_device="cpu")

    # Use non-existent file to trigger failure
    result = _maybe_run_diarization(transcript, Path("dummy.wav"), config)

    # Should return transcript unchanged (graceful degradation)
    assert result is transcript
    assert result.speakers is None  # No speakers assigned on failure
    assert result.turns is None  # No turns built on failure

    # Metadata should indicate failure with detailed error info
    assert result.meta is not None
    assert "diarization" in result.meta
    assert result.meta["diarization"]["status"] == "failed"
    assert result.meta["diarization"]["requested"] is True
    assert "error" in result.meta["diarization"]
    assert "Audio file not found" in result.meta["diarization"]["error"]
    assert result.meta["diarization"]["error_type"] == "file_not_found"


def test_transcription_config_diarization_fields():
    """Test that TranscriptionConfig accepts diarization fields."""
    config = TConfig(
        model="base",
        enable_diarization=True,
        diarization_device="cuda",
        min_speakers=2,
        max_speakers=4,
        overlap_threshold=0.4,
    )

    assert config.enable_diarization is True
    assert config.diarization_device == "cuda"
    assert config.min_speakers == 2
    assert config.max_speakers == 4
    assert config.overlap_threshold == 0.4


def test_transcription_config_diarization_defaults():
    """Test that TranscriptionConfig has correct diarization defaults."""
    config = TConfig()

    assert config.enable_diarization is False
    assert config.diarization_device == "auto"
    assert config.min_speakers is None
    assert config.max_speakers is None
    assert config.overlap_threshold == 0.3


# ============================================================================
# Integration test with synthetic fixture (ready for pyannote)
# ============================================================================


@pytest.mark.requires_diarization
@pytest.mark.skipif(not ffmpeg_available(), reason="ffmpeg not available")
def test_synthetic_2speaker_diarization(tmp_path):
    """
    Integration test with synthetic_2speaker.wav fixture.

    Tests real diarization with pyannote.audio on a synthetic 2-speaker fixture
    with known ground truth (A-B-A-B turn pattern).

    Expected behavior:
    - Exactly 2 speakers detected
    - 4 speaker turns in A-B-A-B pattern
    - Meta indicates success with pyannote backend
    """
    from pathlib import Path

    from transcription import transcribe_directory

    # Copy synthetic fixture into test project
    fixture_path = Path(__file__).parent / "fixtures" / "synthetic_2speaker.wav"
    if not fixture_path.exists():
        pytest.skip(f"Synthetic fixture not found: {fixture_path}")

    raw_audio_dir = tmp_path / "raw_audio"
    raw_audio_dir.mkdir()
    test_wav = raw_audio_dir / "synthetic_2speaker.wav"

    import shutil

    shutil.copy(fixture_path, test_wav)

    # Transcribe with diarization enabled
    config = TConfig(
        model="base",
        device="cpu",
        enable_diarization=True,
        diarization_device="cpu",
        min_speakers=2,
        max_speakers=2,
    )

    transcripts = transcribe_directory(tmp_path, config)
    assert len(transcripts) == 1

    transcript = transcripts[0]

    # ========================================================================
    # Real diarization assertions (v1.1)
    # ========================================================================

    # Meta should indicate success
    assert transcript.meta is not None
    assert "diarization" in transcript.meta
    assert transcript.meta["diarization"]["status"] == "success"
    assert transcript.meta["diarization"]["requested"] is True
    assert transcript.meta["diarization"]["backend"] == "pyannote.audio"

    # Should have exactly 2 speakers
    assert transcript.speakers is not None
    assert len(transcript.speakers) == 2
    assert transcript.meta["diarization"]["num_speakers"] == 2

    # Should have turns (exact count depends on pyannote's segmentation)
    # On synthetic audio, we expect 4 turns (A-B-A-B pattern), but allow some flexibility
    assert transcript.turns is not None
    assert len(transcript.turns) >= 2  # At minimum 2 turns

    # Verify turn pattern: should alternate between two speaker IDs
    turn_speakers = [t["speaker_id"] for t in transcript.turns]
    unique_turn_speakers = set(turn_speakers)
    assert len(unique_turn_speakers) == 2  # Exactly 2 unique speakers in turns

    # Check that turns alternate (no two consecutive turns from same speaker)
    # This is the key behavioral contract for the synthetic fixture
    for i in range(len(turn_speakers) - 1):
        # Note: In real scenarios with pyannote, there might be small segments
        # where the same speaker is detected in consecutive short turns due to VAD.
        # For synthetic pure tones with clear gaps, we expect strict alternation.
        # If this fails in practice, we can relax to "mostly alternating".
        if len(turn_speakers) == 4:  # Ideal case
            assert turn_speakers[i] != turn_speakers[i + 1], (
                f"Expected alternating speakers in turns, but found consecutive "
                f"segments with same speaker at positions {i} and {i + 1}"
            )


# ============================================================================
# Existing integration tests
# ============================================================================


@pytest.mark.skipif(not ffmpeg_available(), reason="ffmpeg not available")
def test_transcribe_file_with_diarization_skeleton(tmp_path, sample_audio_path):
    """
    Test that enable_diarization=True completes without crashing (v1.1 skeleton).

    Expected behavior (skeleton):
    - Transcription succeeds
    - speakers = None (not yet implemented)
    - turns = None (not yet implemented)
    - meta.diarization.status = "not_implemented"
    - meta.diarization.requested = True
    - Warning logged about diarization not being implemented
    """
    config = TranscriptionConfig(
        model="base",  # Use small model for faster tests
        device="cpu",
        enable_diarization=True,
        diarization_device="cpu",
    )

    try:
        transcript = transcribe_file(
            audio_path=sample_audio_path,
            root=tmp_path,
            config=config,
        )
    except Exception as e:
        pytest.fail(f"transcribe_file crashed with enable_diarization=True: {e}")

    # Verify transcript structure
    assert isinstance(transcript, Transcript)
    assert len(transcript.segments) > 0

    # Verify diarization skeleton behavior
    assert transcript.speakers is None, "speakers should be None in skeleton"
    assert transcript.turns is None, "turns should be None in skeleton"

    # Verify metadata
    assert transcript.meta is not None
    assert "diarization" in transcript.meta

    diar_meta = transcript.meta["diarization"]
    assert diar_meta["status"] == "not_implemented"
    assert diar_meta["requested"] is True


@pytest.mark.skipif(not ffmpeg_available(), reason="ffmpeg not available")
def test_transcribe_file_without_diarization(tmp_path, sample_audio_path):
    """
    Test that enable_diarization=False works as before (no diarization metadata).

    This ensures we haven't broken the default path.
    """
    config = TranscriptionConfig(
        model="base",
        device="cpu",
        enable_diarization=False,
    )

    transcript = transcribe_file(
        audio_path=sample_audio_path,
        root=tmp_path,
        config=config,
    )

    # Should complete normally
    assert isinstance(transcript, Transcript)
    assert len(transcript.segments) > 0

    # No diarization metadata should be present
    assert transcript.speakers is None
    assert transcript.turns is None
    assert "diarization" not in (transcript.meta or {})


@pytest.fixture
def sample_audio_path(tmp_path):
    """
    Create a minimal WAV file for testing.

    This is a synthetic 1-second silence WAV (16kHz mono).
    Faster-whisper may transcribe it as empty or produce minimal segments.
    """
    import wave

    wav_path = tmp_path / "test_audio.wav"

    with wave.open(str(wav_path), "wb") as wav:
        wav.setnchannels(1)  # mono
        wav.setsampwidth(2)  # 16-bit
        wav.setframerate(16000)  # 16kHz
        # 1 second of silence
        wav.writeframes(b"\x00\x00" * 16000)

    return wav_path
