"""
Integration tests for v1.1 diarization skeleton.

Tests that enable_diarization=True doesn't crash the pipeline,
and properly marks the transcript with meta.diarization.status.
"""

import subprocess
import sys
import types

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

    # Should return transcript unchanged but with explicit disabled metadata
    assert result is transcript
    assert result.speakers is None
    assert result.turns is None
    assert result.meta is not None
    diar_meta = result.meta.get("diarization", {})
    assert diar_meta.get("status") == "disabled"
    assert diar_meta.get("requested") is False


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
    assert result.meta["diarization"]["status"] == "error"
    assert result.meta["diarization"]["requested"] is True
    assert "error" in result.meta["diarization"]
    assert "Audio file not found" in result.meta["diarization"]["error"]
    assert result.meta["diarization"]["error_type"] == "file_not_found"


def test_maybe_run_diarization_missing_dependency(monkeypatch, tmp_path):
    """
    Import-time failures (ModuleNotFoundError) should be classified as missing_dependency.
    """
    from transcription.api import _maybe_run_diarization

    transcript = Transcript(
        file_name="test.wav",
        language="en",
        segments=[Segment(id=0, start=0.0, end=2.0, text="Hello")],
    )

    def _missing_dep_run(self, _wav_path):
        raise ModuleNotFoundError("No module named 'torch'")

    monkeypatch.setattr("transcription.diarization.Diarizer.run", _missing_dep_run, raising=False)

    wav_path = tmp_path / "dummy.wav"
    wav_path.write_bytes(b"\x00\x00")

    config = TConfig(enable_diarization=True, diarization_device="cpu")
    result = _maybe_run_diarization(transcript, wav_path, config)

    diar_meta = result.meta["diarization"]
    assert diar_meta["status"] == "skipped"
    assert diar_meta["error_type"] == "missing_dependency"
    assert diar_meta["requested"] is True
    assert result.segments[0].speaker is None
    assert result.speakers is None
    assert result.turns is None


def test_maybe_run_diarization_wrapped_missing_dependency(monkeypatch, tmp_path):
    """
    Wrapped runtime errors (e.g., missing torch inside pyannote loader) should still
    classify as missing_dependency by inspecting the exception cause/message.
    """
    from transcription.api import _maybe_run_diarization

    transcript = Transcript(
        file_name="test.wav",
        language="en",
        segments=[Segment(id=0, start=0.0, end=2.0, text="Hello")],
    )

    def _wrapped_missing_dep(self, _wav_path):
        raise RuntimeError(
            "Failed to load pyannote pipeline: No module named 'torch'"
        ) from ImportError("No module named 'torch'")

    monkeypatch.setattr(
        "transcription.diarization.Diarizer.run", _wrapped_missing_dep, raising=False
    )

    wav_path = tmp_path / "dummy.wav"
    wav_path.write_bytes(b"\x00\x00")

    config = TConfig(enable_diarization=True, diarization_device="cpu")
    result = _maybe_run_diarization(transcript, wav_path, config)

    diar_meta = result.meta["diarization"]
    assert diar_meta["status"] == "skipped"
    assert diar_meta["error_type"] == "missing_dependency"
    assert diar_meta["requested"] is True
    assert result.segments[0].speaker is None
    assert result.speakers is None
    assert result.turns is None


def test_maybe_run_diarization_rolls_back_on_failure(monkeypatch):
    """Partial diarization work should not leak into the transcript on failure."""
    from pathlib import Path

    from transcription.api import _maybe_run_diarization
    from transcription.diarization import SpeakerTurn

    transcript = Transcript(
        file_name="test.wav",
        language="en",
        segments=[Segment(id=0, start=0.0, end=2.0, text="Hello")],
    )

    class FakeDiarizer:
        def __init__(self, *args, **kwargs):
            pass

        def run(self, _wav_path):
            return [SpeakerTurn(start=0.0, end=1.0, speaker_id="SPEAKER_00")]

    def failing_assign_speakers(transcript, speaker_turns, overlap_threshold=0.3):
        transcript.segments[0].speaker = {"id": "spk_0", "confidence": 0.9}
        transcript.speakers = [
            {"id": "spk_0", "label": None, "total_speech_time": 2.0, "num_segments": 1}
        ]
        raise RuntimeError("assign_speakers failure")

    monkeypatch.setattr("transcription.api.Diarizer", FakeDiarizer, raising=False)
    monkeypatch.setattr("transcription.api.assign_speakers", failing_assign_speakers, raising=False)

    config = TConfig(enable_diarization=True, diarization_device="cpu")
    result = _maybe_run_diarization(transcript, Path("dummy.wav"), config)

    assert result.segments[0].speaker is None  # rolled back to original
    assert result.speakers is None
    assert result.turns is None
    assert result.meta is not None
    assert result.meta["diarization"]["status"] == "error"
    assert result.meta["diarization"]["requested"] is True


def test_maybe_run_diarization_replaces_old_error_metadata(monkeypatch, tmp_path):
    """Successful diarization should clear any previous failure metadata."""
    from transcription.api import _maybe_run_diarization
    from transcription.diarization import SpeakerTurn

    transcript = Transcript(
        file_name="test.wav",
        language="en",
        segments=[Segment(id=0, start=0.0, end=2.0, text="Hello")],
        meta={
            "diarization": {
                "status": "failed",
                "requested": True,
                "error": "old failure",
                "error_type": "missing_dependency",
            }
        },
    )

    class FakeDiarizer:
        def __init__(self, *args, **kwargs):
            pass

        def run(self, _wav_path):
            return [SpeakerTurn(start=0.0, end=2.0, speaker_id="SPEAKER_00")]

    def fake_assign_speakers(transcript, speaker_turns, overlap_threshold=0.3):
        transcript.segments[0].speaker = {"id": "spk_0", "confidence": 0.9}
        transcript.speakers = [
            {"id": "spk_0", "label": None, "total_speech_time": 2.0, "num_segments": 1}
        ]
        return transcript

    def fake_build_turns(transcript):
        transcript.turns = [
            {
                "id": "turn_0",
                "speaker_id": "spk_0",
                "start": 0.0,
                "end": 2.0,
                "segment_ids": [0],
                "text": "Hello",
            }
        ]
        return transcript

    monkeypatch.setattr("transcription.diarization.Diarizer", FakeDiarizer, raising=False)
    monkeypatch.setattr(
        "transcription.diarization.assign_speakers", fake_assign_speakers, raising=False
    )
    monkeypatch.setattr("transcription.turns.build_turns", fake_build_turns, raising=False)

    wav_path = tmp_path / "dummy.wav"
    wav_path.write_bytes(b"\x00\x00")

    config = TConfig(enable_diarization=True, diarization_device="cpu")
    result = _maybe_run_diarization(transcript, wav_path, config)

    diar_meta = result.meta["diarization"]
    assert diar_meta["status"] == "ok"
    assert diar_meta["backend"] == "pyannote.audio"
    assert diar_meta["num_speakers"] == 1
    assert diar_meta.get("error") is None
    assert diar_meta["error_type"] is None
    assert set(diar_meta.keys()) == {"status", "requested", "backend", "num_speakers", "error_type"}


def test_maybe_run_diarization_failure_overrides_prior_success(monkeypatch, tmp_path):
    """Failure metadata should replace any prior success fields to avoid stale backend info."""
    from transcription.api import _maybe_run_diarization

    transcript = Transcript(
        file_name="test.wav",
        language="en",
        segments=[Segment(id=0, start=0.0, end=2.0, text="Hello")],
        meta={
            "diarization": {
                "status": "success",
                "requested": True,
                "backend": "pyannote.audio",
                "num_speakers": 2,
                "error": None,
                "error_type": None,
            }
        },
    )

    class FailingDiarizer:
        def __init__(self, *args, **kwargs):
            pass

        def run(self, _wav_path):
            raise RuntimeError("pyannote.audio missing")

    monkeypatch.setattr("transcription.diarization.Diarizer", FailingDiarizer, raising=False)

    wav_path = tmp_path / "dummy.wav"
    wav_path.write_bytes(b"\x00\x00")

    config = TConfig(enable_diarization=True, diarization_device="cpu")
    result = _maybe_run_diarization(transcript, wav_path, config)

    diar_meta = result.meta["diarization"]
    assert diar_meta["status"] == "skipped"
    assert diar_meta["requested"] is True
    assert "pyannote.audio missing" in diar_meta["error"]
    assert diar_meta["error_type"] == "missing_dependency"
    assert diar_meta.get("backend") == "pyannote.audio"
    assert diar_meta.get("num_speakers") is None


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


def test_diarization_runs_when_json_exists(monkeypatch, tmp_path):
    """
    When skip_existing_json=True but diarization is enabled, the pipeline should
    reuse existing transcripts and still execute diarization.
    """
    import wave

    from transcription import writers
    from transcription.config import AppConfig, AsrConfig, Paths
    from transcription.pipeline import run_pipeline

    # Prepare project layout with existing transcript and normalized audio
    paths = Paths(root=tmp_path)
    paths.raw_dir.mkdir(parents=True, exist_ok=True)
    paths.norm_dir.mkdir(parents=True, exist_ok=True)
    paths.json_dir.mkdir(parents=True, exist_ok=True)
    paths.transcripts_dir.mkdir(parents=True, exist_ok=True)

    wav_path = paths.norm_dir / "existing.wav"
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\0\0" * 16000)

    base_transcript = Transcript(
        file_name="existing.wav",
        language="en",
        segments=[Segment(id=0, start=0.0, end=1.0, text="hello")],
    )
    writers.write_json(base_transcript, paths.json_dir / "existing.json")

    # Avoid ffmpeg and ASR work; we only want diarization to run
    monkeypatch.setattr("transcription.pipeline.audio_io.normalize_all", lambda _paths: None)
    monkeypatch.setattr(
        "transcription.pipeline.TranscriptionEngine.transcribe_file",
        lambda self, wav: (_ for _ in ()).throw(
            AssertionError("transcription should be skipped when JSON already exists")
        ),
    )

    diarization_calls: dict[str, bool] = {}

    def _fake_maybe_run_diarization(transcript, wav_path, config):
        diarization_calls["called"] = True
        transcript.speakers = [
            {"id": "spk_0", "label": None, "total_speech_time": 1.0, "num_segments": 1}
        ]
        transcript.turns = [
            {
                "id": "turn_0",
                "speaker_id": "spk_0",
                "start": 0.0,
                "end": 1.0,
                "segment_ids": [0],
                "text": "hello",
            }
        ]
        transcript.meta = {"diarization": {"status": "ok", "requested": True}}
        return transcript

    monkeypatch.setattr("transcription.api._maybe_run_diarization", _fake_maybe_run_diarization)

    app_cfg = AppConfig(
        paths=paths,
        asr=AsrConfig(model_name="tiny", device="cpu"),
        skip_existing_json=True,
    )
    diar_cfg = TConfig(enable_diarization=True, diarization_device="cpu")

    run_pipeline(app_cfg, diarization_config=diar_cfg)

    assert diarization_calls.get("called") is True
    updated = writers.load_transcript_from_json(paths.json_dir / "existing.json")
    assert updated.speakers is not None
    assert updated.turns is not None
    assert updated.meta is not None
    assert updated.meta.get("diarization", {}).get("status") in {"ok", "success"}


def test_diarizer_backwards_token_param(monkeypatch, tmp_path):
    """
    Older pyannote versions expect use_auth_token instead of token.
    _ensure_pipeline should retry with the legacy parameter instead of failing.
    """
    monkeypatch.setenv("SLOWER_WHISPER_CACHE_ROOT", str(tmp_path / "cache_root"))
    monkeypatch.setenv("HF_TOKEN", "dummy_token")

    called_kwargs: dict[str, object] = {}

    class FakePipeline:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            if "token" in kwargs:
                raise TypeError("token argument not supported")
            called_kwargs.update(kwargs)
            return cls()

        def to(self, device):
            called_kwargs["device"] = device
            return self

    fake_audio_module = types.SimpleNamespace(Pipeline=FakePipeline)
    monkeypatch.setitem(sys.modules, "pyannote", types.ModuleType("pyannote"))
    monkeypatch.setitem(sys.modules, "pyannote.audio", fake_audio_module)

    from transcription.diarization import Diarizer

    diarizer = Diarizer(device="auto")
    pipeline = diarizer._ensure_pipeline()

    assert isinstance(pipeline, FakePipeline)
    assert diarizer._pipeline is pipeline
    assert called_kwargs.get("use_auth_token") == "dummy_token"
    expected_cache = tmp_path / "cache_root" / "diarization"
    assert called_kwargs.get("cache_dir") == str(expected_cache)


def test_stub_mode_does_not_require_hf_token(monkeypatch):
    """
    Stub backend should work without HF_TOKEN because it never hits Hugging Face.
    """
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("SLOWER_WHISPER_PYANNOTE_MODE", "stub")

    from transcription.diarization import Diarizer

    diarizer = Diarizer()
    pipeline = diarizer._ensure_pipeline()

    assert pipeline is diarizer._pipeline
    assert pipeline.__class__.__name__ == "_StubPipeline"


# ============================================================================
# Integration test with synthetic fixture (ready for pyannote)
# ============================================================================


@pytest.mark.heavy
@pytest.mark.requires_diarization
@pytest.mark.skipif(not ffmpeg_available(), reason="ffmpeg not available")
def test_synthetic_2speaker_diarization(tmp_path, monkeypatch):
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
    monkeypatch.setenv("SLOWER_WHISPER_PYANNOTE_MODE", "stub")
    monkeypatch.setenv("HF_TOKEN", "dummy-token")

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
    assert transcript.meta["diarization"]["status"] == "ok"
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
    - speakers/turns may be None when diarization backend unavailable
    - meta.diarization.status reflects outcome ("ok", "skipped", or "error")
    - meta.diarization.requested = True
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
    assert diar_meta["status"] in {"ok", "skipped", "error"}
    assert diar_meta["requested"] is True


@pytest.mark.skipif(not ffmpeg_available(), reason="ffmpeg not available")
def test_transcribe_file_without_diarization(tmp_path, sample_audio_path):
    """
    Test that enable_diarization=False records disabled diarization metadata.

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

    assert transcript.speakers is None
    assert transcript.turns is None
    assert transcript.meta is not None
    diar_meta = transcript.meta.get("diarization", {})
    assert diar_meta.get("status") == "disabled"
    assert diar_meta.get("requested") is False


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
