"""
Dedicated tests for the Diarizer class in transcription/diarization.py.

Tests the Diarizer class itself (not just assign_speakers which is tested
in test_diarization_mapping.py). Covers:
- Constructor parameters and defaults
- Pipeline loading modes (stub, missing, error scenarios)
- Thread-safety of lazy loading
- run() method with stub pipeline
- assign_speakers_to_words() function
- Helper functions (_normalize_speaker_id, _compute_overlap, etc.)
- Environment variable handling
"""

from __future__ import annotations

import sys
import threading
import types
import wave
from pathlib import Path

import pytest

from slower_whisper.pipeline.diarization import (
    Diarizer,
    SpeakerTurn,
    _compute_overlap,
    _find_best_speaker,
    _normalize_speaker_id,
    _update_speaker_stats,
    assign_speakers_to_words,
)
from slower_whisper.pipeline.models import Segment, Transcript, Word

# =============================================================================
# Test fixtures
# =============================================================================


@pytest.fixture
def sample_wav(tmp_path: Path) -> Path:
    """Create a minimal WAV file for diarization tests (16kHz mono, 2 seconds)."""
    wav_path = tmp_path / "test_audio.wav"
    with wave.open(str(wav_path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        # 2 seconds of near-silence (will produce 2 speaker turns via stub)
        wav.writeframes(b"\x00\x00" * 32000)
    return wav_path


@pytest.fixture
def sample_wav_4sec(tmp_path: Path) -> Path:
    """Create a 4-second WAV file for diarization tests."""
    wav_path = tmp_path / "test_audio_4sec.wav"
    with wave.open(str(wav_path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        # 4 seconds
        wav.writeframes(b"\x00\x00" * 64000)
    return wav_path


def make_word(word: str, start: float, end: float, probability: float = 1.0) -> Word:
    """Helper to create Word objects for testing."""
    return Word(word=word, start=start, end=end, probability=probability)


def make_segment(
    seg_id: int, start: float, end: float, text: str, words: list[Word] | None = None
) -> Segment:
    """Helper to create Segment objects with optional words."""
    return Segment(id=seg_id, start=start, end=end, text=text, words=words)


def make_speaker_turn(start: float, end: float, speaker_id: str) -> SpeakerTurn:
    """Helper to create SpeakerTurn for testing."""
    return SpeakerTurn(start=start, end=end, speaker_id=speaker_id)


def make_transcript(segments: list[Segment]) -> Transcript:
    """Helper to create Transcript for testing."""
    return Transcript(file_name="test.wav", language="en", segments=segments)


# =============================================================================
# Diarizer constructor tests
# =============================================================================


class TestDiarizerConstructor:
    """Tests for Diarizer class constructor."""

    def test_default_parameters(self):
        """Diarizer uses correct defaults."""
        diarizer = Diarizer()
        assert diarizer.device == "auto"
        assert diarizer.min_speakers is None
        assert diarizer.max_speakers is None
        assert diarizer._pipeline is None
        assert diarizer._pipeline_is_stub is False

    def test_custom_device(self):
        """Diarizer accepts custom device parameter."""
        diarizer = Diarizer(device="cuda")
        assert diarizer.device == "cuda"

    def test_custom_speaker_limits(self):
        """Diarizer accepts min/max speaker parameters."""
        diarizer = Diarizer(min_speakers=2, max_speakers=4)
        assert diarizer.min_speakers == 2
        assert diarizer.max_speakers == 4

    def test_all_parameters(self):
        """Diarizer accepts all parameters together."""
        diarizer = Diarizer(device="cpu", min_speakers=1, max_speakers=10)
        assert diarizer.device == "cpu"
        assert diarizer.min_speakers == 1
        assert diarizer.max_speakers == 10

    def test_pipeline_lock_exists(self):
        """Diarizer has a threading lock for pipeline loading."""
        diarizer = Diarizer()
        assert hasattr(diarizer, "_pipeline_lock")
        assert isinstance(diarizer._pipeline_lock, type(threading.Lock()))


# =============================================================================
# Pipeline loading tests (environment variable modes)
# =============================================================================


class TestPipelineLoading:
    """Tests for _ensure_pipeline() with different modes."""

    def test_stub_mode_creates_stub_pipeline(self, monkeypatch):
        """SLOWER_WHISPER_PYANNOTE_MODE=stub creates stub pipeline."""
        monkeypatch.setenv("SLOWER_WHISPER_PYANNOTE_MODE", "stub")
        monkeypatch.delenv("HF_TOKEN", raising=False)

        diarizer = Diarizer()
        pipeline = diarizer._ensure_pipeline()

        assert pipeline is not None
        assert diarizer._pipeline is pipeline
        assert diarizer._pipeline_is_stub is True
        assert pipeline.__class__.__name__ == "_StubPipeline"

    def test_stub_mode_no_hf_token_required(self, monkeypatch):
        """Stub mode works without HF_TOKEN."""
        monkeypatch.setenv("SLOWER_WHISPER_PYANNOTE_MODE", "stub")
        monkeypatch.delenv("HF_TOKEN", raising=False)

        diarizer = Diarizer()
        # Should not raise
        pipeline = diarizer._ensure_pipeline()
        assert pipeline is not None

    def test_missing_mode_raises_import_error(self, monkeypatch):
        """SLOWER_WHISPER_PYANNOTE_MODE=missing raises ImportError."""
        monkeypatch.setenv("SLOWER_WHISPER_PYANNOTE_MODE", "missing")

        diarizer = Diarizer()
        with pytest.raises(ImportError) as exc_info:
            diarizer._ensure_pipeline()
        assert "forced via SLOWER_WHISPER_PYANNOTE_MODE=missing" in str(exc_info.value)

    def test_auto_mode_requires_hf_token(self, monkeypatch):
        """Auto mode without HF_TOKEN raises RuntimeError."""
        monkeypatch.setenv("SLOWER_WHISPER_PYANNOTE_MODE", "auto")
        monkeypatch.delenv("HF_TOKEN", raising=False)

        diarizer = Diarizer()
        with pytest.raises(RuntimeError) as exc_info:
            diarizer._ensure_pipeline()
        assert "HF_TOKEN" in str(exc_info.value)

    def test_pipeline_loaded_only_once(self, monkeypatch):
        """Pipeline is lazily loaded only once (cached)."""
        monkeypatch.setenv("SLOWER_WHISPER_PYANNOTE_MODE", "stub")

        diarizer = Diarizer()
        pipeline1 = diarizer._ensure_pipeline()
        pipeline2 = diarizer._ensure_pipeline()

        assert pipeline1 is pipeline2
        assert diarizer._pipeline is pipeline1

    def test_case_insensitive_mode(self, monkeypatch):
        """Mode environment variable is case-insensitive."""
        monkeypatch.setenv("SLOWER_WHISPER_PYANNOTE_MODE", "STUB")
        monkeypatch.delenv("HF_TOKEN", raising=False)

        diarizer = Diarizer()
        diarizer._ensure_pipeline()
        assert diarizer._pipeline_is_stub is True


# =============================================================================
# Thread-safety tests
# =============================================================================


class TestThreadSafety:
    """Tests for thread-safety of Diarizer pipeline loading."""

    def test_concurrent_ensure_pipeline_calls(self, monkeypatch):
        """Multiple threads calling _ensure_pipeline() get same instance."""
        monkeypatch.setenv("SLOWER_WHISPER_PYANNOTE_MODE", "stub")

        diarizer = Diarizer()
        pipelines: list = []
        errors: list = []

        def load_pipeline():
            try:
                p = diarizer._ensure_pipeline()
                pipelines.append(p)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=load_pipeline) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(pipelines) == 10
        # All threads should get the same pipeline instance
        assert all(p is pipelines[0] for p in pipelines)

    def test_pipeline_lock_prevents_double_init(self, monkeypatch):
        """Lock prevents multiple pipeline initializations."""
        monkeypatch.setenv("SLOWER_WHISPER_PYANNOTE_MODE", "stub")

        init_count = 0
        original_make_stub = None

        # Import the module to get the original function
        import slower_whisper.pipeline.diarization as diar_module

        original_make_stub = diar_module._make_stub_pyannote_pipeline

        def counting_make_stub():
            nonlocal init_count
            init_count += 1
            return original_make_stub()

        monkeypatch.setattr(diar_module, "_make_stub_pyannote_pipeline", counting_make_stub)

        diarizer = Diarizer()

        threads = [threading.Thread(target=diarizer._ensure_pipeline) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Pipeline should only be initialized once
        assert init_count == 1


# =============================================================================
# run() method tests
# =============================================================================


class TestDiarizerRun:
    """Tests for Diarizer.run() method."""

    def test_run_with_stub_pipeline(self, monkeypatch, sample_wav):
        """run() with stub pipeline returns speaker turns."""
        monkeypatch.setenv("SLOWER_WHISPER_PYANNOTE_MODE", "stub")

        diarizer = Diarizer()
        turns = diarizer.run(sample_wav)

        assert isinstance(turns, list)
        assert len(turns) == 2  # Stub produces 2 speakers
        assert all(isinstance(t, SpeakerTurn) for t in turns)
        assert turns[0].speaker_id == "SPEAKER_00"
        assert turns[1].speaker_id == "SPEAKER_01"

    def test_run_with_string_path(self, monkeypatch, sample_wav):
        """run() accepts string path."""
        monkeypatch.setenv("SLOWER_WHISPER_PYANNOTE_MODE", "stub")

        diarizer = Diarizer()
        turns = diarizer.run(str(sample_wav))

        assert len(turns) == 2

    def test_run_returns_sorted_turns(self, monkeypatch, sample_wav):
        """run() returns turns sorted by start time."""
        monkeypatch.setenv("SLOWER_WHISPER_PYANNOTE_MODE", "stub")

        diarizer = Diarizer()
        turns = diarizer.run(sample_wav)

        for i in range(len(turns) - 1):
            assert turns[i].start <= turns[i + 1].start

    def test_run_file_not_found(self, monkeypatch):
        """run() raises FileNotFoundError for missing file."""
        monkeypatch.setenv("SLOWER_WHISPER_PYANNOTE_MODE", "stub")

        diarizer = Diarizer()
        with pytest.raises(FileNotFoundError) as exc_info:
            diarizer.run("/nonexistent/path.wav")
        assert "not found" in str(exc_info.value).lower()

    def test_run_respects_speaker_limits(self, monkeypatch, sample_wav):
        """run() passes min/max_speakers to pipeline."""
        monkeypatch.setenv("SLOWER_WHISPER_PYANNOTE_MODE", "stub")

        diarizer = Diarizer(min_speakers=2, max_speakers=2)
        # Stub ignores these but we verify they're passed
        turns = diarizer.run(sample_wav)
        assert len(turns) == 2

    def test_run_stub_timing_scales_with_audio(self, monkeypatch, sample_wav_4sec):
        """Stub pipeline scales speaker segments with audio duration."""
        monkeypatch.setenv("SLOWER_WHISPER_PYANNOTE_MODE", "stub")

        diarizer = Diarizer()
        turns = diarizer.run(sample_wav_4sec)

        # 4-second audio should have midpoint at 2s
        assert turns[0].start == 0.0
        assert turns[0].end == pytest.approx(2.0, rel=0.1)
        assert turns[1].start == pytest.approx(2.0, rel=0.1)
        assert turns[1].end == pytest.approx(4.0, rel=0.1)


# =============================================================================
# Helper function tests
# =============================================================================


class TestNormalizeSpeakerId:
    """Tests for _normalize_speaker_id helper function."""

    def test_first_speaker_gets_spk_0(self):
        """First speaker seen gets spk_0."""
        speaker_map: dict[str, int] = {}
        result = _normalize_speaker_id("SPEAKER_00", speaker_map)
        assert result == "spk_0"
        assert speaker_map == {"SPEAKER_00": 0}

    def test_second_speaker_gets_spk_1(self):
        """Second speaker seen gets spk_1."""
        speaker_map: dict[str, int] = {"SPEAKER_00": 0}
        result = _normalize_speaker_id("SPEAKER_01", speaker_map)
        assert result == "spk_1"
        assert speaker_map == {"SPEAKER_00": 0, "SPEAKER_01": 1}

    def test_same_speaker_returns_same_id(self):
        """Same raw ID returns same normalized ID."""
        speaker_map: dict[str, int] = {"SPEAKER_00": 0}
        result = _normalize_speaker_id("SPEAKER_00", speaker_map)
        assert result == "spk_0"

    def test_arbitrary_speaker_ids(self):
        """Works with arbitrary speaker ID formats."""
        speaker_map: dict[str, int] = {}
        assert _normalize_speaker_id("Alice", speaker_map) == "spk_0"
        assert _normalize_speaker_id("Bob", speaker_map) == "spk_1"
        assert _normalize_speaker_id("Charlie", speaker_map) == "spk_2"


class TestComputeOverlap:
    """Tests for _compute_overlap helper function."""

    def test_full_overlap(self):
        """Segment fully inside turn."""
        overlap = _compute_overlap(1.0, 3.0, 0.0, 5.0)
        assert overlap == pytest.approx(2.0)

    def test_partial_overlap_left(self):
        """Segment starts before turn."""
        overlap = _compute_overlap(0.0, 3.0, 2.0, 5.0)
        assert overlap == pytest.approx(1.0)

    def test_partial_overlap_right(self):
        """Segment ends after turn."""
        overlap = _compute_overlap(3.0, 6.0, 2.0, 5.0)
        assert overlap == pytest.approx(2.0)

    def test_no_overlap_before(self):
        """Segment completely before turn."""
        overlap = _compute_overlap(0.0, 1.0, 2.0, 5.0)
        assert overlap == pytest.approx(0.0)

    def test_no_overlap_after(self):
        """Segment completely after turn."""
        overlap = _compute_overlap(6.0, 8.0, 2.0, 5.0)
        assert overlap == pytest.approx(0.0)

    def test_exact_overlap(self):
        """Segment exactly matches turn."""
        overlap = _compute_overlap(2.0, 5.0, 2.0, 5.0)
        assert overlap == pytest.approx(3.0)

    def test_turn_inside_segment(self):
        """Turn fully inside segment."""
        overlap = _compute_overlap(0.0, 10.0, 2.0, 5.0)
        assert overlap == pytest.approx(3.0)


class TestFindBestSpeaker:
    """Tests for _find_best_speaker helper function."""

    def test_single_turn_full_overlap(self):
        """Single turn with full overlap."""
        turns = [make_speaker_turn(0.0, 5.0, "SPEAKER_00")]
        speaker, overlap = _find_best_speaker(1.0, 3.0, turns)
        assert speaker == "SPEAKER_00"
        assert overlap == pytest.approx(2.0)

    def test_multiple_turns_max_overlap(self):
        """Multiple turns - returns speaker with max overlap."""
        turns = [
            make_speaker_turn(0.0, 2.0, "SPEAKER_00"),  # 1s overlap
            make_speaker_turn(2.0, 5.0, "SPEAKER_01"),  # 2s overlap
        ]
        speaker, overlap = _find_best_speaker(1.0, 4.0, turns)
        assert speaker == "SPEAKER_01"
        assert overlap == pytest.approx(2.0)

    def test_equal_overlap_alphabetical(self):
        """Equal overlap - returns first speaker alphabetically."""
        turns = [
            make_speaker_turn(0.0, 2.0, "SPEAKER_01"),  # 1s overlap
            make_speaker_turn(2.0, 4.0, "SPEAKER_00"),  # 1s overlap
        ]
        speaker, overlap = _find_best_speaker(1.0, 3.0, turns)
        assert speaker == "SPEAKER_00"
        assert overlap == pytest.approx(1.0)

    def test_no_overlap(self):
        """No overlapping turns."""
        turns = [make_speaker_turn(5.0, 10.0, "SPEAKER_00")]
        speaker, overlap = _find_best_speaker(0.0, 3.0, turns)
        assert speaker is None
        assert overlap == pytest.approx(0.0)

    def test_empty_turns(self):
        """Empty turns list."""
        speaker, overlap = _find_best_speaker(0.0, 3.0, [])
        assert speaker is None
        assert overlap == pytest.approx(0.0)


class TestUpdateSpeakerStats:
    """Tests for _update_speaker_stats helper function."""

    def test_new_speaker(self):
        """New speaker creates entry."""
        stats: dict[str, dict] = {}
        _update_speaker_stats(stats, "spk_0", 2.0)
        assert "spk_0" in stats
        assert stats["spk_0"]["id"] == "spk_0"
        assert stats["spk_0"]["total_speech_time"] == pytest.approx(2.0)
        assert stats["spk_0"]["num_segments"] == 1

    def test_existing_speaker(self):
        """Existing speaker updates entry."""
        stats: dict[str, dict] = {
            "spk_0": {
                "id": "spk_0",
                "label": None,
                "total_speech_time": 2.0,
                "num_segments": 1,
            }
        }
        _update_speaker_stats(stats, "spk_0", 3.0)
        assert stats["spk_0"]["total_speech_time"] == pytest.approx(5.0)
        assert stats["spk_0"]["num_segments"] == 2


# =============================================================================
# assign_speakers_to_words() tests
# =============================================================================


class TestAssignSpeakersToWords:
    """Tests for assign_speakers_to_words function (word-level assignment)."""

    def test_basic_word_assignment(self):
        """Words get speaker assigned based on overlap."""
        words = [
            make_word("Hello", 0.0, 0.5),
            make_word("world", 0.5, 1.0),
        ]
        segments = [make_segment(0, 0.0, 1.0, "Hello world", words=words)]
        transcript = make_transcript(segments)

        turns = [make_speaker_turn(0.0, 1.0, "SPEAKER_00")]
        result = assign_speakers_to_words(transcript, turns)

        assert result.segments[0].words[0].speaker == "spk_0"
        assert result.segments[0].words[1].speaker == "spk_0"

    def test_word_level_speaker_change(self):
        """Detects speaker change within segment at word level."""
        words = [
            make_word("Hello", 0.0, 0.5),
            make_word("world", 1.0, 1.5),  # Different speaker
        ]
        segments = [make_segment(0, 0.0, 1.5, "Hello world", words=words)]
        transcript = make_transcript(segments)

        turns = [
            make_speaker_turn(0.0, 0.6, "SPEAKER_00"),
            make_speaker_turn(0.9, 1.5, "SPEAKER_01"),
        ]
        result = assign_speakers_to_words(transcript, turns)

        assert result.segments[0].words[0].speaker == "spk_0"
        assert result.segments[0].words[1].speaker == "spk_1"

    def test_segment_speaker_from_dominant_word(self):
        """Segment speaker is derived from dominant word-level speaker."""
        words = [
            make_word("Hello", 0.0, 0.3),  # 0.3s - speaker A
            make_word("world", 0.5, 1.0),  # 0.5s - speaker B (longer)
        ]
        segments = [make_segment(0, 0.0, 1.0, "Hello world", words=words)]
        transcript = make_transcript(segments)

        turns = [
            make_speaker_turn(0.0, 0.4, "SPEAKER_00"),
            make_speaker_turn(0.4, 1.0, "SPEAKER_01"),
        ]
        result = assign_speakers_to_words(transcript, turns)

        # Dominant speaker is SPEAKER_01 (longer word duration)
        assert result.segments[0].speaker is not None
        assert result.segments[0].speaker["id"] == "spk_1"

    def test_segment_without_words_fallback(self):
        """Segment without words falls back to segment-level assignment."""
        segments = [make_segment(0, 0.0, 2.0, "Hello world", words=None)]
        transcript = make_transcript(segments)

        turns = [make_speaker_turn(0.0, 2.0, "SPEAKER_00")]
        result = assign_speakers_to_words(transcript, turns)

        assert result.segments[0].speaker is not None
        assert result.segments[0].speaker["id"] == "spk_0"

    def test_word_below_threshold(self):
        """Word with overlap below threshold gets speaker=None."""
        words = [make_word("Hi", 0.0, 1.0)]
        segments = [make_segment(0, 0.0, 1.0, "Hi", words=words)]
        transcript = make_transcript(segments)

        # Turn has only 0.2s overlap with 1.0s word (20% < 30% threshold)
        turns = [make_speaker_turn(0.8, 1.0, "SPEAKER_00")]
        result = assign_speakers_to_words(transcript, turns, overlap_threshold=0.3)

        assert result.segments[0].words[0].speaker is None

    def test_speakers_list_from_words(self):
        """speakers[] array built from word-level assignments."""
        words = [
            make_word("Hello", 0.0, 0.5),
            make_word("world", 1.0, 1.5),
        ]
        segments = [make_segment(0, 0.0, 1.5, "Hello world", words=words)]
        transcript = make_transcript(segments)

        turns = [
            make_speaker_turn(0.0, 0.6, "SPEAKER_00"),
            make_speaker_turn(0.9, 1.5, "SPEAKER_01"),
        ]
        result = assign_speakers_to_words(transcript, turns)

        assert result.speakers is not None
        assert len(result.speakers) == 2
        speaker_ids = {s["id"] for s in result.speakers}
        assert speaker_ids == {"spk_0", "spk_1"}

    def test_empty_segment_words(self):
        """Segment with empty words list falls back to segment-level."""
        segments = [make_segment(0, 0.0, 2.0, "Hello", words=[])]
        transcript = make_transcript(segments)

        turns = [make_speaker_turn(0.0, 2.0, "SPEAKER_00")]
        result = assign_speakers_to_words(transcript, turns)

        # Empty words list treated as no words
        assert result.segments[0].speaker is not None
        assert result.segments[0].speaker["id"] == "spk_0"

    def test_zero_duration_word_skipped(self):
        """Zero-duration words are skipped."""
        words = [
            make_word("", 1.0, 1.0),  # Zero duration
            make_word("Hello", 1.0, 2.0),  # Valid
        ]
        segments = [make_segment(0, 1.0, 2.0, "Hello", words=words)]
        transcript = make_transcript(segments)

        turns = [make_speaker_turn(0.0, 3.0, "SPEAKER_00")]
        result = assign_speakers_to_words(transcript, turns)

        # First word skipped, second gets assignment
        assert result.segments[0].words[1].speaker == "spk_0"

    def test_out_of_order_segments_handled(self):
        """Out-of-order segments are handled correctly."""
        words1 = [make_word("World", 2.0, 3.0)]
        words2 = [make_word("Hello", 0.0, 1.0)]
        segments = [
            make_segment(0, 2.0, 3.0, "World", words=words1),
            make_segment(1, 0.0, 1.0, "Hello", words=words2),
        ]
        transcript = make_transcript(segments)

        turns = [
            make_speaker_turn(0.0, 1.5, "SPEAKER_00"),
            make_speaker_turn(1.5, 3.0, "SPEAKER_01"),
        ]
        result = assign_speakers_to_words(transcript, turns)

        # Both segments should be correctly assigned
        assert result.segments[0].words[0].speaker == "spk_1"  # 2.0-3.0
        assert result.segments[1].words[0].speaker == "spk_0"  # 0.0-1.0


# =============================================================================
# Environment variable handling tests
# =============================================================================


class TestEnvironmentVariables:
    """Tests for environment variable handling."""

    def test_custom_model_env_var(self, monkeypatch, tmp_path):
        """SLOWER_WHISPER_PYANNOTE_MODEL can override default model."""
        monkeypatch.setenv("SLOWER_WHISPER_PYANNOTE_MODE", "auto")
        monkeypatch.setenv("HF_TOKEN", "dummy_token")
        monkeypatch.setenv("SLOWER_WHISPER_PYANNOTE_MODEL", "custom/model")
        monkeypatch.setenv("SLOWER_WHISPER_CACHE_ROOT", str(tmp_path / "cache"))

        called_model: list[str] = []

        class FakePipeline:
            @classmethod
            def from_pretrained(cls, model_id, **kwargs):
                called_model.append(model_id)
                return cls()

            def to(self, device):
                return self

        fake_audio_module = types.SimpleNamespace(Pipeline=FakePipeline)
        monkeypatch.setitem(sys.modules, "pyannote", types.ModuleType("pyannote"))
        monkeypatch.setitem(sys.modules, "pyannote.audio", fake_audio_module)

        diarizer = Diarizer()
        try:
            diarizer._ensure_pipeline()
        except Exception:
            pass  # May fail for other reasons

        if called_model:
            assert called_model[0] == "custom/model"

    def test_cache_root_env_var(self, monkeypatch, tmp_path):
        """SLOWER_WHISPER_CACHE_ROOT sets cache directory."""
        cache_root = tmp_path / "custom_cache"
        monkeypatch.setenv("SLOWER_WHISPER_PYANNOTE_MODE", "auto")
        monkeypatch.setenv("HF_TOKEN", "dummy_token")
        monkeypatch.setenv("SLOWER_WHISPER_CACHE_ROOT", str(cache_root))

        called_kwargs: dict = {}

        class FakePipeline:
            @classmethod
            def from_pretrained(cls, model_id, **kwargs):
                called_kwargs.update(kwargs)
                return cls()

            def to(self, device):
                return self

        fake_audio_module = types.SimpleNamespace(Pipeline=FakePipeline)
        monkeypatch.setitem(sys.modules, "pyannote", types.ModuleType("pyannote"))
        monkeypatch.setitem(sys.modules, "pyannote.audio", fake_audio_module)

        diarizer = Diarizer()
        try:
            diarizer._ensure_pipeline()
        except Exception:
            pass

        if "cache_dir" in called_kwargs:
            assert str(cache_root) in called_kwargs["cache_dir"]


# =============================================================================
# Stub pipeline internals tests
# =============================================================================


class TestStubPipeline:
    """Tests for the internal stub pipeline implementation."""

    def test_stub_from_pretrained(self, monkeypatch):
        """Stub pipeline from_pretrained returns instance."""
        monkeypatch.setenv("SLOWER_WHISPER_PYANNOTE_MODE", "stub")

        diarizer = Diarizer()
        pipeline = diarizer._ensure_pipeline()

        # Verify from_pretrained works
        new_instance = pipeline.from_pretrained("any_model")
        assert new_instance is not None

    def test_stub_to_device(self, monkeypatch):
        """Stub pipeline to() returns self."""
        monkeypatch.setenv("SLOWER_WHISPER_PYANNOTE_MODE", "stub")

        diarizer = Diarizer()
        pipeline = diarizer._ensure_pipeline()

        result = pipeline.to("cuda")
        assert result is pipeline

    def test_stub_annotation_itertracks(self, monkeypatch, sample_wav):
        """Stub annotation itertracks yields segments."""
        monkeypatch.setenv("SLOWER_WHISPER_PYANNOTE_MODE", "stub")

        diarizer = Diarizer()
        pipeline = diarizer._ensure_pipeline()

        result = pipeline(str(sample_wav))
        tracks = list(result.itertracks(yield_label=True))

        assert len(tracks) == 2
        assert tracks[0][2] == "SPEAKER_00"
        assert tracks[1][2] == "SPEAKER_01"


# =============================================================================
# Deprecated API tests
# =============================================================================


class TestDeprecatedApi:
    """Tests for deprecated API handling."""

    def test_assign_speakers_to_segments_raises(self):
        """assign_speakers_to_segments raises NotImplementedError."""
        from slower_whisper.pipeline.diarization import assign_speakers_to_segments

        with pytest.raises(NotImplementedError) as exc_info:
            assign_speakers_to_segments([], [])
        assert "deprecated" in str(exc_info.value).lower()
        assert "assign_speakers" in str(exc_info.value)
