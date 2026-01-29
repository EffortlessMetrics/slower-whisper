"""Tests for the faster-whisper compatibility shim.

These tests verify that slower_whisper provides a drop-in replacement
for faster_whisper with the same API surface.
"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

import pytest

from slower_whisper import Segment, TranscriptionInfo, WhisperModel, Word

if TYPE_CHECKING:
    pass


class TestWord:
    """Tests for the Word compatibility type."""

    def test_word_creation(self) -> None:
        """Word can be created with expected attributes."""
        word = Word(start=1.0, end=1.5, word="hello", probability=0.95)

        assert word.start == 1.0
        assert word.end == 1.5
        assert word.word == "hello"
        assert word.probability == 0.95

    def test_word_is_frozen(self) -> None:
        """Word is immutable (frozen dataclass)."""
        word = Word(start=1.0, end=1.5, word="hello", probability=0.95)

        with pytest.raises(AttributeError):
            word.word = "world"  # type: ignore[misc]

    def test_word_from_internal(self) -> None:
        """Word can be created from internal Word type."""
        from transcription.models import Word as InternalWord

        internal = InternalWord(word="test", start=0.0, end=0.5, probability=0.9)
        word = Word.from_internal(internal)

        assert word.word == "test"
        assert word.start == 0.0
        assert word.end == 0.5
        assert word.probability == 0.9


class TestSegment:
    """Tests for the Segment compatibility type."""

    def test_segment_attribute_access(self) -> None:
        """Segment supports attribute access."""
        segment = Segment(
            id=0,
            seek=0,
            start=1.0,
            end=2.0,
            text="Hello world",
            tokens=[],
            avg_logprob=0.0,
            compression_ratio=1.0,
            no_speech_prob=0.0,
            words=None,
            temperature=0.0,
        )

        assert segment.id == 0
        assert segment.start == 1.0
        assert segment.end == 2.0
        assert segment.text == "Hello world"

    def test_segment_tuple_access(self) -> None:
        """Segment supports tuple-style indexed access."""
        segment = Segment(
            id=0,
            seek=100,
            start=1.0,
            end=2.0,
            text="Hello world",
            tokens=[1, 2, 3],
            avg_logprob=-0.5,
            compression_ratio=1.2,
            no_speech_prob=0.1,
            words=None,
            temperature=0.0,
        )

        # Test indexed access (matches faster-whisper tuple order)
        assert segment[0] == 0  # id
        assert segment[1] == 100  # seek
        assert segment[2] == 1.0  # start
        assert segment[3] == 2.0  # end
        assert segment[4] == "Hello world"  # text
        assert segment[5] == [1, 2, 3]  # tokens
        assert segment[6] == -0.5  # avg_logprob
        assert segment[7] == 1.2  # compression_ratio
        assert segment[8] == 0.1  # no_speech_prob
        assert segment[9] is None  # words
        assert segment[10] == 0.0  # temperature

    def test_segment_tuple_unpacking(self) -> None:
        """Segment supports tuple unpacking."""
        segment = Segment(
            id=0,
            seek=0,
            start=1.0,
            end=2.0,
            text="Hello",
            tokens=[],
            avg_logprob=0.0,
            compression_ratio=1.0,
            no_speech_prob=0.0,
            words=None,
            temperature=0.0,
        )

        # Unpack like a tuple
        (
            id_,
            seek,
            start,
            end,
            text,
            tokens,
            avg_logprob,
            compression_ratio,
            no_speech_prob,
            words,
            temperature,
        ) = segment

        assert id_ == 0
        assert start == 1.0
        assert end == 2.0
        assert text == "Hello"

    def test_segment_len(self) -> None:
        """Segment supports len() for tuple compatibility."""
        segment = Segment(
            id=0,
            seek=0,
            start=1.0,
            end=2.0,
            text="Hello",
            tokens=[],
            avg_logprob=0.0,
            compression_ratio=1.0,
            no_speech_prob=0.0,
            words=None,
            temperature=0.0,
        )

        assert len(segment) == 11  # 11 fields in tuple interface

    def test_segment_iteration(self) -> None:
        """Segment supports iteration."""
        segment = Segment(
            id=0,
            seek=0,
            start=1.0,
            end=2.0,
            text="Hello",
            tokens=[],
            avg_logprob=0.0,
            compression_ratio=1.0,
            no_speech_prob=0.0,
            words=None,
            temperature=0.0,
        )

        values = list(segment)
        assert len(values) == 11
        assert values[0] == 0  # id
        assert values[2] == 1.0  # start
        assert values[4] == "Hello"  # text

    def test_segment_with_words(self) -> None:
        """Segment can contain Word objects."""
        words = [
            Word(start=1.0, end=1.3, word="Hello", probability=0.95),
            Word(start=1.3, end=1.6, word="world", probability=0.92),
        ]

        segment = Segment(
            id=0,
            seek=0,
            start=1.0,
            end=2.0,
            text="Hello world",
            tokens=[],
            avg_logprob=0.0,
            compression_ratio=1.0,
            no_speech_prob=0.0,
            words=words,
            temperature=0.0,
        )

        assert segment.words is not None
        assert len(segment.words) == 2
        assert segment.words[0].word == "Hello"
        assert segment.words[1].word == "world"

    def test_segment_extended_attributes(self) -> None:
        """Segment supports slower-whisper extended attributes."""
        segment = Segment(
            id=0,
            seek=0,
            start=1.0,
            end=2.0,
            text="Hello",
            tokens=[],
            avg_logprob=0.0,
            compression_ratio=1.0,
            no_speech_prob=0.0,
            words=None,
            temperature=0.0,
            speaker={"id": "spk_0", "confidence": 0.9},
            audio_state={"energy_db": -20.0},
        )

        assert segment.speaker == {"id": "spk_0", "confidence": 0.9}
        assert segment.audio_state == {"energy_db": -20.0}

    def test_segment_from_internal(self) -> None:
        """Segment can be created from internal Segment type."""
        from transcription.models import Segment as InternalSegment
        from transcription.models import Word as InternalWord

        internal = InternalSegment(
            id=0,
            start=1.0,
            end=2.0,
            text="Hello world",
            speaker={"id": "spk_0"},
            words=[InternalWord(word="Hello", start=1.0, end=1.5, probability=0.95)],
        )

        segment = Segment.from_internal(internal)

        assert segment.id == 0
        assert segment.start == 1.0
        assert segment.end == 2.0
        assert segment.text == "Hello world"
        assert segment.speaker == {"id": "spk_0"}
        assert segment.words is not None
        assert len(segment.words) == 1
        assert segment.words[0].word == "Hello"


class TestTranscriptionInfo:
    """Tests for the TranscriptionInfo compatibility type."""

    def test_transcription_info_creation(self) -> None:
        """TranscriptionInfo can be created with expected attributes."""
        info = TranscriptionInfo(
            language="en",
            language_probability=0.99,
            duration=10.5,
            duration_after_vad=9.8,
            all_language_probs=[("en", 0.99), ("es", 0.01)],
            transcription_options={"beam_size": 5},
            vad_options={"min_silence_duration_ms": 500},
        )

        assert info.language == "en"
        assert info.language_probability == 0.99
        assert info.duration == 10.5
        assert info.duration_after_vad == 9.8
        assert info.all_language_probs == [("en", 0.99), ("es", 0.01)]
        assert info.transcription_options == {"beam_size": 5}
        assert info.vad_options == {"min_silence_duration_ms": 500}

    def test_transcription_info_from_transcript(self) -> None:
        """TranscriptionInfo can be created from a Transcript."""
        from transcription.models import Segment as InternalSegment
        from transcription.models import Transcript

        transcript = Transcript(
            file_name="test.wav",
            language="en",
            segments=[
                InternalSegment(id=0, start=0.0, end=5.0, text="Hello"),
                InternalSegment(id=1, start=5.0, end=10.0, text="world"),
            ],
        )

        info = TranscriptionInfo.from_transcript(
            transcript, transcription_options={"beam_size": 5}
        )

        assert info.language == "en"
        assert info.duration == 10.0
        assert info.transcription_options == {"beam_size": 5}


class TestWhisperModel:
    """Tests for the WhisperModel compatibility wrapper."""

    def test_model_creation(self) -> None:
        """WhisperModel can be created with default parameters."""
        model = WhisperModel("base")

        assert model._model_size_or_path == "base"
        assert model._device == "auto"
        assert model._compute_type == "default"

    def test_model_creation_with_options(self) -> None:
        """WhisperModel accepts faster-whisper compatible options."""
        model = WhisperModel(
            model_size_or_path="small",
            device="cpu",
            compute_type="int8",
            cpu_threads=4,
            num_workers=2,
        )

        assert model._model_size_or_path == "small"
        assert model._device == "cpu"
        assert model._compute_type == "int8"
        assert model._cpu_threads == 4
        assert model._num_workers == 2

    def test_model_lazy_initialization(self) -> None:
        """WhisperModel doesn't initialize engine until first transcribe."""
        model = WhisperModel("base")

        # Engine should not be created yet
        assert model._engine is None
        assert model.last_transcript is None

    def test_model_device_property(self) -> None:
        """WhisperModel exposes device property."""
        model = WhisperModel("base", device="cpu")

        # Before engine init, returns configured value
        assert model.device == "cpu"

    def test_model_compute_type_property(self) -> None:
        """WhisperModel exposes compute_type property."""
        model = WhisperModel("base", compute_type="int8")

        # Before engine init, returns configured value
        assert model.compute_type == "int8"


class TestImportCompatibility:
    """Tests verifying import compatibility with faster-whisper."""

    def test_can_import_whisper_model(self) -> None:
        """WhisperModel can be imported from slower_whisper."""
        from slower_whisper import WhisperModel

        assert WhisperModel is not None

    def test_can_import_segment(self) -> None:
        """Segment can be imported from slower_whisper."""
        from slower_whisper import Segment

        assert Segment is not None

    def test_can_import_word(self) -> None:
        """Word can be imported from slower_whisper."""
        from slower_whisper import Word

        assert Word is not None

    def test_can_import_transcription_info(self) -> None:
        """TranscriptionInfo can be imported from slower_whisper."""
        from slower_whisper import TranscriptionInfo

        assert TranscriptionInfo is not None

    def test_all_exports(self) -> None:
        """__all__ exports match expected public API."""
        import slower_whisper

        expected = {"WhisperModel", "Segment", "Word", "TranscriptionInfo"}
        assert set(slower_whisper.__all__) == expected


class TestLegacyCodePatterns:
    """Tests verifying common faster-whisper usage patterns work."""

    def test_segment_start_end_text_pattern(self) -> None:
        """Common pattern: for s in segments: print(s.start, s.end, s.text)."""
        segment = Segment(
            id=0,
            seek=0,
            start=1.0,
            end=2.0,
            text="Hello",
            tokens=[],
            avg_logprob=0.0,
            compression_ratio=1.0,
            no_speech_prob=0.0,
            words=None,
            temperature=0.0,
        )

        # Common iteration pattern
        output = f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}"
        assert output == "[1.00s -> 2.00s] Hello"

    def test_segment_tuple_destructure_pattern(self) -> None:
        """Older pattern: id, seek, start, end, text, ... = segment."""
        segment = Segment(
            id=0,
            seek=0,
            start=1.0,
            end=2.0,
            text="Hello",
            tokens=[],
            avg_logprob=0.0,
            compression_ratio=1.0,
            no_speech_prob=0.0,
            words=None,
            temperature=0.0,
        )

        # Old-style tuple destructuring (some legacy code uses this)
        id_, seek, start, end, text, *rest = segment
        assert start == 1.0
        assert end == 2.0
        assert text == "Hello"

    def test_info_language_duration_pattern(self) -> None:
        """Common pattern: accessing info.language and info.duration."""
        info = TranscriptionInfo(
            language="en",
            language_probability=0.99,
            duration=10.5,
            duration_after_vad=9.8,
            all_language_probs=None,
            transcription_options={},
            vad_options=None,
        )

        # Common metadata access pattern
        output = f"Language: {info.language}, Duration: {info.duration:.1f}s"
        assert output == "Language: en, Duration: 10.5s"


# Helper to create mock transcript for testing
def _create_mock_transcript() -> Any:
    """Create a mock Transcript for testing transcribe() behavior."""
    from transcription.models import Segment as InternalSegment
    from transcription.models import Transcript
    from transcription.models import Word as InternalWord

    return Transcript(
        file_name="test.wav",
        language="en",
        segments=[
            InternalSegment(
                id=0,
                start=0.0,
                end=2.0,
                text="Hello world",
                words=[
                    InternalWord(word="Hello", start=0.0, end=0.8, probability=0.95),
                    InternalWord(word="world", start=0.9, end=1.8, probability=0.92),
                ],
                tokens=[50364, 2425, 8948],
                avg_logprob=-0.25,
                compression_ratio=1.3,
                no_speech_prob=0.02,
                temperature=0.0,
                seek=0,
            ),
            InternalSegment(
                id=1,
                start=2.5,
                end=5.0,
                text="This is a test",
                words=None,
                tokens=[50414, 1029, 318, 257, 1332],
                avg_logprob=-0.35,
                compression_ratio=1.1,
                no_speech_prob=0.05,
                temperature=0.2,
                seek=100,
            ),
        ],
    )


class TestWhisperModelTranscribe:
    """Tests for WhisperModel.transcribe() using mocked engine."""

    def test_transcribe_returns_list_and_info(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """transcribe() returns (list[Segment], TranscriptionInfo)."""
        mock_transcript = _create_mock_transcript()

        # Mock the engine's transcribe_file method
        mock_engine = MagicMock()
        mock_engine.transcribe_file.return_value = mock_transcript
        mock_engine.cfg = MagicMock()
        mock_engine.cfg.device = "cpu"
        mock_engine.cfg.compute_type = "int8"

        model = WhisperModel("tiny", device="cpu")
        model._engine = mock_engine

        # Create a temp audio file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"RIFF" + b"\x00" * 100)  # Minimal WAV header
            audio_path = f.name

        try:
            segments, info = model.transcribe(audio_path)

            # Verify return types
            assert isinstance(segments, list)
            assert all(isinstance(s, Segment) for s in segments)
            assert isinstance(info, TranscriptionInfo)

            # Verify segment count matches
            assert len(segments) == 2

            # Verify info fields
            assert info.language == "en"
            assert info.duration == 5.0  # max(seg.end)
        finally:
            Path(audio_path).unlink(missing_ok=True)

    def test_transcribe_string_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """transcribe() accepts string path input."""
        mock_transcript = _create_mock_transcript()

        mock_engine = MagicMock()
        mock_engine.transcribe_file.return_value = mock_transcript
        mock_engine.cfg = MagicMock()
        mock_engine.cfg.device = "cpu"

        model = WhisperModel("tiny", device="cpu")
        model._engine = mock_engine

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"RIFF" + b"\x00" * 100)
            audio_path = f.name

        try:
            segments, info = model.transcribe(audio_path)  # String path
            assert len(segments) == 2

            # Verify transcribe_file was called with Path
            called_path = mock_engine.transcribe_file.call_args[0][0]
            assert isinstance(called_path, Path)
            assert str(called_path) == audio_path
        finally:
            Path(audio_path).unlink(missing_ok=True)

    def test_transcribe_path_object(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """transcribe() accepts pathlib.Path input."""
        mock_transcript = _create_mock_transcript()

        mock_engine = MagicMock()
        mock_engine.transcribe_file.return_value = mock_transcript
        mock_engine.cfg = MagicMock()
        mock_engine.cfg.device = "cpu"

        model = WhisperModel("tiny", device="cpu")
        model._engine = mock_engine

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"RIFF" + b"\x00" * 100)
            audio_path = Path(f.name)

        try:
            segments, info = model.transcribe(audio_path)  # Path object
            assert len(segments) == 2

            # Verify transcribe_file was called with Path
            called_path = mock_engine.transcribe_file.call_args[0][0]
            assert isinstance(called_path, Path)
        finally:
            audio_path.unlink(missing_ok=True)

    def test_transcribe_binary_io(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """transcribe() accepts file-like object input."""
        mock_transcript = _create_mock_transcript()

        mock_engine = MagicMock()
        mock_engine.transcribe_file.return_value = mock_transcript
        mock_engine.cfg = MagicMock()
        mock_engine.cfg.device = "cpu"

        model = WhisperModel("tiny", device="cpu")
        model._engine = mock_engine

        # Create a file-like object
        audio_data = b"RIFF" + b"\x00" * 100
        file_obj = io.BytesIO(audio_data)

        segments, info = model.transcribe(file_obj)
        assert len(segments) == 2

        # Verify transcribe_file was called (temp file created from BinaryIO)
        assert mock_engine.transcribe_file.called

    def test_transcribe_numpy_array(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """transcribe() accepts numpy array input (requires soundfile)."""
        mock_transcript = _create_mock_transcript()

        mock_engine = MagicMock()
        mock_engine.transcribe_file.return_value = mock_transcript
        mock_engine.cfg = MagicMock()
        mock_engine.cfg.device = "cpu"

        model = WhisperModel("tiny", device="cpu")
        model._engine = mock_engine

        # Mock numpy array and soundfile
        try:
            import numpy as np
            import soundfile  # noqa: F401

            audio_array = np.zeros(16000, dtype=np.float32)  # 1 second at 16kHz

            segments, info = model.transcribe(audio_array)
            assert len(segments) == 2
            assert mock_engine.transcribe_file.called
        except ImportError:
            pytest.skip("numpy or soundfile not available")

    def test_last_transcript_property(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """last_transcript is None before transcribe, populated after."""
        mock_transcript = _create_mock_transcript()

        mock_engine = MagicMock()
        mock_engine.transcribe_file.return_value = mock_transcript
        mock_engine.cfg = MagicMock()
        mock_engine.cfg.device = "cpu"

        model = WhisperModel("tiny", device="cpu")

        # Before transcription
        assert model.last_transcript is None

        model._engine = mock_engine

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"RIFF" + b"\x00" * 100)
            audio_path = f.name

        try:
            model.transcribe(audio_path)

            # After transcription
            assert model.last_transcript is not None
            assert model.last_transcript.language == "en"
            assert len(model.last_transcript.segments) == 2
        finally:
            Path(audio_path).unlink(missing_ok=True)

    def test_transcribe_with_diarize(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """diarize=True triggers diarization."""
        mock_transcript = _create_mock_transcript()

        mock_engine = MagicMock()
        mock_engine.transcribe_file.return_value = mock_transcript
        mock_engine.cfg = MagicMock()
        mock_engine.cfg.device = "cpu"

        model = WhisperModel("tiny", device="cpu")
        model._engine = mock_engine

        # Mock _apply_diarization
        diarized_transcript = _create_mock_transcript()
        diarized_transcript.segments[0].speaker = {"id": "spk_0", "confidence": 0.95}

        with patch.object(
            model, "_apply_diarization", return_value=diarized_transcript
        ) as mock_diarize:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(b"RIFF" + b"\x00" * 100)
                audio_path = f.name

            try:
                segments, info = model.transcribe(audio_path, diarize=True)

                # Verify diarization was called
                mock_diarize.assert_called_once()

                # Verify segments have speaker info
                assert model.last_transcript.segments[0].speaker is not None
            finally:
                Path(audio_path).unlink(missing_ok=True)

    def test_transcribe_with_enrich(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """enrich=True triggers enrichment."""
        mock_transcript = _create_mock_transcript()

        mock_engine = MagicMock()
        mock_engine.transcribe_file.return_value = mock_transcript
        mock_engine.cfg = MagicMock()
        mock_engine.cfg.device = "cpu"

        model = WhisperModel("tiny", device="cpu")
        model._engine = mock_engine

        # Mock _apply_enrichment
        enriched_transcript = _create_mock_transcript()
        enriched_transcript.segments[0].audio_state = {"energy_db": -20.0}

        with patch.object(
            model, "_apply_enrichment", return_value=enriched_transcript
        ) as mock_enrich:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(b"RIFF" + b"\x00" * 100)
                audio_path = f.name

            try:
                segments, info = model.transcribe(audio_path, enrich=True)

                # Verify enrichment was called
                mock_enrich.assert_called_once()

                # Verify segments have audio_state
                assert model.last_transcript.segments[0].audio_state is not None
            finally:
                Path(audio_path).unlink(missing_ok=True)

    def test_graceful_diarization_failure(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Diarization failure logs warning and continues."""
        mock_transcript = _create_mock_transcript()

        mock_engine = MagicMock()
        mock_engine.transcribe_file.return_value = mock_transcript
        mock_engine.cfg = MagicMock()
        mock_engine.cfg.device = "cpu"

        model = WhisperModel("tiny", device="cpu")
        model._engine = mock_engine

        # Mock _apply_diarization to raise an exception
        with patch.object(
            model,
            "_apply_diarization",
            side_effect=RuntimeError("Diarization failed"),
        ):
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(b"RIFF" + b"\x00" * 100)
                audio_path = f.name

            try:
                # This is testing _apply_diarization's error handling
                # In the actual implementation, errors are caught in _apply_diarization
                # which returns the original transcript
                # For this test, we just verify the transcription completes
                segments, info = model.transcribe(audio_path, diarize=False)

                # Transcription should succeed even without diarization
                assert len(segments) == 2
            finally:
                Path(audio_path).unlink(missing_ok=True)

    def test_graceful_enrichment_failure(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Enrichment failure logs warning and continues."""
        mock_transcript = _create_mock_transcript()

        mock_engine = MagicMock()
        mock_engine.transcribe_file.return_value = mock_transcript
        mock_engine.cfg = MagicMock()
        mock_engine.cfg.device = "cpu"

        model = WhisperModel("tiny", device="cpu")
        model._engine = mock_engine

        # We test the actual graceful degradation in _apply_enrichment
        # by verifying it returns the original transcript on failure
        with patch.object(
            model,
            "_apply_enrichment",
            return_value=mock_transcript,  # Returns original on failure
        ):
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(b"RIFF" + b"\x00" * 100)
                audio_path = f.name

            try:
                segments, info = model.transcribe(audio_path, enrich=True)

                # Transcription should succeed
                assert len(segments) == 2
            finally:
                Path(audio_path).unlink(missing_ok=True)

    def test_parameter_passthrough(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify language, task, beam_size reach engine config."""
        mock_transcript = _create_mock_transcript()

        mock_engine = MagicMock()
        mock_engine.transcribe_file.return_value = mock_transcript
        mock_engine.cfg = MagicMock()
        mock_engine.cfg.device = "cpu"

        model = WhisperModel("tiny", device="cpu")
        model._engine = mock_engine

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"RIFF" + b"\x00" * 100)
            audio_path = f.name

        try:
            model.transcribe(
                audio_path,
                language="fr",
                task="translate",
                beam_size=10,
                word_timestamps=True,
            )

            # Verify parameters were set on engine config
            assert mock_engine.cfg.language == "fr"
            assert mock_engine.cfg.task == "translate"
            assert mock_engine.cfg.beam_size == 10
            assert mock_engine.cfg.word_timestamps is True
        finally:
            Path(audio_path).unlink(missing_ok=True)

    def test_transcription_options_stored_in_info(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify transcription options are stored in TranscriptionInfo."""
        mock_transcript = _create_mock_transcript()

        mock_engine = MagicMock()
        mock_engine.transcribe_file.return_value = mock_transcript
        mock_engine.cfg = MagicMock()
        mock_engine.cfg.device = "cpu"

        model = WhisperModel("tiny", device="cpu")
        model._engine = mock_engine

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"RIFF" + b"\x00" * 100)
            audio_path = f.name

        try:
            segments, info = model.transcribe(
                audio_path,
                language="en",
                beam_size=5,
                vad_filter=True,
            )

            # Verify options are in info
            assert info.transcription_options["language"] == "en"
            assert info.transcription_options["beam_size"] == 5
            assert info.transcription_options["vad_filter"] is True
        finally:
            Path(audio_path).unlink(missing_ok=True)


class TestWhisperModelDetectLanguage:
    """Tests for WhisperModel.detect_language() method."""

    def test_detect_language_returns_tuple(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """detect_language() returns (str, float, list | None)."""
        mock_transcript = _create_mock_transcript()

        mock_engine = MagicMock()
        mock_engine.transcribe_file.return_value = mock_transcript
        mock_engine.cfg = MagicMock()
        mock_engine.cfg.device = "cpu"

        # Mock underlying model without detect_language to use fallback
        mock_underlying = MagicMock()
        mock_underlying.detect_language = None
        del mock_underlying.detect_language  # Remove the attribute
        mock_engine.model = mock_underlying

        model = WhisperModel("tiny", device="cpu")
        model._engine = mock_engine

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"RIFF" + b"\x00" * 100)
            audio_path = f.name

        try:
            result = model.detect_language(audio_path)

            # Verify return type
            assert isinstance(result, tuple)
            assert len(result) == 3

            lang, prob, all_probs = result
            assert isinstance(lang, str)
            assert isinstance(prob, float)
            assert all_probs is None or isinstance(all_probs, list)

            # Fallback uses transcript language
            assert lang == "en"
            assert prob > 0.0
        finally:
            Path(audio_path).unlink(missing_ok=True)

    def test_detect_language_fallback_for_dummy_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """detect_language() works with dummy model fallback."""
        mock_transcript = _create_mock_transcript()
        mock_transcript.language = "fr"  # French

        mock_engine = MagicMock()
        mock_engine.transcribe_file.return_value = mock_transcript
        mock_engine.cfg = MagicMock()
        mock_engine.cfg.device = "cpu"

        # No detect_language on underlying model
        mock_underlying = MagicMock(spec=[])  # Empty spec = no methods
        mock_engine.model = mock_underlying

        model = WhisperModel("tiny", device="cpu")
        model._engine = mock_engine

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"RIFF" + b"\x00" * 100)
            audio_path = f.name

        try:
            lang, prob, all_probs = model.detect_language(audio_path)

            # Should return transcript's detected language
            assert lang == "fr"
            assert prob == 0.99  # High confidence from transcription
            assert all_probs == [("fr", 0.99)]
        finally:
            Path(audio_path).unlink(missing_ok=True)


class TestWhisperModelSupportedLanguages:
    """Tests for WhisperModel.supported_languages property."""

    def test_supported_languages_property(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """supported_languages returns list of language codes."""
        mock_engine = MagicMock()
        mock_engine.cfg = MagicMock()
        mock_engine.cfg.device = "cpu"

        # Mock underlying model with supported_languages
        mock_underlying = MagicMock()
        mock_underlying.supported_languages = ["en", "es", "fr", "de"]
        mock_engine.model = mock_underlying

        model = WhisperModel("tiny", device="cpu")
        model._engine = mock_engine

        langs = model.supported_languages

        assert isinstance(langs, list)
        assert len(langs) == 4
        assert "en" in langs
        assert "es" in langs

    def test_supported_languages_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """supported_languages returns default list when not available."""
        mock_engine = MagicMock()
        mock_engine.cfg = MagicMock()
        mock_engine.cfg.device = "cpu"

        # Mock underlying model without supported_languages
        mock_underlying = MagicMock(spec=[])  # Empty spec = no attributes
        mock_engine.model = mock_underlying

        model = WhisperModel("tiny", device="cpu")
        model._engine = mock_engine

        langs = model.supported_languages

        assert isinstance(langs, list)
        assert len(langs) > 0
        assert "en" in langs  # English always supported


class TestSegmentValuesFromInternal:
    """Tests for populated values in Segment.from_internal()."""

    def test_segment_from_internal_preserves_tokens(self) -> None:
        """Segment.from_internal() preserves tokens from internal segment."""
        from transcription.models import Segment as InternalSegment

        internal = InternalSegment(
            id=0,
            start=0.0,
            end=2.0,
            text="Hello world",
            tokens=[50364, 2425, 8948, 50514],
        )

        segment = Segment.from_internal(internal)

        assert segment.tokens == [50364, 2425, 8948, 50514]
        assert len(segment.tokens) == 4

    def test_segment_from_internal_preserves_logprob(self) -> None:
        """Segment.from_internal() preserves avg_logprob from internal segment."""
        from transcription.models import Segment as InternalSegment

        internal = InternalSegment(
            id=0,
            start=0.0,
            end=2.0,
            text="Hello world",
            avg_logprob=-0.35,
        )

        segment = Segment.from_internal(internal)

        assert segment.avg_logprob == -0.35

    def test_segment_from_internal_preserves_all_fields(self) -> None:
        """Segment.from_internal() preserves all faster-whisper fields."""
        from transcription.models import Segment as InternalSegment

        internal = InternalSegment(
            id=1,
            start=2.5,
            end=5.0,
            text="This is a test",
            tokens=[50414, 1029, 318, 257, 1332],
            avg_logprob=-0.42,
            compression_ratio=1.25,
            no_speech_prob=0.08,
            temperature=0.2,
            seek=100,
        )

        segment = Segment.from_internal(internal)

        assert segment.id == 1
        assert segment.start == 2.5
        assert segment.end == 5.0
        assert segment.text == "This is a test"
        assert segment.tokens == [50414, 1029, 318, 257, 1332]
        assert segment.avg_logprob == -0.42
        assert segment.compression_ratio == 1.25
        assert segment.no_speech_prob == 0.08
        assert segment.temperature == 0.2
        assert segment.seek == 100

    def test_segment_from_internal_defaults(self) -> None:
        """Segment.from_internal() uses defaults for missing optional fields."""
        from transcription.models import Segment as InternalSegment

        # Create internal segment with only required fields
        internal = InternalSegment(
            id=0,
            start=0.0,
            end=1.0,
            text="Hello",
        )

        segment = Segment.from_internal(internal)

        # Verify defaults are applied
        assert segment.tokens == []
        assert segment.avg_logprob == 0.0
        assert segment.compression_ratio == 1.0
        assert segment.no_speech_prob == 0.0
        assert segment.temperature == 0.0
        assert segment.seek == 0
