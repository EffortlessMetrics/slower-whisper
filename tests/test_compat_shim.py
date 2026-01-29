"""Tests for the faster-whisper compatibility shim.

These tests verify that slower_whisper provides a drop-in replacement
for faster_whisper with the same API surface.
"""

import pytest

from slower_whisper import Segment, TranscriptionInfo, WhisperModel, Word


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
