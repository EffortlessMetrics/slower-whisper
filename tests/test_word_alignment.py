"""
Unit tests for word-level alignment feature (v1.8+).

Tests the Word dataclass, serialization/deserialization, and integration
with Segment and Transcript models.
"""

import json

from slower_whisper.pipeline.models import WORD_ALIGNMENT_VERSION, Segment, Transcript, Word


class TestWordDataclass:
    """Tests for the Word dataclass."""

    def test_word_creation_basic(self):
        """Word can be created with required fields."""
        word = Word(word="hello", start=0.0, end=0.5, probability=0.95)

        assert word.word == "hello"
        assert word.start == 0.0
        assert word.end == 0.5
        assert word.probability == 0.95
        assert word.speaker is None  # Default

    def test_word_creation_with_speaker(self):
        """Word can include speaker assignment."""
        word = Word(word="hello", start=0.0, end=0.5, probability=0.95, speaker="spk_0")

        assert word.speaker == "spk_0"

    def test_word_to_dict(self):
        """Word.to_dict() produces correct JSON-serializable dict."""
        word = Word(word="hello", start=1.0, end=1.5, probability=0.85)

        result = word.to_dict()

        assert result == {
            "word": "hello",
            "start": 1.0,
            "end": 1.5,
            "probability": 0.85,
        }

    def test_word_to_dict_with_speaker(self):
        """Word.to_dict() includes speaker when present."""
        word = Word(word="hello", start=1.0, end=1.5, probability=0.85, speaker="spk_1")

        result = word.to_dict()

        assert result == {
            "word": "hello",
            "start": 1.0,
            "end": 1.5,
            "probability": 0.85,
            "speaker": "spk_1",
        }

    def test_word_from_dict_basic(self):
        """Word.from_dict() reconstructs Word from dict."""
        data = {"word": "world", "start": 2.0, "end": 2.5, "probability": 0.9}

        word = Word.from_dict(data)

        assert word.word == "world"
        assert word.start == 2.0
        assert word.end == 2.5
        assert word.probability == 0.9
        assert word.speaker is None

    def test_word_from_dict_with_speaker(self):
        """Word.from_dict() handles speaker field."""
        data = {"word": "world", "start": 2.0, "end": 2.5, "probability": 0.9, "speaker": "spk_2"}

        word = Word.from_dict(data)

        assert word.speaker == "spk_2"

    def test_word_from_dict_missing_fields_use_defaults(self):
        """Word.from_dict() handles missing optional fields."""
        data = {"word": "test"}

        word = Word.from_dict(data)

        assert word.word == "test"
        assert word.start == 0.0  # Default
        assert word.end == 0.0  # Default
        assert word.probability == 1.0  # Default
        assert word.speaker is None

    def test_word_roundtrip_serialization(self):
        """Word can be serialized and deserialized correctly."""
        original = Word(word="example", start=5.5, end=6.0, probability=0.78, speaker="spk_0")

        json_str = json.dumps(original.to_dict())
        restored = Word.from_dict(json.loads(json_str))

        assert restored.word == original.word
        assert restored.start == original.start
        assert restored.end == original.end
        assert restored.probability == original.probability
        assert restored.speaker == original.speaker


class TestSegmentWithWords:
    """Tests for Segment with word-level timestamps."""

    def test_segment_without_words(self):
        """Segment works without words (backward compatibility)."""
        segment = Segment(id=0, start=0.0, end=2.0, text="Hello world")

        assert segment.words is None

    def test_segment_with_words(self):
        """Segment can include word-level timestamps."""
        words = [
            Word(word="Hello", start=0.0, end=0.5, probability=0.9),
            Word(word="world", start=0.6, end=1.0, probability=0.85),
        ]
        segment = Segment(id=0, start=0.0, end=1.0, text="Hello world", words=words)

        assert segment.words is not None
        assert len(segment.words) == 2
        assert segment.words[0].word == "Hello"
        assert segment.words[1].word == "world"

    def test_segment_words_with_speaker(self):
        """Segment words can have speaker assignments."""
        words = [
            Word(word="Hello", start=0.0, end=0.5, probability=0.9, speaker="spk_0"),
            Word(word="world", start=0.6, end=1.0, probability=0.85, speaker="spk_0"),
        ]
        segment = Segment(id=0, start=0.0, end=1.0, text="Hello world", words=words)

        assert segment.words[0].speaker == "spk_0"
        assert segment.words[1].speaker == "spk_0"


class TestWritersSerialization:
    """Tests for JSON serialization of words in writers.py."""

    def test_segment_with_words_serializes(self, tmp_path):
        """Segment with words serializes to JSON correctly."""
        from slower_whisper.pipeline.writers import write_json

        words = [
            Word(word="Hello", start=0.0, end=0.5, probability=0.9),
            Word(word="world", start=0.6, end=1.0, probability=0.85),
        ]
        segment = Segment(id=0, start=0.0, end=1.0, text="Hello world", words=words)
        transcript = Transcript(file_name="test.wav", language="en", segments=[segment])

        json_path = tmp_path / "test.json"
        write_json(transcript, json_path)

        # Verify JSON structure
        with open(json_path) as f:
            data = json.load(f)

        seg_data = data["segments"][0]
        assert "words" in seg_data
        assert len(seg_data["words"]) == 2
        assert seg_data["words"][0]["word"] == "Hello"
        assert seg_data["words"][0]["probability"] == 0.9

    def test_segment_without_words_no_words_key(self, tmp_path):
        """Segment without words omits words key from JSON."""
        from slower_whisper.pipeline.writers import write_json

        segment = Segment(id=0, start=0.0, end=1.0, text="Hello world")
        transcript = Transcript(file_name="test.wav", language="en", segments=[segment])

        json_path = tmp_path / "test.json"
        write_json(transcript, json_path)

        with open(json_path) as f:
            data = json.load(f)

        seg_data = data["segments"][0]
        assert "words" not in seg_data  # Key should be omitted, not null

    def test_words_roundtrip(self, tmp_path):
        """Words survive JSON save/load roundtrip."""
        from slower_whisper.pipeline.writers import load_transcript_from_json, write_json

        words = [
            Word(word="Test", start=0.0, end=0.3, probability=0.95, speaker="spk_1"),
            Word(word="words", start=0.4, end=0.8, probability=0.88),
        ]
        segment = Segment(id=0, start=0.0, end=0.8, text="Test words", words=words)
        transcript = Transcript(file_name="test.wav", language="en", segments=[segment])

        json_path = tmp_path / "test.json"
        write_json(transcript, json_path)
        loaded = load_transcript_from_json(json_path)

        assert loaded.segments[0].words is not None
        assert len(loaded.segments[0].words) == 2
        assert loaded.segments[0].words[0].word == "Test"
        assert loaded.segments[0].words[0].speaker == "spk_1"
        assert loaded.segments[0].words[1].word == "words"
        assert loaded.segments[0].words[1].speaker is None

    def test_load_old_json_without_words(self, tmp_path):
        """Loading old JSON without words field works (backward compat)."""
        from slower_whisper.pipeline.writers import load_transcript_from_json

        # Simulate old JSON format without words
        old_json = {
            "schema_version": 2,
            "file": "old.wav",
            "language": "en",
            "meta": {},
            "segments": [{"id": 0, "start": 0.0, "end": 1.0, "text": "Old format"}],
        }

        json_path = tmp_path / "old.json"
        json_path.write_text(json.dumps(old_json))

        loaded = load_transcript_from_json(json_path)

        assert loaded.segments[0].words is None


class TestWordAlignmentVersion:
    """Tests for word alignment versioning."""

    def test_word_alignment_version_exists(self):
        """WORD_ALIGNMENT_VERSION constant is defined."""
        assert WORD_ALIGNMENT_VERSION == "1.0.0"


class TestConfigWordTimestamps:
    """Tests for word_timestamps config option."""

    def test_transcription_config_word_timestamps_default(self):
        """word_timestamps defaults to False."""
        from slower_whisper.pipeline.config import TranscriptionConfig

        config = TranscriptionConfig(model="base")

        assert config.word_timestamps is False

    def test_transcription_config_word_timestamps_enabled(self):
        """word_timestamps can be enabled."""
        from slower_whisper.pipeline.config import TranscriptionConfig

        config = TranscriptionConfig(model="base", word_timestamps=True)

        assert config.word_timestamps is True

    def test_asr_config_word_timestamps_default(self):
        """AsrConfig word_timestamps defaults to False."""
        from slower_whisper.pipeline.config import AsrConfig

        config = AsrConfig()

        assert config.word_timestamps is False

    def test_asr_config_word_timestamps_enabled(self):
        """AsrConfig word_timestamps can be enabled."""
        from slower_whisper.pipeline.config import AsrConfig

        config = AsrConfig(word_timestamps=True)

        assert config.word_timestamps is True


class TestWordLevelSpeakerAlignment:
    """Tests for word-level speaker alignment (v1.8+)."""

    def test_assign_speakers_to_words_basic(self):
        """Words get speaker labels based on diarization overlap."""
        from slower_whisper.pipeline.diarization import SpeakerTurn, assign_speakers_to_words

        words = [
            Word(word="Hello", start=0.0, end=0.5, probability=0.9),
            Word(word="world", start=0.6, end=1.0, probability=0.85),
        ]
        segment = Segment(id=0, start=0.0, end=1.0, text="Hello world", words=words)
        transcript = Transcript(file_name="test.wav", language="en", segments=[segment])

        turns = [SpeakerTurn(start=0.0, end=2.0, speaker_id="SPEAKER_00")]

        result = assign_speakers_to_words(transcript, turns, overlap_threshold=0.3)

        # Both words should be assigned to spk_0
        assert result.segments[0].words[0].speaker == "spk_0"
        assert result.segments[0].words[1].speaker == "spk_0"
        assert result.segments[0].speaker["id"] == "spk_0"

    def test_assign_speakers_to_words_split_segment(self):
        """Words in same segment can have different speakers."""
        from slower_whisper.pipeline.diarization import SpeakerTurn, assign_speakers_to_words

        words = [
            Word(word="Hello", start=0.0, end=0.5, probability=0.9),
            Word(word="there", start=0.6, end=1.0, probability=0.85),
            Word(word="how", start=1.5, end=1.8, probability=0.9),
            Word(word="are", start=1.9, end=2.1, probability=0.85),
            Word(word="you", start=2.2, end=2.5, probability=0.9),
        ]
        segment = Segment(id=0, start=0.0, end=2.5, text="Hello there how are you", words=words)
        transcript = Transcript(file_name="test.wav", language="en", segments=[segment])

        # Speaker A speaks first two words, Speaker B speaks last three
        turns = [
            SpeakerTurn(start=0.0, end=1.2, speaker_id="SPEAKER_A"),
            SpeakerTurn(start=1.4, end=3.0, speaker_id="SPEAKER_B"),
        ]

        result = assign_speakers_to_words(transcript, turns, overlap_threshold=0.3)

        # First two words → spk_0 (SPEAKER_A)
        assert result.segments[0].words[0].speaker == "spk_0"
        assert result.segments[0].words[1].speaker == "spk_0"
        # Last three words → spk_1 (SPEAKER_B)
        assert result.segments[0].words[2].speaker == "spk_1"
        assert result.segments[0].words[3].speaker == "spk_1"
        assert result.segments[0].words[4].speaker == "spk_1"

    def test_assign_speakers_to_words_no_words_fallback(self):
        """Segments without words fall back to segment-level alignment."""
        from slower_whisper.pipeline.diarization import SpeakerTurn, assign_speakers_to_words

        segment = Segment(id=0, start=0.0, end=1.0, text="Hello world")  # No words
        transcript = Transcript(file_name="test.wav", language="en", segments=[segment])

        turns = [SpeakerTurn(start=0.0, end=2.0, speaker_id="SPEAKER_00")]

        result = assign_speakers_to_words(transcript, turns, overlap_threshold=0.3)

        # Segment should still get speaker assignment
        assert result.segments[0].speaker is not None
        assert result.segments[0].speaker["id"] == "spk_0"

    def test_assign_speakers_to_words_below_threshold(self):
        """Words with low overlap get None speaker."""
        from slower_whisper.pipeline.diarization import SpeakerTurn, assign_speakers_to_words

        words = [
            Word(word="Hello", start=0.0, end=0.5, probability=0.9),
            Word(word="world", start=5.0, end=5.5, probability=0.85),  # Far from speaker turn
        ]
        segment = Segment(id=0, start=0.0, end=5.5, text="Hello world", words=words)
        transcript = Transcript(file_name="test.wav", language="en", segments=[segment])

        turns = [SpeakerTurn(start=0.0, end=1.0, speaker_id="SPEAKER_00")]

        result = assign_speakers_to_words(transcript, turns, overlap_threshold=0.3)

        # First word overlaps speaker turn
        assert result.segments[0].words[0].speaker == "spk_0"
        # Second word doesn't overlap any speaker turn
        assert result.segments[0].words[1].speaker is None

    def test_assign_speakers_to_words_dominant_speaker(self):
        """Segment speaker derived from dominant word-level speaker."""
        from slower_whisper.pipeline.diarization import SpeakerTurn, assign_speakers_to_words

        words = [
            Word(word="One", start=0.0, end=0.3, probability=0.9),
            Word(word="Two", start=1.0, end=1.3, probability=0.9),
            Word(word="Three", start=1.5, end=1.8, probability=0.9),
            Word(word="Four", start=2.0, end=2.3, probability=0.9),
        ]
        segment = Segment(id=0, start=0.0, end=2.5, text="One Two Three Four", words=words)
        transcript = Transcript(file_name="test.wav", language="en", segments=[segment])

        # Speaker A says "One", Speaker B says "Two Three Four"
        turns = [
            SpeakerTurn(start=0.0, end=0.5, speaker_id="SPEAKER_A"),
            SpeakerTurn(start=0.8, end=3.0, speaker_id="SPEAKER_B"),
        ]

        result = assign_speakers_to_words(transcript, turns, overlap_threshold=0.3)

        # Segment should be attributed to dominant speaker (B, who said more words)
        assert result.segments[0].speaker["id"] == "spk_1"  # SPEAKER_B

    def test_assign_speakers_to_words_speakers_list(self):
        """speakers[] list built from unique speakers."""
        from slower_whisper.pipeline.diarization import SpeakerTurn, assign_speakers_to_words

        words = [
            Word(word="Hello", start=0.0, end=0.5, probability=0.9),
            Word(word="Hi", start=1.0, end=1.5, probability=0.85),
        ]
        segment = Segment(id=0, start=0.0, end=1.5, text="Hello Hi", words=words)
        transcript = Transcript(file_name="test.wav", language="en", segments=[segment])

        turns = [
            SpeakerTurn(start=0.0, end=0.6, speaker_id="SPEAKER_A"),
            SpeakerTurn(start=0.9, end=2.0, speaker_id="SPEAKER_B"),
        ]

        result = assign_speakers_to_words(transcript, turns, overlap_threshold=0.3)

        assert result.speakers is not None
        assert len(result.speakers) == 2
        speaker_ids = {s["id"] for s in result.speakers}
        assert "spk_0" in speaker_ids
        assert "spk_1" in speaker_ids
