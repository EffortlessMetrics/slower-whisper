"""Compatibility types for faster-whisper drop-in replacement.

This module provides types that match faster-whisper's public interface,
allowing slower-whisper to be used as a drop-in replacement:

    # Before
    from faster_whisper import WhisperModel

    # After
    from slower_whisper import WhisperModel

The key types are:
- Segment: Matches faster_whisper.transcribe.Segment (supports both attribute and tuple access)
- Word: Matches faster_whisper.transcribe.Word
- TranscriptionInfo: Matches faster_whisper.transcribe.TranscriptionInfo
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from transcription.models import Segment as InternalSegment
    from transcription.models import Transcript
    from transcription.models import Word as InternalWord

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Word:
    """Compatibility Word matching faster-whisper's Word type.

    Attributes:
        start: Start time in seconds.
        end: End time in seconds.
        word: The transcribed word text.
        probability: Confidence score (0.0-1.0).
    """

    start: float
    end: float
    word: str
    probability: float

    @classmethod
    def from_internal(cls, word: InternalWord) -> Word:
        """Convert from slower-whisper's internal Word type."""
        return cls(
            start=word.start,
            end=word.end,
            word=word.word,
            probability=word.probability,
        )


class Segment:
    """Compatibility Segment matching faster-whisper's Segment type.

    This class supports both attribute access (segment.text) and tuple-style
    unpacking for backwards compatibility with code that treats segments as tuples.

    The tuple order matches faster-whisper:
        (id, seek, start, end, text, tokens, avg_logprob, compression_ratio,
         no_speech_prob, words, temperature)

    Attributes:
        id: Segment index (0-based).
        seek: Seek position (always 0 for compatibility).
        start: Start time in seconds.
        end: End time in seconds.
        text: Transcribed text.
        tokens: Token IDs (empty list for compatibility).
        avg_logprob: Average log probability (0.0 for compatibility).
        compression_ratio: Compression ratio (1.0 for compatibility).
        no_speech_prob: No-speech probability (0.0 for compatibility).
        words: Optional list of Word objects with word-level timestamps.
        temperature: Temperature used (0.0 for compatibility).
    """

    __slots__ = (
        "id",
        "seek",
        "start",
        "end",
        "text",
        "tokens",
        "avg_logprob",
        "compression_ratio",
        "no_speech_prob",
        "words",
        "temperature",
        # Extended attributes for slower-whisper enrichment
        "_speaker",
        "_audio_state",
    )

    def __init__(
        self,
        id: int,
        seek: int,
        start: float,
        end: float,
        text: str,
        tokens: list[int],
        avg_logprob: float,
        compression_ratio: float,
        no_speech_prob: float,
        words: list[Word] | None,
        temperature: float,
        # Extended (not part of tuple interface)
        speaker: dict[str, Any] | None = None,
        audio_state: dict[str, Any] | None = None,
    ) -> None:
        self.id = id
        self.seek = seek
        self.start = start
        self.end = end
        self.text = text
        self.tokens = tokens
        self.avg_logprob = avg_logprob
        self.compression_ratio = compression_ratio
        self.no_speech_prob = no_speech_prob
        self.words = words
        self.temperature = temperature
        self._speaker = speaker
        self._audio_state = audio_state

    # Extended properties for slower-whisper enrichment (not part of faster-whisper API)
    @property
    def speaker(self) -> dict[str, Any] | None:
        """Speaker information from diarization (slower-whisper extension)."""
        return self._speaker

    @property
    def audio_state(self) -> dict[str, Any] | None:
        """Audio enrichment state (slower-whisper extension)."""
        return self._audio_state

    # Tuple compatibility: support indexed access and unpacking
    _TUPLE_FIELDS = (
        "id",
        "seek",
        "start",
        "end",
        "text",
        "tokens",
        "avg_logprob",
        "compression_ratio",
        "no_speech_prob",
        "words",
        "temperature",
    )

    def __getitem__(self, index: int) -> Any:
        """Support tuple-style indexed access: segment[2] -> start time."""
        return getattr(self, self._TUPLE_FIELDS[index])

    def __iter__(self) -> Iterator[Any]:
        """Support tuple unpacking: id, seek, start, end, ... = segment."""
        for field in self._TUPLE_FIELDS:
            yield getattr(self, field)

    def __len__(self) -> int:
        """Support len(segment) for tuple compatibility."""
        return len(self._TUPLE_FIELDS)

    def __repr__(self) -> str:
        return (
            f"Segment(id={self.id}, seek={self.seek}, start={self.start:.3f}, "
            f"end={self.end:.3f}, text={self.text!r}, ...)"
        )

    @classmethod
    def from_internal(cls, seg: InternalSegment) -> Segment:
        """Convert from slower-whisper's internal Segment type."""
        words: list[Word] | None = None
        if seg.words:
            words = [Word.from_internal(w) for w in seg.words]

        # Extract faster-whisper compatibility fields from internal segment
        # These are now tracked in the internal Segment model
        # Track which fields used defaults for debugging
        defaults_used: list[str] = []

        tokens = getattr(seg, "tokens", None)
        if tokens is None:
            tokens = []
            defaults_used.append("tokens")

        avg_logprob = getattr(seg, "avg_logprob", None)
        if avg_logprob is None:
            avg_logprob = 0.0
            defaults_used.append("avg_logprob")

        compression_ratio = getattr(seg, "compression_ratio", None)
        if compression_ratio is None:
            compression_ratio = 1.0
            defaults_used.append("compression_ratio")

        no_speech_prob = getattr(seg, "no_speech_prob", None)
        if no_speech_prob is None:
            no_speech_prob = 0.0
            defaults_used.append("no_speech_prob")

        temperature = getattr(seg, "temperature", None)
        if temperature is None:
            temperature = 0.0
            defaults_used.append("temperature")

        seek = getattr(seg, "seek", None)
        if seek is None:
            seek = 0
            defaults_used.append("seek")

        if defaults_used:
            logger.debug(
                "Segment %d: using default values for fields: %s",
                seg.id,
                ", ".join(defaults_used),
            )

        return cls(
            id=seg.id,
            seek=seek,
            start=seg.start,
            end=seg.end,
            text=seg.text,
            tokens=list(tokens) if tokens else [],
            avg_logprob=float(avg_logprob),
            compression_ratio=float(compression_ratio),
            no_speech_prob=float(no_speech_prob),
            words=words,
            temperature=float(temperature),
            speaker=seg.speaker,
            audio_state=seg.audio_state,
        )


@dataclass(frozen=True, slots=True)
class TranscriptionInfo:
    """Compatibility TranscriptionInfo matching faster-whisper's type.

    Attributes:
        language: Detected or specified language code.
        language_probability: Confidence in language detection (1.0 default).
        duration: Total audio duration in seconds.
        duration_after_vad: Duration after VAD filtering (same as duration if no VAD).
        all_language_probs: List of (language, probability) tuples.
        transcription_options: Transcription options used.
        vad_options: VAD options used (None if no VAD).
    """

    language: str
    language_probability: float
    duration: float
    duration_after_vad: float
    all_language_probs: list[tuple[str, float]] | None
    transcription_options: dict[str, Any]
    vad_options: dict[str, Any] | None

    @classmethod
    def from_transcript(
        cls,
        transcript: Transcript,
        language_probability: float = 1.0,
        transcription_options: dict[str, Any] | None = None,
        vad_options: dict[str, Any] | None = None,
    ) -> TranscriptionInfo:
        """Create TranscriptionInfo from a slower-whisper Transcript."""
        duration = transcript.duration
        # Use VAD duration from transcript if available, otherwise fall back to total duration
        duration_after_vad = getattr(transcript, "duration_after_vad", None)
        if duration_after_vad is None:
            duration_after_vad = duration
        return cls(
            language=transcript.language,
            language_probability=language_probability,
            duration=duration,
            duration_after_vad=duration_after_vad,
            all_language_probs=[(transcript.language, language_probability)],
            transcription_options=transcription_options or {},
            vad_options=vad_options,
        )
