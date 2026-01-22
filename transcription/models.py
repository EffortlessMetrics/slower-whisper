"""Core data models for transcripts, segments, and enriched features.

This module defines the dataclass models that form the backbone of slower-whisper:

- Transcript: Complete transcript with metadata, segments, turns, and annotations
- Segment: Single transcribed segment with optional audio_state
- DiarizationMeta: Speaker diarization metadata
- BatchFileResult: Result of processing a single file in batch mode
- BatchProcessingResult: Summary of batch transcription operation
- EnrichmentFileResult: Result of enriching a single transcript file
- EnrichmentBatchResult: Summary of batch enrichment operation
- SCHEMA_VERSION: Current JSON schema version (v2)
- AUDIO_STATE_VERSION: Audio enrichment schema version

All models support serialization to/from JSON via to_dict() methods and
are validated against the JSON schema in schemas/transcript-v2.schema.json.
"""

from dataclasses import dataclass, field
from typing import Any, Literal

SCHEMA_VERSION: int = 2
AUDIO_STATE_VERSION: str = "1.0.0"
WORD_ALIGNMENT_VERSION: str = "1.0.0"


@dataclass
class Word:
    """
    A single word with timing and confidence information.

    This dataclass represents word-level alignment data extracted from
    faster-whisper's word_timestamps feature, enabling precise timing
    for each word within a segment.

    Attributes:
        word: The transcribed word text (may include leading/trailing punctuation).
        start: Start time in seconds.
        end: End time in seconds.
        probability: Confidence score from the ASR model (0.0-1.0).
        speaker: Optional speaker ID assigned via word-level diarization alignment.
                 None when diarization is disabled or speaker assignment is ambiguous.
    """

    word: str
    start: float
    end: float
    probability: float = 1.0
    speaker: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize word to a JSON-serializable dict.

        Used by writers.py for JSON serialization of word-level timestamps.
        """
        result: dict[str, Any] = {
            "word": self.word,
            "start": self.start,
            "end": self.end,
            "probability": self.probability,
        }
        if self.speaker is not None:
            result["speaker"] = self.speaker
        return result

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Word":
        """Deserialize word from a dict."""
        return cls(
            word=str(d.get("word", "")),
            start=float(d.get("start", 0.0)),
            end=float(d.get("end", 0.0)),
            probability=float(d.get("probability", 1.0)),
            speaker=d.get("speaker"),
        )


@dataclass
class DiarizationMeta:
    """Structured status block for diarization runs (meta.diarization)."""

    requested: bool
    status: Literal["disabled", "skipped", "ok", "error"]
    backend: str | None = None
    num_speakers: int | None = None
    error_type: str | None = None
    message: str | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        # Keep message/error in sync for backward compatibility.
        if self.message is None and self.error is not None:
            self.message = self.error
        if self.error is None and self.message is not None:
            self.error = self.message

    def to_dict(self) -> dict[str, Any]:
        """Serialize diarization metadata to a JSON-serializable dict.

        Includes optional message/error fields only when set.
        Used by writers.py for JSON output and by _to_dict() helper.
        """
        data = {
            "requested": self.requested,
            "status": self.status,
            "backend": self.backend,
            "num_speakers": self.num_speakers,
            "error_type": self.error_type,
        }
        if self.message is not None:
            data["message"] = self.message
        if self.error is not None:
            data["error"] = self.error
        return data


@dataclass
class TurnMeta:
    question_count: int = 0
    interruption_started_here: bool = False
    avg_pause_ms: float | None = None
    disfluency_ratio: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize turn metadata to a JSON-serializable dict.

        Used by turns_enrich.py when attaching enriched metadata to turns.
        """
        return {
            "question_count": self.question_count,
            "interruption_started_here": self.interruption_started_here,
            "avg_pause_ms": self.avg_pause_ms,
            "disfluency_ratio": self.disfluency_ratio,
        }


@dataclass
class Turn:
    """
    Canonical turn representation (v1.1+), kept dict-compatible for v1.0/1.1 users.
    """

    id: str
    speaker_id: str
    segment_ids: list[int]
    start: float
    end: float
    text: str
    metadata: dict[str, Any] | None = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize turn to a JSON-serializable dict.

        Converts segment_ids to a fresh list and ensures metadata is non-null.
        Used by writers.py via _to_dict() for JSON serialization.
        """
        return {
            "id": self.id,
            "speaker_id": self.speaker_id,
            "segment_ids": list(self.segment_ids),
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "metadata": self.metadata or {},
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Turn":
        return cls(
            id=d.get("id", ""),
            speaker_id=d.get("speaker_id") or d.get("speaker") or "spk_0",
            segment_ids=list(d.get("segment_ids", [])),
            start=float(d.get("start", 0.0)),
            end=float(d.get("end", 0.0)),
            text=str(d.get("text", "")),
            metadata=d.get("metadata") or d.get("meta") or {},
        )


@dataclass
class Chunk:
    """
    Turn-aware chunk for downstream retrieval (v1.3 scaffold).

    Chunks group one or more turns/segments into a RAG-ready slice with
    stable IDs and lightweight metadata for loaders.

    Turn-aware metadata (v1.3.1+):
        crosses_turn_boundary: True if chunk contains segments from multiple turns.
        turn_boundary_count: Number of speaker turn transitions within this chunk.
        has_rapid_turn_taking: True if chunk contains rapid turn-taking patterns
            (speaker switches with minimal gaps).
        has_overlapping_speech: True if chunk contains overlapping speech segments.
    """

    id: str
    start: float
    end: float
    segment_ids: list[int] = field(default_factory=list)
    turn_ids: list[str] = field(default_factory=list)
    speaker_ids: list[str] = field(default_factory=list)
    token_count_estimate: int = 0
    text: str = ""
    # Turn-aware metadata (v1.3.1+)
    crosses_turn_boundary: bool = False
    turn_boundary_count: int = 0
    has_rapid_turn_taking: bool = False
    has_overlapping_speech: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize chunk to a JSON-serializable dict.

        Converts all ID lists to fresh Python lists for JSON safety.
        Used by writers.py via _to_dict() for JSON serialization.
        """
        return {
            "id": self.id,
            "start": self.start,
            "end": self.end,
            "segment_ids": list(self.segment_ids),
            "turn_ids": list(self.turn_ids),
            "speaker_ids": list(self.speaker_ids),
            "token_count_estimate": self.token_count_estimate,
            "text": self.text,
            "crosses_turn_boundary": self.crosses_turn_boundary,
            "turn_boundary_count": self.turn_boundary_count,
            "has_rapid_turn_taking": self.has_rapid_turn_taking,
            "has_overlapping_speech": self.has_overlapping_speech,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Chunk":
        return cls(
            id=str(d.get("id", "")),
            start=float(d.get("start", 0.0)),
            end=float(d.get("end", 0.0)),
            segment_ids=list(d.get("segment_ids", [])),
            turn_ids=list(d.get("turn_ids", [])),
            speaker_ids=list(d.get("speaker_ids", [])),
            token_count_estimate=int(d.get("token_count_estimate", 0)),
            text=str(d.get("text", "")),
            crosses_turn_boundary=bool(d.get("crosses_turn_boundary", False)),
            turn_boundary_count=int(d.get("turn_boundary_count", 0)),
            has_rapid_turn_taking=bool(d.get("has_rapid_turn_taking", False)),
            has_overlapping_speech=bool(d.get("has_overlapping_speech", False)),
        )


@dataclass
class ProsodySummary:
    pitch_median_hz: float | None = None
    energy_median_db: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert ProsodySummary to a JSON-serializable dict."""
        return {
            "pitch_median_hz": self.pitch_median_hz,
            "energy_median_db": self.energy_median_db,
        }


@dataclass
class SentimentSummary:
    positive: float = 0.0
    neutral: float = 0.0
    negative: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert SentimentSummary to a JSON-serializable dict."""
        return {
            "positive": self.positive,
            "neutral": self.neutral,
            "negative": self.negative,
        }


@dataclass
class SpeakerStats:
    speaker_id: str
    total_talk_time: float
    num_turns: int
    avg_turn_duration: float
    interruptions_initiated: int
    interruptions_received: int
    question_turns: int
    prosody_summary: ProsodySummary
    sentiment_summary: SentimentSummary

    def to_dict(self) -> dict[str, Any]:
        """Serialize speaker stats to a JSON-serializable dict.

        Expands nested prosody_summary and sentiment_summary dataclasses
        into plain dicts. Used by writers.py via _to_dict() for JSON output.
        """
        return {
            "speaker_id": self.speaker_id,
            "total_talk_time": self.total_talk_time,
            "num_turns": self.num_turns,
            "avg_turn_duration": self.avg_turn_duration,
            "interruptions_initiated": self.interruptions_initiated,
            "interruptions_received": self.interruptions_received,
            "question_turns": self.question_turns,
            "prosody_summary": {
                "pitch_median_hz": self.prosody_summary.pitch_median_hz,
                "energy_median_db": self.prosody_summary.energy_median_db,
            },
            "sentiment_summary": {
                "positive": self.sentiment_summary.positive,
                "neutral": self.sentiment_summary.neutral,
                "negative": self.sentiment_summary.negative,
            },
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SpeakerStats":
        prosody = d.get("prosody_summary") or {}
        sentiment = d.get("sentiment_summary") or {}
        return cls(
            speaker_id=d.get("speaker_id", ""),
            total_talk_time=float(d.get("total_talk_time", 0.0)),
            num_turns=int(d.get("num_turns", 0)),
            avg_turn_duration=float(d.get("avg_turn_duration", 0.0)),
            interruptions_initiated=int(d.get("interruptions_initiated", 0)),
            interruptions_received=int(d.get("interruptions_received", 0)),
            question_turns=int(d.get("question_turns", 0)),
            prosody_summary=ProsodySummary(
                pitch_median_hz=prosody.get("pitch_median_hz"),
                energy_median_db=prosody.get("energy_median_db"),
            ),
            sentiment_summary=SentimentSummary(
                positive=float(sentiment.get("positive", 0.0)),
                neutral=float(sentiment.get("neutral", 0.0)),
                negative=float(sentiment.get("negative", 0.0)),
            ),
        )


@dataclass
class BatchFileResult:
    """Result of processing a single file in batch transcription mode.

    Represents the outcome of transcribing a single audio file, capturing
    success/failure status, the resulting transcript (if successful), and
    any error information.

    Attributes:
        file_path: Absolute path to the input audio file.
        status: Processing outcome - "success" or "error".
        transcript: The resulting Transcript object if successful, None otherwise.
        error_type: Short error category (e.g., "FileNotFoundError", "ASRError") if failed.
        error_message: Human-readable error description if failed.
    """

    file_path: str
    status: Literal["success", "error"]
    transcript: "Transcript | None" = None
    error_type: str | None = None
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize batch file result to a JSON-serializable dict.

        Returns:
            Dict with file_path, status, error fields. Transcript is excluded
            (too large for batch summaries).
        """
        return {
            "file_path": self.file_path,
            "status": self.status,
            "error_type": self.error_type,
            "error_message": self.error_message,
        }


@dataclass
class BatchProcessingResult:
    """Summary of batch transcription operation across multiple files.

    Aggregates results from processing multiple audio files in a batch,
    providing both individual file results and summary statistics.

    Attributes:
        total_files: Total number of files attempted.
        successful: Number of files successfully transcribed.
        failed: Number of files that failed transcription.
        results: List of individual BatchFileResult objects for each file.
    """

    total_files: int
    successful: int
    failed: int
    results: list[BatchFileResult] = field(default_factory=list)

    def get_failures(self) -> list[BatchFileResult]:
        """Return only the failed file results.

        Returns:
            List of BatchFileResult objects where status == "error".
        """
        return [r for r in self.results if r.status == "error"]

    def get_transcripts(self) -> list["Transcript"]:
        """Return all successfully generated transcripts.

        Returns:
            List of Transcript objects from successful results.
        """
        return [r.transcript for r in self.results if r.transcript is not None]

    def to_dict(self) -> dict[str, Any]:
        """Serialize batch processing summary to a JSON-serializable dict.

        Returns:
            Dict with counts and list of individual file result summaries.
        """
        return {
            "total_files": self.total_files,
            "successful": self.successful,
            "failed": self.failed,
            "results": [r.to_dict() for r in self.results],
        }


@dataclass
class EnrichmentFileResult:
    """Result of enriching a single transcript file with audio features.

    Represents the outcome of audio enrichment for a single transcript,
    capturing success/failure status and any error information.

    Attributes:
        transcript_path: Absolute path to the input transcript JSON file.
        status: Enrichment outcome - "success", "partial", or "error".
        enriched_transcript: The enriched Transcript object if successful, None otherwise.
        error_type: Short error category (e.g., "AudioNotFoundError") if failed.
        error_message: Human-readable error description if failed.
        warnings: List of non-fatal warnings (e.g., missing optional features).
    """

    transcript_path: str
    status: Literal["success", "partial", "error"]
    enriched_transcript: "Transcript | None" = None
    error_type: str | None = None
    error_message: str | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize enrichment file result to a JSON-serializable dict.

        Returns:
            Dict with transcript_path, status, error fields, and warnings.
            Enriched transcript is excluded (too large for batch summaries).
        """
        return {
            "transcript_path": self.transcript_path,
            "status": self.status,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "warnings": self.warnings,
        }


@dataclass
class EnrichmentBatchResult:
    """Result of batch enrichment operation across multiple transcript files.

    Aggregates results from enriching multiple transcripts, providing both
    individual file results and summary statistics.

    Attributes:
        total_files: Total number of transcript files attempted.
        successful: Number of files fully enriched without errors.
        partial: Number of files partially enriched (some features missing).
        failed: Number of files that failed enrichment completely.
        results: List of individual EnrichmentFileResult objects for each file.
    """

    total_files: int
    successful: int
    partial: int
    failed: int
    results: list[EnrichmentFileResult] = field(default_factory=list)

    def get_failures(self) -> list[EnrichmentFileResult]:
        """Return only the failed enrichment results.

        Returns:
            List of EnrichmentFileResult objects where status == "error".
        """
        return [r for r in self.results if r.status == "error"]

    def get_transcripts(self) -> list["Transcript"]:
        """Return all successfully enriched transcripts (including partial).

        Returns:
            List of enriched Transcript objects from successful and partial results.
        """
        return [r.enriched_transcript for r in self.results if r.enriched_transcript is not None]

    def to_dict(self) -> dict[str, Any]:
        """Serialize batch enrichment summary to a JSON-serializable dict.

        Returns:
            Dict with counts and list of individual enrichment result summaries.
        """
        return {
            "total_files": self.total_files,
            "successful": self.successful,
            "partial": self.partial,
            "failed": self.failed,
            "results": [r.to_dict() for r in self.results],
        }


@dataclass
class Segment:
    """
    A single segment of transcribed audio.

    This dataclass represents a discrete unit of transcribed speech with optional
    enriched audio feature information. The audio_state field enables storage of
    prosodic, emotional, and voice quality features extracted during preprocessing.

    Attributes:
        id: Integer index of the segment within the transcript.
        start: Start time in seconds.
        end: End time in seconds.
        text: Transcribed text for this segment.
        speaker: Optional speaker metadata dict (v1.1+ diarization).
                 Structure: {"id": "spk_0", "confidence": 0.87}
                 None when diarization is disabled or speaker is unknown.
        tone: Optional tone label (for future tone tagging).
        audio_state: Optional dictionary containing enriched audio features and
                     prosodic information for this segment. This field supports
                     storage of features such as:
                     - prosody features (pitch, energy, duration)
                     - emotional indicators
                     - voice quality metrics
                     - speaker characteristics
                     The structure and content of this dictionary is defined by
                     AUDIO_STATE_VERSION. When None, indicates no audio features
                     have been extracted or enriched for this segment.
        words: Optional list of Word objects with per-word timestamps (v1.8+).
               Populated when word_timestamps=True during transcription.
               Each Word contains: word, start, end, probability, and optional speaker.
               The structure is defined by WORD_ALIGNMENT_VERSION.
    """

    id: int
    start: float
    end: float
    text: str
    speaker: dict[str, Any] | None = None
    tone: str | None = None
    audio_state: dict[str, Any] | None = None
    words: list[Word] | None = None


@dataclass
class Transcript:
    """
    A complete transcript for a single audio file.

    Attributes:
        file_name: Name of the audio file (e.g. "meeting1.wav").
        language: Language code reported by the ASR engine (e.g. "en").
        segments: Ordered list of segments making up the transcript.
        meta: Optional metadata dictionary describing how/when this
              transcript was generated (model, device, etc.).
        speakers: Optional list of speaker metadata (v1.1+).
                  Each speaker dict contains: {id, label, total_speech_time, num_segments}.
                  Null in v1.0 transcripts; empty array if diarization finds no speakers.
        turns: Optional list of conversational turns (v1.1+).
               Each turn dict contains: {speaker_id, start, end, segment_ids, text}.
               Null in v1.0 transcripts; populated after diarization in v1.1+.
        speaker_stats: Optional list of per-speaker aggregates (v1.2 scaffolding).
                       Each entry can be a dict or SpeakerStats dataclass.
    """

    file_name: str
    language: str
    segments: list[Segment] = field(default_factory=list)
    meta: dict[str, Any] | None = None
    annotations: dict[str, Any] | None = None
    speakers: list[dict[str, Any]] | None = None
    turns: list[Turn | dict[str, Any]] | None = None
    speaker_stats: list[SpeakerStats | dict[str, Any]] | None = None
    chunks: list[Chunk | dict[str, Any]] | None = None

    @property
    def full_text(self) -> str:
        """Return concatenated text from all segments.

        Joins all segment text with spaces, stripping leading/trailing whitespace
        from each segment. This provides a convenient way to access the complete
        transcription as a single string.

        Returns:
            The full transcript text as a single string.

        Example:
            >>> transcript.full_text
            'Hello world. How are you today?'
        """
        return " ".join(seg.text.strip() for seg in self.segments if seg.text)

    @property
    def duration(self) -> float:
        """Return total audio duration in seconds.

        Calculated as the end time of the last segment. Returns 0.0 if
        there are no segments.

        Returns:
            Total duration in seconds.

        Example:
            >>> transcript.duration
            125.5
        """
        if not self.segments:
            return 0.0
        return max(seg.end for seg in self.segments)

    def get_segments_by_speaker(self, speaker_id: str) -> list[Segment]:
        """Return all segments for a given speaker.

        This is an alias for segments_by_speaker() provided for API consistency.

        Args:
            speaker_id: The speaker ID to filter by (e.g., "spk_0", "SPEAKER_01").

        Returns:
            List of Segment objects belonging to the specified speaker.
            Empty list if no segments match or diarization was not performed.

        Example:
            >>> segments = transcript.get_segments_by_speaker("spk_0")
            >>> for seg in segments:
            ...     print(f"{seg.start:.2f}s: {seg.text}")
        """
        return self.segments_by_speaker(speaker_id)

    def get_segment_at_time(self, time: float) -> Segment | None:
        """Return the segment containing the given timestamp.

        Finds the first segment where start <= time < end. If no segment
        contains the exact timestamp, returns None.

        Args:
            time: Timestamp in seconds to search for.

        Returns:
            The Segment containing the timestamp, or None if not found.

        Example:
            >>> segment = transcript.get_segment_at_time(10.5)
            >>> if segment:
            ...     print(f"At 10.5s: {segment.text}")
        """
        for seg in self.segments:
            if seg.start <= time < seg.end:
                return seg
        return None

    def word_count(self) -> int:
        """Total word count across all segments.

        Uses word-level timestamps (seg.words) when available for accurate
        counts. Falls back to text.split() when words are not present.
        """
        total = 0
        for seg in self.segments:
            words = getattr(seg, "words", None)
            if words:
                total += len(words)
            elif seg.text:
                total += len(seg.text.split())
        return total

    def speaker_ids(self) -> list[str]:
        """List of unique speaker IDs found in segments.

        Handles multiple speaker representations:
        - dict with "id" key: {"id": "spk_0", "confidence": 0.9}
        - plain string: "spk_0"
        """
        from .speaker_id import get_speaker_id

        ids: set[str] = set()
        for seg in self.segments:
            speaker = getattr(seg, "speaker", None)
            if speaker is not None:
                sid = get_speaker_id(speaker)
                if sid:
                    ids.add(sid)
        return sorted(ids)

    def is_enriched(self) -> bool:
        """True if audio enrichment has been applied to any segment."""
        return any(seg.audio_state is not None for seg in self.segments)

    def segments_by_speaker(self, speaker_id: str) -> list[Segment]:
        """Filter segments by speaker ID.

        Handles multiple speaker representations:
        - dict with "id" key: {"id": "spk_0", "confidence": 0.9}
        - plain string: "spk_0"
        """
        from .speaker_id import get_speaker_id

        return [
            seg
            for seg in self.segments
            if get_speaker_id(getattr(seg, "speaker", None)) == speaker_id
        ]

    def segments_in_range(self, start: float, end: float) -> list[Segment]:
        """Get segments that overlap with the given time range."""
        return [seg for seg in self.segments if seg.end > start and seg.start < end]
