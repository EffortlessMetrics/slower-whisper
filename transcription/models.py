from dataclasses import dataclass, field
from typing import Any, Literal

SCHEMA_VERSION: int = 2
AUDIO_STATE_VERSION: str = "1.0.0"


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
    """

    id: str
    start: float
    end: float
    segment_ids: list[int] = field(default_factory=list)
    turn_ids: list[str] = field(default_factory=list)
    speaker_ids: list[str] = field(default_factory=list)
    token_count_estimate: int = 0
    text: str = ""

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
    """

    id: int
    start: float
    end: float
    text: str
    speaker: dict[str, Any] | None = None
    tone: str | None = None
    audio_state: dict[str, Any] | None = None


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
