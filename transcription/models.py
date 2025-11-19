from dataclasses import dataclass, field
from typing import Any

SCHEMA_VERSION: int = 2
AUDIO_STATE_VERSION: str = "1.0.0"


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
    """

    file_name: str
    language: str
    segments: list[Segment] = field(default_factory=list)
    meta: dict[str, Any] | None = None
    speakers: list[dict[str, Any]] | None = None
    turns: list[dict[str, Any]] | None = None
