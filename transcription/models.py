from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

SCHEMA_VERSION: int = 1


@dataclass
class Segment:
    """
    A single segment of transcribed audio.

    Attributes:
        id: Integer index of the segment within the transcript.
        start: Start time in seconds.
        end: End time in seconds.
        text: Transcribed text for this segment.
        speaker: Optional speaker label (for future diarization).
        tone: Optional tone label (for future tone tagging).
    """
    id: int
    start: float
    end: float
    text: str
    speaker: Optional[str] = None
    tone: Optional[str] = None


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
    """
    file_name: str
    language: str
    segments: List[Segment] = field(default_factory=list)
    meta: Optional[Dict[str, Any]] = None
