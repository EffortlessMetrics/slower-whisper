"""
Speaker diarization for slower-whisper.

This module provides speaker diarization (who spoke when) to populate
the `speakers[]` and `segment.speaker` fields in the transcript schema.

**Status**: v1.1 - IMPLEMENTED

**Approach** (WhisperX-style):
1. Run pyannote.audio speaker diarization on normalized WAV
2. Align diarization timestamps with existing ASR segments
3. Assign speaker labels based on maximum overlap
4. Build global speakers table with aggregate stats

**Design constraints**:
- Never re-run ASR (operates on existing transcript + audio)
- Cacheable by hash (audio hash + diarization config)
- Graceful degradation (missing speakers → segment.speaker = None)
- Multi-speaker baselines for prosody (per-speaker pitch/energy normalization)

**Requirements**:
- Install with: uv sync --extra diarization
- Requires HF_TOKEN environment variable for pyannote model access
- Accept license at: https://huggingface.co/pyannote/speaker-diarization-3.1
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from transcription.models import Transcript


@dataclass
class SpeakerTurn:
    """
    A contiguous time interval attributed to a single speaker.

    This represents the output of the diarization model before alignment
    with ASR segments.

    Attributes:
        start: Start time in seconds.
        end: End time in seconds.
        speaker_id: Speaker identifier (e.g., "SPEAKER_00", "SPEAKER_01").
        confidence: Optional confidence score from diarization model (0.0-1.0).
    """

    start: float
    end: float
    speaker_id: str
    confidence: float | None = None


@dataclass
class SpeakerInfo:
    """
    Global speaker metadata for the speakers[] array in schema.

    Attributes:
        id: Speaker identifier matching segment.speaker field.
        label: Human-readable label (defaults to id).
        total_speech_time: Total seconds attributed to this speaker.
        num_segments: Number of ASR segments attributed to this speaker.
        confidence_mean: Mean confidence across all attributed segments.
    """

    id: str
    label: str | None = None
    total_speech_time: float = 0.0
    num_segments: int = 0
    confidence_mean: float | None = None


class Diarizer:
    """
    Speaker diarization engine using pyannote.audio.

    **Implementation**:
    - Uses pyannote/speaker-diarization-3.1 (https://github.com/pyannote/pyannote-audio)
    - GPU acceleration when available
    - Fallback to CPU (slower but functional)
    - Lazy pipeline initialization (only loads model on first run() call)

    **Usage**:
        ```python
        diarizer = Diarizer(device="cuda", min_speakers=2, max_speakers=2)
        speaker_turns = diarizer.run("path/to/audio.wav")
        # Returns list of SpeakerTurn with (start, end, speaker_id)
        ```

    **Requirements**:
    - Install with: uv sync --extra diarization
    - Set HF_TOKEN environment variable
    - Accept model license at https://huggingface.co/pyannote/speaker-diarization-3.1

    Args:
        device: Device for inference ("cuda", "cpu", or "auto").
        min_speakers: Minimum number of speakers (None = auto-detect).
        max_speakers: Maximum number of speakers (None = auto-detect).
    """

    def __init__(
        self,
        device: str = "auto",
        min_speakers: int | None = None,
        max_speakers: int | None = None,
    ):
        self.device = device
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self._pipeline = None  # Will hold pyannote pipeline

    def _ensure_pipeline(self):
        """
        Lazy-load pyannote pipeline on first use.

        This avoids expensive model loading unless diarization is actually used.

        Raises:
            ImportError: If pyannote.audio not installed.
            RuntimeError: If HuggingFace token not configured or model load fails.
        """
        if self._pipeline is not None:
            return self._pipeline

        try:
            from pyannote.audio import Pipeline
        except ImportError as exc:
            raise ImportError(
                "pyannote.audio is required for speaker diarization. "
                "Install with: uv sync --extra diarization"
            ) from exc

        # Load pipeline (this will raise if HF_TOKEN not set)
        # Device handling: pyannote expects torch device strings ("cuda", "cpu", etc.)
        device_str = None if self.device == "auto" else self.device

        try:
            self._pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=True,  # Uses HF_TOKEN env var
            )

            if device_str:
                import torch

                self._pipeline = self._pipeline.to(torch.device(device_str))

        except Exception as exc:
            raise RuntimeError(
                f"Failed to load pyannote pipeline: {exc}. "
                "Ensure HF_TOKEN environment variable is set and you have accepted "
                "the model license at https://huggingface.co/pyannote/speaker-diarization-3.1"
            ) from exc

        return self._pipeline

    def run(self, audio_path: Path | str) -> list[SpeakerTurn]:
        """
        Run speaker diarization on audio file.

        Args:
            audio_path: Path to WAV file (must be 16kHz mono, normalized).

        Returns:
            List of SpeakerTurn objects with speaker labels and timestamps.

        Raises:
            FileNotFoundError: If audio_path does not exist.
            ImportError: If pyannote.audio not installed.
            RuntimeError: If diarization fails (missing HF token, model error, etc.).
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Load pipeline (lazy initialization)
        pipeline = self._ensure_pipeline()

        # Run diarization
        # pyannote returns an Annotation object with segments
        diarization_result = pipeline(
            str(audio_path),
            min_speakers=self.min_speakers,
            max_speakers=self.max_speakers,
        )

        # Convert pyannote output to list[SpeakerTurn]
        turns: list[SpeakerTurn] = []
        for segment, _, label in diarization_result.itertracks(yield_label=True):
            turns.append(
                SpeakerTurn(
                    start=float(segment.start),
                    end=float(segment.end),
                    speaker_id=str(label),
                    confidence=None,  # pyannote doesn't provide per-segment confidence
                )
            )

        # Sort by start time (pyannote usually returns sorted, but be explicit)
        turns.sort(key=lambda t: t.start)

        return turns


def _normalize_speaker_id(raw_id: str, speaker_map: dict[str, int]) -> str:
    """
    Map backend speaker IDs to spk_N format.

    Args:
        raw_id: Raw speaker ID from backend (e.g., "SPEAKER_00").
        speaker_map: Mutable mapping from raw_id to integer index.

    Returns:
        Normalized speaker ID (e.g., "spk_0").
    """
    if raw_id not in speaker_map:
        speaker_map[raw_id] = len(speaker_map)
    return f"spk_{speaker_map[raw_id]}"


def _compute_overlap(seg_start: float, seg_end: float, turn_start: float, turn_end: float) -> float:
    """
    Compute overlap duration between segment and speaker turn.

    Args:
        seg_start: Segment start time.
        seg_end: Segment end time.
        turn_start: Speaker turn start time.
        turn_end: Speaker turn end time.

    Returns:
        Overlap duration in seconds (0.0 if no overlap).
    """
    overlap_start = max(seg_start, turn_start)
    overlap_end = min(seg_end, turn_end)
    return max(0.0, overlap_end - overlap_start)


def assign_speakers(
    transcript: Transcript,
    speaker_turns: list[SpeakerTurn],
    overlap_threshold: float = 0.3,
) -> Transcript:
    """
    Assign speaker labels to ASR segments based on diarization output.

    This function aligns diarization timestamps with existing ASR segments
    and populates the segment.speaker field + builds the speakers[] array.

    **Assignment logic** (v1.1 contract):
    1. For each ASR segment [s_start, s_end]:
       - Compute overlap duration with all speaker turns
       - Choose speaker with **maximum overlap duration**
       - Compute overlap_ratio = overlap_duration / segment_duration
    2. If overlap_ratio >= threshold: assign speaker
    3. Else: segment.speaker = None (unknown)

    **Edge cases**:
    - No overlap: segment.speaker = None
    - Equal overlap: choose first speaker ID alphabetically (deterministic)
    - Zero-duration segment: segment.speaker = None

    **Mutates transcript in-place** and returns it for convenience.

    Args:
        transcript: Transcript with segments from ASR.
        speaker_turns: List of SpeakerTurn from diarization (sorted or unsorted).
        overlap_threshold: Minimum overlap ratio to assign speaker (default 0.3).

    Returns:
        Updated transcript with:
        - segment.speaker populated (or None if low confidence)
        - transcript.speakers list built from unique speaker IDs

    Example:
        >>> from transcription.models import Transcript, Segment
        >>> transcript = Transcript(
        ...     file_name="test.wav",
        ...     language="en",
        ...     segments=[Segment(id=0, start=0.0, end=2.0, text="Hello")],
        ... )
        >>> turns = [SpeakerTurn(start=0.0, end=5.0, speaker_id="SPEAKER_00")]
        >>> result = assign_speakers(transcript, turns)
        >>> result.segments[0].speaker
        {'id': 'spk_0', 'confidence': 1.0}
        >>> result.speakers
        [{'id': 'spk_0', 'label': None, 'total_speech_time': 2.0, 'num_segments': 1}]
    """

    # Normalize speaker IDs: backend IDs → spk_N
    speaker_map: dict[str, int] = {}

    # Per-speaker aggregates for building speakers[] array
    speaker_stats: dict[str, dict[str, Any]] = {}

    # Assign speakers to each segment
    for segment in transcript.segments:
        seg_start = segment.start
        seg_end = segment.end
        seg_duration = seg_end - seg_start

        if seg_duration <= 0:
            # Zero or negative duration → skip assignment
            segment.speaker = None
            continue

        # Find best speaker by max overlap
        best_speaker_id: str | None = None
        max_overlap_duration = 0.0

        for turn in speaker_turns:
            overlap_duration = _compute_overlap(seg_start, seg_end, turn.start, turn.end)

            if overlap_duration > max_overlap_duration:
                max_overlap_duration = overlap_duration
                best_speaker_id = turn.speaker_id
            elif overlap_duration == max_overlap_duration and overlap_duration > 0:
                # Equal overlap: choose first alphabetically (deterministic)
                if best_speaker_id is None or turn.speaker_id < best_speaker_id:
                    best_speaker_id = turn.speaker_id

        # Compute confidence = overlap_ratio
        overlap_ratio = max_overlap_duration / seg_duration if seg_duration > 0 else 0.0

        # Assign if above threshold
        if overlap_ratio >= overlap_threshold and best_speaker_id is not None:
            normalized_id = _normalize_speaker_id(best_speaker_id, speaker_map)
            segment.speaker = {"id": normalized_id, "confidence": overlap_ratio}

            # Update speaker stats
            if normalized_id not in speaker_stats:
                speaker_stats[normalized_id] = {
                    "id": normalized_id,
                    "label": None,
                    "total_speech_time": 0.0,
                    "num_segments": 0,
                }
            speaker_stats[normalized_id]["total_speech_time"] += seg_duration
            speaker_stats[normalized_id]["num_segments"] += 1
        else:
            segment.speaker = None

    # Build speakers[] array
    transcript.speakers = sorted(speaker_stats.values(), key=lambda s: s["id"])

    return transcript


def assign_speakers_to_segments(
    segments: list[dict[str, Any]],
    speaker_turns: list[SpeakerTurn],
    overlap_threshold: float = 0.5,
) -> tuple[list[dict[str, Any]], list[SpeakerInfo]]:
    """
    **DEPRECATED**: Use `assign_speakers(transcript, turns)` instead.

    Legacy function signature for backward compatibility.
    This will be removed in v1.2.

    Args:
        segments: List of segment dicts from Transcript (with id, start, end, text).
        speaker_turns: List of SpeakerTurn from diarization.
        overlap_threshold: Minimum overlap ratio to assign speaker (0.0-1.0).

    Returns:
        Tuple of (updated_segments, speakers_info):
        - updated_segments: Segments with speaker field populated
        - speakers_info: List of SpeakerInfo for global speakers[] array

    Raises:
        NotImplementedError: Use `assign_speakers()` instead.
    """
    raise NotImplementedError(
        "assign_speakers_to_segments() is deprecated. "
        "Use assign_speakers(transcript, speaker_turns) instead. "
        "See docs/SPEAKER_DIARIZATION.md for updated API."
    )
