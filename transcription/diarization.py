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
- HF_TOKEN required only when using the real pyannote backend (auto)
- Accept license at: https://huggingface.co/pyannote/speaker-diarization-3.1
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from transcription.models import Transcript

logger = logging.getLogger(__name__)


def _torchcodec_available() -> bool:
    """Detect whether torchcodec can decode audio for pyannote."""
    try:
        from torchcodec.decoders import AudioDecoder
    except Exception as exc:  # noqa: BLE001
        logger.debug("torchcodec unavailable for pyannote input: %s", exc)
        return False
    return AudioDecoder is not None


def _load_waveform_input(audio_path: Path) -> dict[str, Any]:
    """Load audio as in-memory waveform for pyannote when torchcodec is missing."""
    import soundfile as sf
    import torch

    data, sample_rate = sf.read(str(audio_path), always_2d=True, dtype="float32")
    waveform = torch.from_numpy(data.T).contiguous()
    return {"waveform": waveform, "sample_rate": int(sample_rate), "uri": audio_path.stem}


def _make_stub_pyannote_pipeline():
    """
    Lightweight pyannote-like pipeline used for tests when the real dependency
    is unavailable. Generates an alternating 2-speaker pattern.
    """

    def _infer_duration_seconds(audio_path: str | os.PathLike[str]) -> float:
        try:
            import soundfile as sf

            with sf.SoundFile(audio_path, "r") as f:
                if f.samplerate:
                    return float(max(len(f) / f.samplerate, 0.5))
        except Exception as e:
            # Log why duration detection failed (helps debug stub mode issues)
            logger.debug(
                "Could not infer audio duration from %s: %s. Using fallback of 4.0s.",
                audio_path,
                e,
            )
        return 4.0

    class _StubSegment:
        def __init__(self, start: float, end: float):
            self.start = start
            self.end = end

    class _StubAnnotation:
        def __init__(self, segments: list[tuple[_StubSegment, str]]):
            # Mirror newer pyannote outputs with speaker_diarization attribute
            self.speaker_diarization = self
            self._segments = segments

        def itertracks(self, yield_label: bool = True):
            for seg, label in self._segments:
                yield seg, None, label

    class _StubPipeline:
        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            return cls()

        def to(self, _device):
            return self

        def __call__(self, audio_path, **_kwargs):
            duration = _infer_duration_seconds(audio_path)
            midpoint = max(duration / 2.0, 0.5)
            segments = [
                (_StubSegment(0.0, midpoint), "SPEAKER_00"),
                (_StubSegment(midpoint, max(duration, midpoint + 0.1)), "SPEAKER_01"),
            ]
            return _StubAnnotation(segments)

    return _StubPipeline()


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
        self._pipeline: Any | None = None  # Will hold pyannote pipeline
        self._pipeline_is_stub: bool = False
        self._pipeline_lock = threading.Lock()  # Thread-safety for lazy loading

    def _ensure_pipeline(self):
        """
        Lazy-load pyannote pipeline on first use.

        This avoids expensive model loading unless diarization is actually used.
        Thread-safe: Uses double-checked locking to avoid race conditions.

        Raises:
            ImportError: If pyannote.audio not installed.
            RuntimeError: If HuggingFace token not configured or model load fails.
        """
        if self._pipeline is not None:
            return self._pipeline

        with self._pipeline_lock:
            # Double-check after acquiring lock (thread-safe pattern for concurrency)
            if self._pipeline is not None:
                return self._pipeline  # type: ignore[unreachable]

            mode = os.getenv("SLOWER_WHISPER_PYANNOTE_MODE", "auto").lower()
            if mode == "missing":
                raise ImportError(
                    "pyannote.audio is unavailable (forced via SLOWER_WHISPER_PYANNOTE_MODE=missing)"
                )
            if mode == "stub":
                self._pipeline = _make_stub_pyannote_pipeline()
                self._pipeline_is_stub = True
                return self._pipeline

            token = os.getenv("HF_TOKEN")
            if token is None:
                raise RuntimeError(
                    "HF_TOKEN environment variable is required for pyannote.audio diarization"
                )

            # Suppress torchcodec/FFmpeg warnings from pyannote.audio's torchaudio import chain.
            from transcription._import_guards import suppress_optional_dependency_warnings

            suppress_optional_dependency_warnings()

            try:
                from pyannote.audio import Pipeline
            except ImportError as exc:
                raise ImportError(
                    "pyannote.audio is required for speaker diarization. "
                    "Install with: uv sync --extra diarization"
                ) from exc

            # Load pipeline - device handling: pyannote expects torch device strings
            device_str = None if self.device == "auto" else self.device

            try:
                from .cache import CachePaths

                paths = CachePaths.from_env().ensure_dirs()
                token_arg = os.getenv("HF_TOKEN") or True
                from_pretrained: Callable[..., Any] = cast(
                    Callable[..., Any], Pipeline.from_pretrained
                )
                model_id = os.getenv(
                    "SLOWER_WHISPER_PYANNOTE_MODEL", "pyannote/speaker-diarization-3.1"
                )

                try:
                    pipeline = from_pretrained(
                        model_id,
                        token=token_arg,
                        cache_dir=str(paths.diarization_root),
                    )
                except TypeError as exc:
                    # Backward compatibility for older pyannote.audio versions
                    logger.debug(
                        "Pipeline.from_pretrained() rejected token=..., retrying with use_auth_token",
                        exc_info=exc,
                    )
                    pipeline = from_pretrained(
                        model_id,
                        use_auth_token=token_arg,
                        cache_dir=str(paths.diarization_root),
                    )

                if device_str and pipeline is not None:
                    import torch

                    pipeline = pipeline.to(torch.device(device_str))

                self._pipeline = pipeline
                self._pipeline_is_stub = False

            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load pyannote pipeline ({model_id}): {exc}. "
                    "Ensure HF_TOKEN environment variable is set and you have accepted "
                    "the model license for the selected model (default: "
                    "https://huggingface.co/pyannote/speaker-diarization-3.1)"
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

        # torchcodec is flaky on some platforms; pre-load waveform to bypass it when missing.
        use_waveform_input = not self._pipeline_is_stub and not _torchcodec_available()
        file_input: Any = str(audio_path)
        if use_waveform_input:
            try:
                file_input = _load_waveform_input(audio_path)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Falling back to file path for pyannote input: %s", exc)
                file_input = str(audio_path)

        # Run diarization
        # pyannote returns different output types depending on version:
        # - Older versions: Annotation object with itertracks()
        # - v3.1+: DiarizeOutput wrapper with .speaker_diarization attribute (Annotation)
        diarization_result = pipeline(
            file_input,
            min_speakers=self.min_speakers,
            max_speakers=self.max_speakers,
        )

        # Convert pyannote output to list[SpeakerTurn]
        # Handle both old (Annotation) and new (DiarizeOutput) API
        turns: list[SpeakerTurn] = []

        # Extract Annotation object from result
        if hasattr(diarization_result, "speaker_diarization"):
            # New API (v3.1+): DiarizeOutput.speaker_diarization
            annotation = diarization_result.speaker_diarization
        else:
            # Old API: result is already an Annotation
            annotation = diarization_result

        # Iterate over annotation tracks
        for segment, _, label in annotation.itertracks(yield_label=True):
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

    # Sort turns by start time for optimized search (copy to avoid mutating caller's list)
    sorted_turns = sorted(speaker_turns, key=lambda t: t.start)

    # State for sliding window optimization
    turn_idx = 0
    prev_seg_start = -1.0

    # Assign speakers to each segment
    for segment in transcript.segments:
        seg_start = segment.start
        seg_end = segment.end
        seg_duration = seg_end - seg_start

        if seg_duration <= 0:
            # Zero or negative duration → skip assignment
            segment.speaker = None
            continue

        # Reset optimization if segments are not sorted (safety fallback)
        if seg_start < prev_seg_start:
            turn_idx = 0
        prev_seg_start = seg_start

        # Advance window: skip turns that end before this segment starts.
        # Since sorted_turns is sorted by start, and segments are generally sorted by start,
        # we can safely permanently skip turns that have fully passed.
        while turn_idx < len(sorted_turns) and sorted_turns[turn_idx].end <= seg_start:
            turn_idx += 1

        # Find best speaker by max overlap
        best_speaker_id: str | None = None
        max_overlap_duration = 0.0

        # Only iterate relevant turns starting from current window
        for i in range(turn_idx, len(sorted_turns)):
            turn = sorted_turns[i]

            # Optimization: Since turns are sorted by start time,
            # if a turn starts after the segment ends, all subsequent turns also will.
            if turn.start >= seg_end:
                break

            # Optimization: Skip turns that end before the segment starts
            # (still needed for turns inside the window that might end early)
            if turn.end <= seg_start:
                continue

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


def assign_speakers_to_words(
    transcript: Transcript,
    speaker_turns: list[SpeakerTurn],
    overlap_threshold: float = 0.3,
) -> Transcript:
    """
    Assign speaker labels to individual words within segments (v1.8+).

    This function provides more granular speaker assignment than segment-level
    alignment. It's particularly useful for:
    - Detecting speaker changes within a segment
    - Precise speaker attribution in overlapping speech
    - Improving downstream turn detection accuracy

    **Assignment logic**:
    1. For each word [w_start, w_end]:
       - Compute overlap duration with all speaker turns
       - Choose speaker with maximum overlap duration
       - Compute overlap_ratio = overlap_duration / word_duration
    2. If overlap_ratio >= threshold: assign speaker to word
    3. Else: word.speaker = None

    **Note**: This function also updates segment.speaker based on the
    dominant word-level speaker (most common speaker among words).

    **Mutates transcript in-place** and returns it for convenience.

    Args:
        transcript: Transcript with segments that have word-level timestamps.
        speaker_turns: List of SpeakerTurn from diarization.
        overlap_threshold: Minimum overlap ratio to assign speaker (default 0.3).

    Returns:
        Updated transcript with:
        - word.speaker populated for each word (or None if low confidence)
        - segment.speaker derived from dominant word-level speaker
        - transcript.speakers list built from unique speaker IDs
    """

    # Normalize speaker IDs: backend IDs → spk_N
    speaker_map: dict[str, int] = {}

    # Per-speaker aggregates
    speaker_stats: dict[str, dict[str, Any]] = {}

    # Sort turns by start time for optimized search (copy to avoid mutating caller's list)
    sorted_turns = sorted(speaker_turns, key=lambda t: t.start)

    # State for sliding window optimization
    turn_idx = 0
    prev_seg_start = -1.0

    for segment in transcript.segments:
        # Reset optimization if segments are not sorted (safety fallback)
        if segment.start < prev_seg_start:
            turn_idx = 0
        prev_seg_start = segment.start

        # Advance window: skip turns that end before this segment starts
        while turn_idx < len(sorted_turns) and sorted_turns[turn_idx].end <= segment.start:
            turn_idx += 1

        # Optimization: Filter turns relevant to this segment once
        # This drastically reduces the number of turns checked for each word.
        # Since sorted_turns is sorted, we can use early break in iteration.
        relevant_turns: list[SpeakerTurn] = []
        for i in range(turn_idx, len(sorted_turns)):
            t = sorted_turns[i]
            if t.start >= segment.end:
                break
            # Note: t.end > segment.start check is still needed because turns sorted by
            # start time might have ends out of order (e.g. nested turns).
            if t.end > segment.start:
                relevant_turns.append(t)

        if not segment.words:
            # Fall back to segment-level assignment if no words
            seg_duration = segment.end - segment.start
            if seg_duration <= 0:
                segment.speaker = None
                continue

            best_speaker_id, max_overlap = _find_best_speaker(
                segment.start, segment.end, relevant_turns
            )
            overlap_ratio = max_overlap / seg_duration if seg_duration > 0 else 0.0

            if overlap_ratio >= overlap_threshold and best_speaker_id is not None:
                normalized_id = _normalize_speaker_id(best_speaker_id, speaker_map)
                segment.speaker = {"id": normalized_id, "confidence": overlap_ratio}
                _update_speaker_stats(speaker_stats, normalized_id, seg_duration)
            else:
                segment.speaker = None
            continue

        # Word-level speaker assignment
        word_speaker_counts: dict[str, int] = {}
        word_durations_by_speaker: dict[str, float] = {}

        for word in segment.words:
            word_duration = word.end - word.start
            if word_duration <= 0:
                continue

            best_speaker_id, max_overlap = _find_best_speaker(word.start, word.end, relevant_turns)
            overlap_ratio = max_overlap / word_duration if word_duration > 0 else 0.0

            if overlap_ratio >= overlap_threshold and best_speaker_id is not None:
                normalized_id = _normalize_speaker_id(best_speaker_id, speaker_map)
                word.speaker = normalized_id
                word_speaker_counts[normalized_id] = word_speaker_counts.get(normalized_id, 0) + 1
                word_durations_by_speaker[normalized_id] = (
                    word_durations_by_speaker.get(normalized_id, 0.0) + word_duration
                )
            else:
                word.speaker = None

        # Update speaker stats for ALL word-level speakers
        for spk_id, duration in word_durations_by_speaker.items():
            _update_speaker_stats(speaker_stats, spk_id, duration)

        # Derive segment speaker from dominant word-level speaker
        if word_speaker_counts:
            # Choose speaker with most word duration (not just count)
            dominant_speaker = max(word_durations_by_speaker.items(), key=lambda x: x[1])[0]
            seg_duration = segment.end - segment.start
            # Confidence = ratio of dominant speaker's word duration to segment duration
            confidence = (
                word_durations_by_speaker[dominant_speaker] / seg_duration
                if seg_duration > 0
                else 0.0
            )
            segment.speaker = {"id": dominant_speaker, "confidence": min(1.0, confidence)}
        else:
            segment.speaker = None

    # Build speakers[] array
    transcript.speakers = sorted(speaker_stats.values(), key=lambda s: s["id"])

    return transcript


def _find_best_speaker(
    start: float, end: float, speaker_turns: list[SpeakerTurn]
) -> tuple[str | None, float]:
    """Find speaker with maximum overlap for a time range.

    Assumes speaker_turns is sorted by start time in ascending order. Any subsets
    passed to this helper must preserve this ordering (e.g., a slice or filtered
    view of an already-sorted list) so that the early-exit optimization remains valid.
    """
    best_speaker_id: str | None = None
    max_overlap_duration = 0.0

    for turn in speaker_turns:
        # Optimization: Since turns are sorted by start time,
        # if a turn starts after the range ends, all subsequent turns will also start after.
        if turn.start >= end:
            break

        # Optimization: Skip turns that end before the range starts
        if turn.end <= start:
            continue

        overlap_duration = _compute_overlap(start, end, turn.start, turn.end)

        if overlap_duration > max_overlap_duration:
            max_overlap_duration = overlap_duration
            best_speaker_id = turn.speaker_id
        elif overlap_duration == max_overlap_duration and overlap_duration > 0:
            # Equal overlap: choose first alphabetically (deterministic)
            if best_speaker_id is None or turn.speaker_id < best_speaker_id:
                best_speaker_id = turn.speaker_id

    return best_speaker_id, max_overlap_duration


def _update_speaker_stats(
    stats: dict[str, dict[str, Any]], speaker_id: str, duration: float
) -> None:
    """Update speaker statistics for a speaker."""
    if speaker_id not in stats:
        stats[speaker_id] = {
            "id": speaker_id,
            "label": None,
            "total_speech_time": 0.0,
            "num_segments": 0,
        }
    stats[speaker_id]["total_speech_time"] += duration
    stats[speaker_id]["num_segments"] += 1


def assign_speakers_to_segments(
    segments: list[dict[str, Any]],
    speaker_turns: list[SpeakerTurn],
    overlap_threshold: float = 0.3,
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
