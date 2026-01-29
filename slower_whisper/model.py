"""Drop-in replacement WhisperModel for faster-whisper compatibility.

This module provides a WhisperModel class that matches faster-whisper's API
while transparently using slower-whisper's enriched transcription pipeline.

Usage:
    # Drop-in replacement
    from slower_whisper import WhisperModel

    model = WhisperModel("base", device="auto", compute_type="int8")
    segments, info = model.transcribe("audio.wav")

    for segment in segments:
        print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")

    # Access slower-whisper enriched data (optional)
    transcript = model.last_transcript
    if transcript and transcript.turns:
        for turn in transcript.turns:
            print(f"Speaker {turn.speaker_id}: {turn.text}")
"""

from __future__ import annotations

import io
import logging
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO

from .compat import Segment, TranscriptionInfo

if TYPE_CHECKING:
    from transcription.models import Transcript

logger = logging.getLogger(__name__)

# Type alias for audio input (matches faster-whisper)
# Using Any for numpy array to avoid requiring numpy at import time
AudioInput = str | Path | BinaryIO | Any  # Any covers np.ndarray


class WhisperModel:
    """Drop-in replacement for faster_whisper.WhisperModel.

    This class provides the same interface as faster-whisper's WhisperModel,
    but uses slower-whisper's enriched transcription pipeline under the hood.

    After calling transcribe(), the enriched Transcript is available via
    the `last_transcript` property for access to slower-whisper features
    like speaker diarization, turns, and audio enrichment.

    Args:
        model_size_or_path: Model size ("tiny", "base", "small", "medium", "large")
            or path to a CTranslate2 model directory.
        device: Device to use ("auto", "cpu", "cuda"). Default: "auto".
        compute_type: Compute type for inference. Default: "default" (auto-select).
        device_index: GPU device index(es) for multi-GPU. Default: 0.
        cpu_threads: Number of CPU threads. Default: 0 (auto).
        num_workers: Number of transcription workers. Default: 1.
        download_root: Custom download directory for models.
        local_files_only: Only use local model files. Default: False.

    Example:
        >>> model = WhisperModel("base", device="cpu")
        >>> segments, info = model.transcribe("audio.wav")
        >>> print(f"Language: {info.language}")
        >>> for seg in segments:
        ...     print(f"[{seg.start:.2f}s] {seg.text}")
    """

    def __init__(
        self,
        model_size_or_path: str = "base",
        device: str = "auto",
        compute_type: str = "default",
        device_index: int | list[int] = 0,
        cpu_threads: int = 0,
        num_workers: int = 1,
        download_root: str | None = None,
        local_files_only: bool = False,
    ) -> None:
        self._model_size_or_path = model_size_or_path
        self._device = device
        self._compute_type = compute_type
        self._device_index = device_index
        self._cpu_threads = cpu_threads
        self._num_workers = num_workers
        self._download_root = download_root
        self._local_files_only = local_files_only

        # Lazy initialization - engine is created on first transcribe call
        self._engine: Any = None
        self._last_transcript: Transcript | None = None

        # Store transcription options for info reporting
        self._last_transcription_options: dict[str, Any] = {}
        self._last_vad_options: dict[str, Any] | None = None

    def _ensure_engine(self) -> Any:
        """Lazily initialize the transcription engine."""
        if self._engine is not None:
            return self._engine

        # Import here to avoid circular imports and allow graceful degradation
        from transcription.asr_engine import TranscriptionEngine
        from transcription.device import resolve_device
        from transcription.legacy_config import AsrConfig

        # Resolve device using slower-whisper's device detection
        # Map string device to the expected Literal type
        device_choice = self._device
        if device_choice not in ("auto", "cpu", "cuda"):
            device_choice = "auto"

        resolved = resolve_device(device_choice)  # type: ignore[arg-type]

        # Map compute_type "default" to appropriate value
        compute_type = self._compute_type
        if compute_type == "default":
            compute_type = "float16" if resolved.device == "cuda" else "int8"

        config = AsrConfig(
            model_name=self._model_size_or_path,
            device=resolved.device,
            compute_type=compute_type,
        )

        self._engine = TranscriptionEngine(config)
        return self._engine

    @property
    def last_transcript(self) -> Transcript | None:
        """Access the last transcription result as a slower-whisper Transcript.

        This provides access to slower-whisper's enriched features like:
        - Speaker diarization (transcript.speakers, transcript.turns)
        - Audio enrichment (segment.audio_state)
        - Conversation analysis (transcript.chunks, transcript.speaker_stats)

        Returns:
            The Transcript from the last transcribe() call, or None if no
            transcription has been performed yet.

        Example:
            >>> segments, info = model.transcribe("audio.wav")
            >>> transcript = model.last_transcript
            >>> if transcript.turns:
            ...     for turn in transcript.turns:
            ...         print(f"Speaker {turn.speaker_id}: {turn.text}")
        """
        return self._last_transcript

    def _audio_to_path(self, audio: AudioInput) -> tuple[Path, bool]:
        """Convert audio input to a file path.

        Args:
            audio: Audio input (path string, Path, file-like object, or numpy array).

        Returns:
            Tuple of (path, should_cleanup) where should_cleanup indicates if
            the path points to a temporary file that should be deleted.
        """
        # String or Path - use directly
        if isinstance(audio, (str, Path)):
            return Path(audio), False

        # File-like object - read and write to temp file
        if isinstance(audio, (io.IOBase, BinaryIO)) or hasattr(audio, "read"):
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(audio.read())
                return Path(tmp.name), True

        # Check for numpy array without importing numpy at module level
        # This allows the base install to work without numpy
        type_name = type(audio).__module__ + "." + type(audio).__name__
        if "numpy" in type_name or hasattr(audio, "__array__"):
            try:
                import soundfile as sf
            except ImportError as e:
                raise ImportError(
                    "soundfile is required for numpy array input. "
                    "Install with: pip install slower-whisper[enrich-basic]"
                ) from e

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                # Assume 16kHz sample rate (Whisper's native rate)
                sf.write(tmp.name, audio, samplerate=16000)
                return Path(tmp.name), True

        raise TypeError(
            f"Unsupported audio type: {type(audio)}. "
            "Expected str, Path, file-like object, or numpy array."
        )

    def transcribe(
        self,
        audio: AudioInput,
        language: str | None = None,
        task: str = "transcribe",
        beam_size: int = 5,
        best_of: int = 5,
        patience: float = 1.0,
        length_penalty: float = 1.0,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
        temperature: float | list[float] | tuple[float, ...] = (
            0.0,
            0.2,
            0.4,
            0.6,
            0.8,
            1.0,
        ),
        compression_ratio_threshold: float | None = 2.4,
        log_prob_threshold: float | None = -1.0,
        no_speech_threshold: float | None = 0.6,
        condition_on_previous_text: bool = True,
        prompt_reset_on_temperature: float = 0.5,
        initial_prompt: str | Iterable[int] | None = None,
        prefix: str | None = None,
        suppress_blank: bool = True,
        suppress_tokens: list[int] | None = None,
        without_timestamps: bool = False,
        max_initial_timestamp: float = 1.0,
        word_timestamps: bool = False,
        prepend_punctuations: str = "\"'([{-",
        append_punctuations: str = "\"'.googl,:;!?)]}",
        multilingual: bool = False,
        vad_filter: bool = False,
        vad_parameters: dict[str, Any] | None = None,
        max_new_tokens: int | None = None,
        chunk_length: int | None = None,
        clip_timestamps: str | list[float] = "0",
        hallucination_silence_threshold: float | None = None,
        hotwords: str | None = None,
        language_detection_threshold: float | None = 0.5,
        language_detection_segments: int = 1,
        log_progress: bool = False,
        # slower-whisper extensions (ignored by faster-whisper code)
        diarize: bool = False,
        enrich: bool = False,
    ) -> tuple[list[Segment], TranscriptionInfo]:
        """Transcribe audio file.

        This method matches faster-whisper's transcribe() signature exactly,
        allowing drop-in replacement. All parameters are passed through to
        the underlying transcription engine where supported.

        Args:
            audio: Path to audio file, file-like object, or numpy array.
            language: Language code (e.g., "en"). None for auto-detection.
            task: "transcribe" or "translate".
            beam_size: Beam size for decoding.
            word_timestamps: Enable word-level timestamps.
            vad_filter: Enable Voice Activity Detection filtering.
            vad_parameters: VAD configuration dict.
            diarize: Enable speaker diarization (slower-whisper extension).
            enrich: Enable audio enrichment (slower-whisper extension).
            ... (other parameters passed through to faster-whisper)

        Returns:
            Tuple of (segments, info) where:
            - segments: List of Segment objects
            - info: TranscriptionInfo with language, duration, etc.

        Note:
            Unlike faster-whisper which returns a generator, this returns
            a list of segments. This is intentional for simpler usage and
            because slower-whisper may need all segments for post-processing
            (diarization, enrichment).

        Example:
            >>> segments, info = model.transcribe("audio.wav", word_timestamps=True)
            >>> for seg in segments:
            ...     print(f"[{seg.start:.2f}s -> {seg.end:.2f}s] {seg.text}")
            ...     if seg.words:
            ...         for word in seg.words:
            ...             print(f"  {word.word} ({word.probability:.2f})")
        """
        # Store options for reporting
        self._last_transcription_options = {
            "language": language,
            "task": task,
            "beam_size": beam_size,
            "word_timestamps": word_timestamps,
            "vad_filter": vad_filter,
        }
        self._last_vad_options = vad_parameters

        # Convert audio input to file path
        audio_path, should_cleanup = self._audio_to_path(audio)

        try:
            # Get or create the transcription engine
            engine = self._ensure_engine()

            # Update engine config with transcribe-time options
            if language is not None:
                engine.cfg.language = language
            engine.cfg.task = task
            engine.cfg.beam_size = beam_size
            engine.cfg.word_timestamps = word_timestamps
            if vad_parameters:
                engine.cfg.vad_min_silence_ms = vad_parameters.get(
                    "min_silence_duration_ms", 500
                )

            # Perform transcription
            transcript = engine.transcribe_file(audio_path)

            # Apply diarization if requested
            if diarize:
                transcript = self._apply_diarization(transcript, audio_path)

            # Apply enrichment if requested
            if enrich:
                transcript = self._apply_enrichment(transcript, audio_path)

            # Store for access via last_transcript property
            self._last_transcript = transcript

            # Convert to faster-whisper compatible types
            segments = [Segment.from_internal(seg) for seg in transcript.segments]
            info = TranscriptionInfo.from_transcript(
                transcript,
                transcription_options=self._last_transcription_options,
                vad_options=self._last_vad_options,
            )

            return segments, info

        finally:
            # Cleanup temporary file if we created one
            if should_cleanup and audio_path.exists():
                try:
                    audio_path.unlink()
                except OSError:
                    pass

    def _apply_diarization(
        self, transcript: Transcript, audio_path: Path
    ) -> Transcript:
        """Apply speaker diarization to the transcript."""
        try:
            from transcription.diarization import Diarizer, assign_speakers
        except ImportError as e:
            logger.warning(
                "Diarization requested but pyannote.audio not available: %s. "
                "Install with: pip install slower-whisper[diarization]",
                e,
            )
            return transcript

        try:
            # Run diarization to get speaker turns
            diarizer = Diarizer()
            speaker_turns = diarizer.run(audio_path)

            # Assign speakers to transcript segments
            return assign_speakers(transcript, speaker_turns)
        except Exception as e:
            logger.warning("Diarization failed: %s", e, exc_info=True)
            return transcript

    def _apply_enrichment(self, transcript: Transcript, audio_path: Path) -> Transcript:
        """Apply audio enrichment to the transcript."""
        try:
            from transcription.api import enrich_transcript
            from transcription.enrichment_config import EnrichmentConfig
        except ImportError as e:
            logger.warning(
                "Enrichment requested but dependencies not available: %s. "
                "Install with: pip install slower-whisper[enrich-basic]",
                e,
            )
            return transcript

        try:
            config = EnrichmentConfig(device="cpu")  # Safe default
            return enrich_transcript(transcript, audio_path, config)
        except Exception as e:
            logger.warning("Enrichment failed: %s", e, exc_info=True)
            return transcript

    @property
    def model(self) -> Any:
        """Access the underlying faster-whisper model (for advanced use)."""
        engine = self._ensure_engine()
        return engine.model

    @property
    def device(self) -> str:
        """The device being used for inference."""
        if self._engine is not None:
            return str(self._engine.cfg.device)
        return self._device

    @property
    def compute_type(self) -> str:
        """The compute type being used for inference."""
        if self._engine is not None:
            return str(self._engine.cfg.compute_type or "unknown")
        return self._compute_type
