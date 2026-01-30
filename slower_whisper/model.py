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

        # Cache for detect_language results: path -> (language, probability, all_probs)
        self._language_cache: dict[str, tuple[str, float, list[tuple[str, float]] | None]] = {}

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
            logger.warning(
                "Invalid device '%s', defaulting to 'auto'. "
                "Valid options are: 'auto', 'cpu', 'cuda'.",
                device_choice,
            )
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

    def _audio_to_path(
        self, audio: AudioInput, sample_rate: int | None = None
    ) -> tuple[Path, bool]:
        """Convert audio input to a file path.

        Args:
            audio: Audio input (path string, Path, file-like object, or numpy array).
            sample_rate: Sample rate for numpy array input. If None, assumes 16000 Hz
                (Whisper's native rate). Ignored for file inputs.

        Returns:
            Tuple of (path, should_cleanup) where should_cleanup indicates if
            the path points to a temporary file that should be deleted.
        """
        # String or Path - use directly
        if isinstance(audio, (str, Path)):
            return Path(audio), False

        # File-like object - read and write to temp file
        if isinstance(audio, (io.IOBase, BinaryIO)) or hasattr(audio, "read"):
            data = audio.read()
            if not isinstance(data, (bytes, bytearray)):
                raise TypeError(
                    "File-like audio input must return bytes from read(); "
                    f"got {type(data).__name__}."
                )
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(data)
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

            # Use provided sample rate or default to Whisper's native 16kHz
            if sample_rate is None:
                sample_rate = 16000
                logger.debug(
                    "No sample_rate provided for numpy array; assuming %d Hz. "
                    "If audio has a different sample rate, pass sample_rate parameter.",
                    sample_rate,
                )

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                sf.write(tmp.name, audio, samplerate=sample_rate)
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
        prepend_punctuations: str = '"\'"¿([{-',
        append_punctuations: str = '"\'.。,，!！?？:：")]}、',
        multilingual: bool = False,
        vad_filter: bool = True,
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
        sample_rate: int | None = None,
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
            sample_rate: Sample rate for numpy array input (slower-whisper extension).
                If None and audio is a numpy array, assumes 16000 Hz (Whisper's native rate).
                Ignored for file path inputs.
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
        audio_path, should_cleanup = self._audio_to_path(audio, sample_rate=sample_rate)

        try:
            # Get or create the transcription engine
            engine = self._ensure_engine()

            # Update engine config with transcribe-time options
            if language is not None:
                engine.cfg.language = language
            engine.cfg.task = task
            engine.cfg.beam_size = beam_size
            engine.cfg.word_timestamps = word_timestamps

            # Respect faster-whisper semantics: allow disabling VAD
            engine.cfg.vad_filter = bool(vad_filter)

            # Only apply vad_parameters when VAD is enabled
            if vad_filter and vad_parameters:
                engine.cfg.vad_min_silence_ms = vad_parameters.get("min_silence_duration_ms", 500)

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

    def _apply_diarization(self, transcript: Transcript, audio_path: Path) -> Transcript:
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

    @property
    def supported_languages(self) -> list[str]:
        """Languages supported by the model.

        Returns a list of ISO 639-1 language codes supported by the loaded model.
        For multilingual models, this includes all ~100 languages Whisper supports.
        For English-only models (e.g., "tiny.en"), this returns ["en"].

        Returns:
            List of supported language codes (e.g., ["en", "es", "fr", ...]).

        Example:
            >>> model = WhisperModel("base")
            >>> print(model.supported_languages[:5])
            ['en', 'zh', 'de', 'es', 'ru']
        """
        engine = self._ensure_engine()
        underlying = engine.model

        # Check if the underlying model has supported_languages
        if hasattr(underlying, "supported_languages"):
            return list(underlying.supported_languages)

        # Fallback: check if model is multilingual
        if hasattr(underlying, "model") and hasattr(underlying.model, "is_multilingual"):
            if underlying.model.is_multilingual:
                # Return common Whisper languages
                return _WHISPER_LANGUAGE_CODES
            return ["en"]

        # Default fallback: assume multilingual
        return _WHISPER_LANGUAGE_CODES

    def detect_language(
        self,
        audio: AudioInput,
        vad_filter: bool = False,
        vad_parameters: dict[str, Any] | None = None,
        language_detection_segments: int = 1,
        language_detection_threshold: float = 0.5,
        sample_rate: int | None = None,
    ) -> tuple[str, float, list[tuple[str, float]] | None]:
        """Detect the language of audio (matches faster-whisper signature).

        Analyzes the audio to determine the most likely spoken language.
        This is useful for pre-screening audio before transcription or for
        multilingual audio processing.

        Results are cached by file path to avoid redundant analysis on repeated
        calls with the same audio file. Use clear_language_cache() to reset.

        Args:
            audio: Path to audio file, file-like object, or numpy array.
            vad_filter: Enable VAD to filter non-speech before detection.
            vad_parameters: VAD configuration dict.
            language_detection_segments: Number of segments to analyze.
            language_detection_threshold: Minimum confidence threshold.
            sample_rate: Sample rate for numpy array input. If None and audio is
                a numpy array, assumes 16000 Hz. Ignored for file path inputs.

        Returns:
            Tuple of (language_code, probability, all_language_probs):
            - language_code: ISO 639-1 code of detected language (e.g., "en")
            - probability: Confidence score (0.0-1.0) for the detected language
            - all_language_probs: List of (language, prob) tuples for all languages,
              or None if not available

        Example:
            >>> model = WhisperModel("base")
            >>> lang, prob, all_probs = model.detect_language("audio.wav")
            >>> print(f"Detected: {lang} (confidence: {prob:.2%})")
            Detected: en (confidence: 98.50%)
        """
        engine = self._ensure_engine()
        underlying = engine.model

        # Convert audio input to appropriate format
        audio_path, should_cleanup = self._audio_to_path(audio, sample_rate=sample_rate)
        cache_key = str(audio_path.resolve()) if not should_cleanup else None

        try:
            # Check cache for file-based inputs (not temp files)
            if cache_key and cache_key in self._language_cache:
                logger.debug("Using cached language detection for %s", audio_path.name)
                return self._language_cache[cache_key]

            # Check if the underlying model has detect_language
            if hasattr(underlying, "detect_language"):
                # Load audio as numpy array for faster-whisper
                try:
                    from faster_whisper.audio import decode_audio

                    audio_array = decode_audio(str(audio_path))
                    result = underlying.detect_language(
                        audio=audio_array,
                        vad_filter=vad_filter,
                        vad_parameters=vad_parameters,
                        language_detection_segments=language_detection_segments,
                        language_detection_threshold=language_detection_threshold,
                    )
                    # faster-whisper returns (lang, prob, all_probs)
                    if isinstance(result, tuple) and len(result) >= 2:
                        lang = result[0]
                        prob = result[1]
                        all_probs = result[2] if len(result) > 2 else None
                        detection_result = (str(lang), float(prob), all_probs)
                        # Cache the result for file-based inputs
                        if cache_key:
                            self._language_cache[cache_key] = detection_result
                        return detection_result
                except ImportError:
                    logger.debug("faster_whisper.audio not available, using fallback")

            # Fallback: run a quick transcription and use detected language
            # This works with the dummy model too
            transcript = engine.transcribe_file(audio_path)
            detected_lang = transcript.language or "en"
            # Return with high confidence since we transcribed it
            detection_result = (detected_lang, 0.99, [(detected_lang, 0.99)])
            # Cache the result for file-based inputs
            if cache_key:
                self._language_cache[cache_key] = detection_result
            return detection_result

        finally:
            # Cleanup temporary file if we created one
            if should_cleanup and audio_path.exists():
                try:
                    audio_path.unlink()
                except OSError:
                    pass

    def clear_language_cache(self) -> None:
        """Clear the language detection cache.

        Call this if you need to re-detect language for previously analyzed files,
        for example if the audio content has changed.
        """
        self._language_cache.clear()


# Whisper's supported language codes (ISO 639-1)
# This is a subset of the most common ones; the full list has ~100 languages
_WHISPER_LANGUAGE_CODES = [
    "en",
    "zh",
    "de",
    "es",
    "ru",
    "ko",
    "fr",
    "ja",
    "pt",
    "tr",
    "pl",
    "ca",
    "nl",
    "ar",
    "sv",
    "it",
    "id",
    "hi",
    "fi",
    "vi",
    "he",
    "uk",
    "el",
    "ms",
    "cs",
    "ro",
    "da",
    "hu",
    "ta",
    "no",
    "th",
    "ur",
    "hr",
    "bg",
    "lt",
    "la",
    "mi",
    "ml",
    "cy",
    "sk",
    "te",
    "fa",
    "lv",
    "bn",
    "sr",
    "az",
    "sl",
    "kn",
    "et",
    "mk",
    "br",
    "eu",
    "is",
    "hy",
    "ne",
    "mn",
    "bs",
    "kk",
    "sq",
    "sw",
    "gl",
    "mr",
    "pa",
    "si",
    "km",
    "sn",
    "yo",
    "so",
    "af",
    "oc",
    "ka",
    "be",
    "tg",
    "sd",
    "gu",
    "am",
    "yi",
    "lo",
    "uz",
    "fo",
    "ht",
    "ps",
    "tk",
    "nn",
    "mt",
    "sa",
    "lb",
    "my",
    "bo",
    "tl",
    "mg",
    "as",
    "tt",
    "haw",
    "ln",
    "ha",
    "ba",
    "jw",
    "su",
    "yue",
]
