"""ASR engine wrapper for faster-whisper transcription.

This module provides the TranscriptionEngine class which wraps faster-whisper
for speech-to-text conversion. It handles model loading, device selection
(CPU/GPU), and batch transcription with progress tracking.

The module uses a Protocol pattern to gracefully handle optional dependencies,
falling back to no-op implementations when faster-whisper is unavailable.
"""

# cSpell: ignore samplerate
import inspect
import logging
import math
from collections.abc import Iterable, Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Protocol, cast

from .cache import CachePaths
from .config import AsrConfig
from .models import Segment, Transcript, Word

logger = logging.getLogger(__name__)

TranscriptionResult = tuple[Iterable[Any], Any]
VADParameters = Mapping[str, Any] | None


class WhisperModelProtocol(Protocol):
    def transcribe(
        self,
        audio_path: str,
        beam_size: int | None = ...,
        vad_filter: bool = ...,
        vad_parameters: VADParameters = ...,
        language: str | None = ...,
        task: str | None = ...,
    ) -> TranscriptionResult: ...


WhisperModel: type[Any] | None
try:
    from faster_whisper import WhisperModel as _WhisperModel

    WhisperModel = _WhisperModel
    _FASTER_WHISPER_AVAILABLE = True
except Exception:
    WhisperModel = None
    _FASTER_WHISPER_AVAILABLE = False
# Backward-compatible alias for older references.
_faster_whisper_available = _FASTER_WHISPER_AVAILABLE


class DummyWhisperModel:
    """Lightweight fallback model used when faster-whisper is unavailable."""

    def __init__(self, cfg: AsrConfig):
        self.cfg = cfg

    def transcribe(
        self,
        audio_path: str,
        beam_size: int | None = None,
        vad_filter: bool = True,
        vad_parameters: VADParameters = None,
        language: str | None = None,
        task: str | None = None,
    ) -> TranscriptionResult:
        try:
            import soundfile as sf

            with sf.SoundFile(audio_path, "r") as f:
                duration = len(f) / f.samplerate if f.samplerate else 1.0
        except Exception:
            # If we cannot read the file (mocked or missing deps), fall back to 1s
            duration = 1.0

        # Provide a single dummy segment covering the clip
        segment = SimpleNamespace(start=0.0, end=max(duration, 0.5), text="dummy segment")
        info = SimpleNamespace(language=language or self.cfg.language or "unknown")
        return [segment], info


class TranscriptionEngine:
    """
    Thin wrapper around faster-whisper that returns Transcript objects.
    """

    def __init__(self, cfg: AsrConfig) -> None:
        self.cfg = cfg
        self.model_load_error: str | None = None
        self.model_load_warnings: list[str] = []
        logger.info(
            "=== Step 2: Loading Whisper model %s on %s (%s) ===",
            cfg.model_name,
            cfg.device,
            cfg.compute_type,
            extra={"device": cfg.device, "compute_type": cfg.compute_type},
        )
        self.model: WhisperModelProtocol = self._init_model()
        self._supports_vad_filter = True
        self._supports_vad_parameters = True
        self._warned_vad_unsupported = False
        self._supports_vad_filter, self._supports_vad_parameters = self._detect_vad_support()
        # Track whether we're using the dummy fallback so downstream code can adapt.
        self.using_dummy = isinstance(self.model, DummyWhisperModel)

    def _load_whisper_model(
        self, device: str, compute_type: str, download_root: Path
    ) -> WhisperModelProtocol:
        """Instantiate WhisperModel with shared parameters."""
        if WhisperModel is None:
            raise RuntimeError("WhisperModel is not available")
        model = WhisperModel(
            self.cfg.model_name,
            device=device,
            compute_type=compute_type,
            download_root=str(download_root),
        )
        return cast(WhisperModelProtocol, model)

    def _init_model(self) -> WhisperModelProtocol:
        self.model_load_error = None
        self.model_load_warnings = []
        requested_device = self.cfg.device
        requested_compute_type = self.cfg.compute_type or "int8"
        try:
            if not _FASTER_WHISPER_AVAILABLE or WhisperModel is None:
                raise ImportError("faster-whisper not available; using dummy model")

            # Use centralized cache for Whisper model downloads
            paths = CachePaths.from_env().ensure_dirs()
            attempts = [
                (requested_device, requested_compute_type),
            ]
            cpu_fallback = ("cpu", "int8")

            # If the requested configuration is likely to fail (GPU unavailable or CPU + float16),
            # enqueue a safer CPU/int8 retry before resorting to the dummy model.
            if requested_device != "cpu":
                attempts.append(cpu_fallback)
            elif requested_compute_type != "int8":
                attempts.append(cpu_fallback)

            # Deduplicate while preserving order
            seen: set[tuple[str, str]] = set()
            ordered_attempts: list[tuple[str, str]] = []
            for attempt in attempts:
                if attempt in seen:
                    continue
                seen.add(attempt)
                ordered_attempts.append(attempt)

            failures: list[str] = []
            for idx, (device, compute_type) in enumerate(ordered_attempts):
                try:
                    model = self._load_whisper_model(
                        device=device,
                        compute_type=compute_type,
                        download_root=paths.whisper_root,
                    )
                except Exception as load_err:
                    failures.append(f"{device} ({compute_type}) load failed: {load_err}")

                    # If there's another attempt queued, warn and retry.
                    if idx + 1 < len(ordered_attempts):
                        next_device, next_compute_type = ordered_attempts[idx + 1]
                        next_device_label = (
                            next_device.upper() if next_device.lower() == "cpu" else next_device
                        )
                        if device != next_device:
                            logger.warning(
                                "Whisper model load failed on %s: %s",
                                device,
                                load_err,
                                extra={"device": device},
                                exc_info=True,
                            )
                            logger.warning(
                                "Retrying on %s with compute_type=%s",
                                next_device_label,
                                next_compute_type,
                                extra={"device": next_device, "compute_type": next_compute_type},
                            )
                        else:
                            logger.warning(
                                "Whisper model load failed with compute_type=%s: %s",
                                compute_type,
                                load_err,
                                extra={"compute_type": compute_type},
                                exc_info=True,
                            )
                            logger.warning(
                                "Retrying on %s with compute_type=%s",
                                next_device_label,
                                next_compute_type,
                                extra={"device": next_device, "compute_type": next_compute_type},
                            )
                    continue

                # Success: update config to reflect the actual model in use if it changed.
                if failures:
                    self.model_load_warnings.extend(failures)
                if (device, compute_type) != (requested_device, requested_compute_type):
                    self.cfg.device = device
                    self.cfg.compute_type = compute_type
                return model

            aggregated = "; ".join(failures)
            self.model_load_warnings.extend(failures)
            raise RuntimeError(aggregated)
        except Exception as e:
            # Fall back to a lightweight dummy model so tests can run without heavy deps
            logger.warning("Using dummy Whisper model fallback: %s", e, exc_info=True)
            self.model_load_error = str(e)
            return DummyWhisperModel(self.cfg)

    def _detect_vad_support(self) -> tuple[bool, bool]:
        """Detect whether model.transcribe accepts VAD kwargs to avoid legacy breakage."""
        transcribe_fn = getattr(self.model, "transcribe", None)
        if transcribe_fn is None:
            return True, True

        try:
            signature = inspect.signature(transcribe_fn)
        except (TypeError, ValueError):
            # Builtins/extension functions may not expose a signature; assume support.
            return True, True

        has_kwargs = any(
            param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()
        )
        if has_kwargs:
            return True, True

        return (
            "vad_filter" in signature.parameters,
            "vad_parameters" in signature.parameters,
        )

    def _materialize_segments(self, segments: Iterable[Any]) -> list[Any]:
        """Eagerly consume a generator so lazy errors surface here."""
        try:
            return list(segments)
        except Exception as exc:  # noqa: BLE001 - want the raw error for fallback context
            raise RuntimeError(f"Failed while streaming Whisper segments: {exc}") from exc

    def _warn_vad_disabled(self) -> None:
        """Warn once if we have to drop VAD arguments for legacy faster-whisper versions."""
        if self._warned_vad_unsupported:
            return
        if not self._supports_vad_filter and not self._supports_vad_parameters:
            detail = "VAD parameters; running without VAD filtering."
        elif not self._supports_vad_parameters:
            detail = "vad_parameters; running with vad_filter only."
        else:
            detail = "vad_filter; running without VAD filtering."
        logger.warning("faster-whisper transcribe() does not support %s", detail)
        self._warned_vad_unsupported = True

    def _parse_vad_kwargs_error(self, exc: TypeError) -> tuple[bool, bool]:
        """Return which VAD kwargs triggered a TypeError (vad_filter, vad_parameters)."""
        message = str(exc).lower()
        return ("vad_filter" in message), ("vad_parameters" in message)

    def _build_transcribe_kwargs(self, include_vad: bool = True) -> dict[str, Any]:
        """Build keyword arguments for model.transcribe with optional VAD filtering."""
        kwargs: dict[str, Any] = {
            "beam_size": self.cfg.beam_size,
            "language": self.cfg.language,
            "task": self.cfg.task,
        }
        if include_vad and self._supports_vad_filter:
            kwargs["vad_filter"] = True
        if include_vad and self._supports_vad_parameters:
            kwargs["vad_parameters"] = {"min_silence_duration_ms": self.cfg.vad_min_silence_ms}
        # Word-level timestamps (v1.8+)
        if getattr(self.cfg, "word_timestamps", False):
            kwargs["word_timestamps"] = True
        return kwargs

    def _transcribe_with_model(self, audio_path: Path) -> TranscriptionResult:
        """
        Call the underlying model.transcribe, retrying without VAD kwargs on legacy versions.
        """
        if not (self._supports_vad_filter and self._supports_vad_parameters):
            self._warn_vad_disabled()

        include_vad = True
        stripped_all_vad = False

        while True:
            kwargs = self._build_transcribe_kwargs(include_vad=include_vad)
            try:
                return self.model.transcribe(str(audio_path), **kwargs)
            except TypeError as exc:
                unsupported_filter, unsupported_params = self._parse_vad_kwargs_error(exc)
                if not (unsupported_filter or unsupported_params):
                    raise

                # Legacy faster-whisper builds before vad_filter/vad_parameters support.
                if unsupported_filter:
                    self._supports_vad_filter = False
                if unsupported_params:
                    self._supports_vad_parameters = False
                self._warn_vad_disabled()

                if stripped_all_vad:
                    # Already tried without any VAD kwargs; surface the error.
                    raise

                # If both VAD kwargs are rejected, drop VAD entirely and retry once more.
                if not (self._supports_vad_filter or self._supports_vad_parameters):
                    include_vad = False
                    stripped_all_vad = True
                    continue

                # Otherwise, loop once more with the supported subset of VAD kwargs.
                continue

    def _normalize_language(self, info: Any) -> str:
        """Ensure we always return a language string, even if the model omits it."""
        if isinstance(info, str):
            language_value: Any | None = info
        else:
            language_value = getattr(info, "language", None)
            if language_value is None and isinstance(info, Mapping):
                info_mapping = cast(Mapping[str, Any], info)
                language_value = info_mapping.get("language")

        language: str | None = language_value.strip() if isinstance(language_value, str) else None

        if not language:
            cfg_language = self.cfg.language
            language = cfg_language.strip() if isinstance(cfg_language, str) else None

        return language or "unknown"

    def _build_segments(self, raw_segments: Iterable[Any]) -> list[Segment]:
        """Convert raw Whisper segments into our Segment dataclass."""
        # Collect validated segments with optional word-level data and faster-whisper fields
        # Tuple: (start, end, text, words, tokens, avg_logprob, compression_ratio,
        #         no_speech_prob, temperature, seek)
        validated: list[
            tuple[
                float,
                float,
                str,
                list[Word] | None,
                list[int] | None,
                float,
                float,
                float,
                float,
                int,
            ]
        ] = []
        extract_words = getattr(self.cfg, "word_timestamps", False)

        for idx, seg in enumerate(raw_segments):
            try:
                start = float(seg.start)
                end = float(seg.end)
                text = getattr(seg, "text", "")
            except Exception as exc:  # noqa: BLE001 - guard against arbitrary model objects
                raise ValueError(f"Invalid segment payload at index {idx}") from exc

            if not (math.isfinite(start) and math.isfinite(end)):
                raise ValueError(
                    f"Invalid segment timing at index {idx}: start={start}, end={end} (non-finite)"
                )
            if start < 0 or end < 0:
                raise ValueError(
                    f"Invalid segment timing at index {idx}: start={start}, end={end} (negative)"
                )
            if end < start:
                raise ValueError(
                    f"Invalid segment timing at index {idx}: start={start}, end={end} (end < start)"
                )

            text_str = "" if text is None else str(text)

            # Extract word-level timestamps if available
            words: list[Word] | None = None
            if extract_words:
                raw_words = getattr(seg, "words", None)
                if raw_words:
                    words = self._build_words(raw_words, idx)

            # Extract faster-whisper compatibility fields
            raw_tokens = getattr(seg, "tokens", None)
            tokens: list[int] | None = list(raw_tokens) if raw_tokens is not None else None
            avg_logprob = float(getattr(seg, "avg_logprob", 0.0))
            compression_ratio = float(getattr(seg, "compression_ratio", 1.0))
            no_speech_prob = float(getattr(seg, "no_speech_prob", 0.0))
            temperature = float(getattr(seg, "temperature", 0.0))
            seek = int(getattr(seg, "seek", 0))

            validated.append(
                (
                    start,
                    end,
                    text_str.strip(),
                    words,
                    tokens,
                    avg_logprob,
                    compression_ratio,
                    no_speech_prob,
                    temperature,
                    seek,
                )
            )

        is_monotonic = all(
            validated[i][0] <= validated[i + 1][0] for i in range(len(validated) - 1)
        )
        if not is_monotonic:
            # Keep segments in chronological order if the backend emitted them out of order.
            validated = sorted(validated, key=lambda seg: (seg[0], seg[1]))

        seg_objs = [
            Segment(
                id=idx,
                start=start,
                end=end,
                text=text,
                words=words,
                tokens=tokens,
                avg_logprob=avg_logprob,
                compression_ratio=compression_ratio,
                no_speech_prob=no_speech_prob,
                temperature=temperature,
                seek=seek,
            )
            for idx, (
                start,
                end,
                text,
                words,
                tokens,
                avg_logprob,
                compression_ratio,
                no_speech_prob,
                temperature,
                seek,
            ) in enumerate(validated)
        ]
        return seg_objs

    def _build_words(self, raw_words: Iterable[Any], segment_idx: int) -> list[Word]:
        """Convert raw faster-whisper words into our Word dataclass."""
        words: list[Word] = []
        for word_idx, w in enumerate(raw_words):
            try:
                word_text = getattr(w, "word", "")
                word_start = float(getattr(w, "start", 0.0))
                word_end = float(getattr(w, "end", 0.0))
                word_prob = float(getattr(w, "probability", 1.0))
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Invalid word at segment %d, word %d: %s",
                    segment_idx,
                    word_idx,
                    exc,
                )
                continue

            # Validate word timing
            if not (math.isfinite(word_start) and math.isfinite(word_end)):
                logger.warning(
                    "Skipping word with non-finite timing at segment %d, word %d",
                    segment_idx,
                    word_idx,
                )
                continue

            # Clamp probability to valid range
            word_prob = max(0.0, min(1.0, word_prob))

            words.append(
                Word(
                    word=str(word_text) if word_text else "",
                    start=word_start,
                    end=word_end,
                    probability=word_prob,
                )
            )
        return words

    def transcribe_file(self, audio_path: Path) -> Transcript:
        """
        Transcribe a single audio file into a Transcript.
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        if not audio_path.is_file():
            raise IsADirectoryError(f"Audio path is not a file: {audio_path}")

        logger.info("Transcribing file: %s", audio_path.name, extra={"file": audio_path.name})
        used_dummy = isinstance(self.model, DummyWhisperModel)
        info: Any | None = None
        fallback_reason: str | None = None

        if not hasattr(self.model, "transcribe"):
            # Extremely defensive: model object is missing transcribe()
            segments, info = DummyWhisperModel(self.cfg).transcribe(str(audio_path))
            used_dummy = True
            fallback_reason = self.model_load_error or "Whisper model missing transcribe()"
        else:
            try:
                raw_segments, info = self._transcribe_with_model(audio_path)
                segments = self._materialize_segments(raw_segments)
            except Exception as exc:
                # Graceful degradation if the real model fails to run (including lazy generator failures)
                logger.warning(
                    "Whisper inference failed for %s: %s. Falling back to dummy output.",
                    audio_path.name,
                    exc,
                    extra={"file": audio_path.name},
                    exc_info=True,
                )
                segments, fallback_info = DummyWhisperModel(self.cfg).transcribe(str(audio_path))
                info = info or fallback_info
                used_dummy = True
                fallback_reason = str(exc)

        placeholder_segments = False

        try:
            seg_objs = self._build_segments(segments)
        except Exception as exc:
            if used_dummy:
                # Building segments for dummy output should never fail; bubble up
                raise
            logger.warning(
                "Whisper produced invalid segments for %s: %s. Falling back to dummy output.",
                audio_path.name,
                exc,
                extra={"file": audio_path.name},
                exc_info=True,
            )
            segments, fallback_info = DummyWhisperModel(self.cfg).transcribe(str(audio_path))
            info = info or fallback_info
            seg_objs = self._build_segments(segments)
            used_dummy = True
            fallback_reason = fallback_reason or str(exc)

        if not seg_objs:
            # Handle edge cases like silent audio where faster-whisper returns no segments.
            logger.warning(
                "Whisper produced no segments for %s; falling back to dummy output.",
                audio_path.name,
                extra={"file": audio_path.name},
            )
            segments, fallback_info = DummyWhisperModel(self.cfg).transcribe(str(audio_path))
            info = info or fallback_info
            seg_objs = self._build_segments(segments)
            if used_dummy:
                fallback_reason = fallback_reason or "no segments produced"
            else:
                # Keep backend metadata as the real model but surface the placeholder.
                placeholder_segments = True
                fallback_reason = fallback_reason or "no segments produced"

        self.using_dummy = used_dummy

        # Extract duration_after_vad from faster-whisper's TranscriptionInfo
        duration_after_vad: float | None = None
        if info is not None and hasattr(info, "duration_after_vad"):
            try:
                duration_after_vad = float(info.duration_after_vad)
            except (TypeError, ValueError):
                pass

        transcript = Transcript(
            file_name=audio_path.name,
            language=self._normalize_language(info),
            segments=seg_objs,
            duration_after_vad=duration_after_vad,
        )
        actual_device = "cpu" if used_dummy else self.cfg.device
        actual_compute_type = "n/a" if used_dummy else (self.cfg.compute_type or "unknown")
        asr_meta: dict[str, str | list[str] | bool] = {
            "asr_backend": "dummy" if used_dummy else "faster-whisper",
            "asr_device": actual_device,
            "asr_compute_type": actual_compute_type,
        }
        if self.model_load_warnings:
            # Surface model load retries (e.g., GPU→CPU) for downstream diagnostics.
            asr_meta["asr_model_load_warnings"] = list(self.model_load_warnings)
        reason = fallback_reason or self.model_load_error
        if reason:
            asr_meta["asr_fallback_reason"] = reason
            if placeholder_segments and not used_dummy:
                asr_meta["asr_placeholder_segments"] = True

        transcript.meta = {**(transcript.meta or {}), **asr_meta}

        return transcript

    def transcribe_many(self, audio_files: Iterable[Path]) -> Iterable[Transcript]:
        """
        Convenience generator to transcribe multiple files.
        """
        for path in audio_files:
            yield self.transcribe_file(path)
