"""Fail-closed ASR engine wrapper for faster-whisper transcription.

The engine may retry a real backend on a safer device/compute configuration,
but it never manufactures transcript text. Backend availability, model loading,
inference, malformed output, and genuine silence remain distinct outcomes.
"""

# cSpell: ignore samplerate
from __future__ import annotations

import inspect
import logging
import math
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any, Protocol, cast

from .cache import CachePaths
from .config import AsrConfig
from .exceptions import (
    ASRInferenceError,
    ASRModelLoadError,
    ASROutputError,
    ASRUnavailableError,
)
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


BackendFactory = Callable[[str, str, str, Path], WhisperModelProtocol]


WhisperModel: type[Any] | None
_FASTER_WHISPER_IMPORT_ERROR: Exception | None = None
try:
    from faster_whisper import WhisperModel as _WhisperModel

    WhisperModel = _WhisperModel
    _FASTER_WHISPER_AVAILABLE = True
except Exception as exc:  # pragma: no cover - environment-dependent import
    WhisperModel = None
    _FASTER_WHISPER_AVAILABLE = False
    _FASTER_WHISPER_IMPORT_ERROR = exc

# Backward-compatible alias for older references.
_faster_whisper_available = _FASTER_WHISPER_AVAILABLE


class TranscriptionEngine:
    """Thin, fail-closed wrapper around faster-whisper."""

    def __init__(
        self,
        cfg: AsrConfig,
        *,
        backend_factory: BackendFactory | None = None,
    ) -> None:
        self.cfg = cfg
        self._backend_factory = backend_factory
        self.model_load_attempts: list[dict[str, str]] = []
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

    def _load_whisper_model(
        self,
        device: str,
        compute_type: str,
        download_root: Path,
    ) -> WhisperModelProtocol:
        """Instantiate the configured backend with shared parameters."""
        if self._backend_factory is not None:
            return self._backend_factory(
                self.cfg.model_name,
                device,
                compute_type,
                download_root,
            )
        if WhisperModel is None:
            raise ASRUnavailableError(
                "The faster-whisper backend is unavailable",
                context={"backend": "faster-whisper"},
            )
        model = WhisperModel(
            self.cfg.model_name,
            device=device,
            compute_type=compute_type,
            download_root=str(download_root),
        )
        return cast(WhisperModelProtocol, model)

    def _ordered_load_configs(self) -> list[tuple[str, str]]:
        requested = (self.cfg.device, self.cfg.compute_type or "int8")
        attempts = [requested]
        cpu_fallback = ("cpu", "int8")
        if requested[0] != "cpu" or requested[1] != "int8":
            attempts.append(cpu_fallback)

        seen: set[tuple[str, str]] = set()
        ordered: list[tuple[str, str]] = []
        for attempt in attempts:
            if attempt not in seen:
                seen.add(attempt)
                ordered.append(attempt)
        return ordered

    def _init_model(self) -> WhisperModelProtocol:
        """Load a real model, preserving an ordered receipt for every attempt."""
        self.model_load_attempts = []
        self.model_load_warnings = []

        if self._backend_factory is None and (
            not _FASTER_WHISPER_AVAILABLE or WhisperModel is None
        ):
            unavailable_error = ASRUnavailableError(
                "The faster-whisper backend is unavailable",
                context={"backend": "faster-whisper"},
            )
            if _FASTER_WHISPER_IMPORT_ERROR is not None:
                raise unavailable_error from _FASTER_WHISPER_IMPORT_ERROR
            raise unavailable_error

        try:
            paths = CachePaths.from_env().ensure_dirs()
        except Exception as exc:  # noqa: BLE001 - preserve provider/cache cause locally
            raise ASRModelLoadError(
                "The ASR model cache could not be prepared",
                context={
                    "backend": "faster-whisper",
                    "model": self.cfg.model_name,
                },
            ) from exc

        requested = (self.cfg.device, self.cfg.compute_type or "int8")
        ordered_attempts = self._ordered_load_configs()
        last_error: Exception | None = None

        for index, (device, compute_type) in enumerate(ordered_attempts):
            try:
                model = self._load_whisper_model(
                    device=device,
                    compute_type=compute_type,
                    download_root=paths.whisper_root,
                )
            except ASRUnavailableError:
                raise
            except Exception as exc:  # noqa: BLE001 - backend exceptions are chained
                last_error = exc
                self.model_load_attempts.append(
                    {
                        "device": device,
                        "compute_type": compute_type,
                        "outcome": "failed",
                        "reason_code": ASRModelLoadError.default_reason_code,
                    }
                )
                self.model_load_warnings.append(f"{device} ({compute_type}) load failed")
                logger.warning(
                    "Whisper model load failed on %s with compute_type=%s",
                    device,
                    compute_type,
                    extra={"device": device, "compute_type": compute_type},
                    exc_info=True,
                )

                if index + 1 < len(ordered_attempts):
                    next_device, next_compute_type = ordered_attempts[index + 1]
                    next_device_label = (
                        next_device.upper() if next_device.lower() == "cpu" else next_device
                    )
                    logger.warning(
                        "Retrying on %s with compute_type=%s",
                        next_device_label,
                        next_compute_type,
                        extra={
                            "device": next_device,
                            "compute_type": next_compute_type,
                        },
                    )
                continue

            self.model_load_attempts.append(
                {
                    "device": device,
                    "compute_type": compute_type,
                    "outcome": "selected",
                    "reason_code": "ok",
                }
            )
            if (device, compute_type) != requested:
                self.cfg.device = device
                self.cfg.compute_type = compute_type
            return model

        load_error = ASRModelLoadError(
            "The ASR model could not be loaded with any supported runtime configuration",
            context={
                "backend": "faster-whisper",
                "model": self.cfg.model_name,
                "attempts": [dict(attempt) for attempt in self.model_load_attempts],
            },
        )
        if last_error is not None:
            raise load_error from last_error
        raise load_error

    def _detect_vad_support(self) -> tuple[bool, bool]:
        """Detect whether model.transcribe accepts VAD kwargs."""
        transcribe_fn = getattr(self.model, "transcribe", None)
        if transcribe_fn is None:
            return True, True

        try:
            signature = inspect.signature(transcribe_fn)
        except (TypeError, ValueError):
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
        """Consume lazy segments while distinguishing bad shape from lazy failure."""
        if isinstance(segments, (str, bytes, bytearray, Mapping)):
            raise ASROutputError(
                "The ASR backend returned an invalid segment collection",
                context={"violation": "segment_collection_type"},
            )

        try:
            iterator = iter(segments)
        except TypeError as exc:
            raise ASROutputError(
                "The ASR backend returned a non-iterable segment collection",
                context={"violation": "segments_not_iterable"},
            ) from exc

        try:
            return list(iterator)
        except Exception as exc:  # noqa: BLE001 - lazy provider failure is inference
            raise ASRInferenceError(
                "ASR inference failed while consuming segment output",
                context={"phase": "segment_iteration"},
            ) from exc

    def _warn_vad_disabled(self) -> None:
        """Warn once if legacy backend signatures require dropping VAD arguments."""
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
        """Return which VAD kwargs triggered a TypeError."""
        message = str(exc).lower()
        return ("vad_filter" in message), ("vad_parameters" in message)

    def _build_transcribe_kwargs(self, include_vad: bool = True) -> dict[str, Any]:
        """Build keyword arguments for model.transcribe."""
        kwargs: dict[str, Any] = {
            "beam_size": self.cfg.beam_size,
            "language": self.cfg.language,
            "task": self.cfg.task,
        }

        vad_enabled = bool(getattr(self.cfg, "vad_filter", True))
        include_vad = include_vad and vad_enabled
        if include_vad and self._supports_vad_filter:
            kwargs["vad_filter"] = True
        if include_vad and self._supports_vad_parameters:
            kwargs["vad_parameters"] = {"min_silence_duration_ms": self.cfg.vad_min_silence_ms}
        if getattr(self.cfg, "word_timestamps", False):
            kwargs["word_timestamps"] = True
        return kwargs

    def _transcribe_with_model(self, audio_path: Path) -> Any:
        """Call model.transcribe, retrying only while VAD kwargs make progress."""
        transcribe_fn = getattr(self.model, "transcribe", None)
        if not callable(transcribe_fn):
            raise ASRInferenceError(
                "The initialized ASR backend has no callable transcribe method",
                context={"phase": "dispatch"},
            )

        if not (self._supports_vad_filter and self._supports_vad_parameters):
            self._warn_vad_disabled()

        include_vad = True

        while True:
            kwargs = self._build_transcribe_kwargs(include_vad=include_vad)
            try:
                return transcribe_fn(str(audio_path), **kwargs)
            except TypeError as exc:
                unsupported_filter, unsupported_params = self._parse_vad_kwargs_error(exc)
                if not (unsupported_filter or unsupported_params):
                    raise

                previous_state = (
                    include_vad,
                    self._supports_vad_filter,
                    self._supports_vad_parameters,
                )
                if unsupported_filter:
                    self._supports_vad_filter = False
                if unsupported_params:
                    self._supports_vad_parameters = False
                if not (self._supports_vad_filter or self._supports_vad_parameters):
                    include_vad = False
                self._warn_vad_disabled()

                current_state = (
                    include_vad,
                    self._supports_vad_filter,
                    self._supports_vad_parameters,
                )
                if current_state == previous_state:
                    raise ASRInferenceError(
                        "ASR inference failed while negotiating VAD arguments",
                        context={"phase": "vad_negotiation"},
                    ) from exc

    def _normalize_language(self, info: Any) -> str:
        """Return a language string even when backend info omits one."""
        if isinstance(info, str):
            language_value: Any | None = info
        else:
            language_value = getattr(info, "language", None)
            if language_value is None and isinstance(info, Mapping):
                language_value = cast(Mapping[str, Any], info).get("language")

        language = language_value.strip() if isinstance(language_value, str) else None
        if not language:
            cfg_language = self.cfg.language
            language = cfg_language.strip() if isinstance(cfg_language, str) else None
        return language or "unknown"

    def _segment_output_error(
        self,
        index: int,
        violation: str,
        *,
        cause: Exception | None = None,
    ) -> ASROutputError:
        error = ASROutputError(
            "The ASR backend returned an invalid segment",
            context={"segment_index": index, "violation": violation},
        )
        if cause is not None:
            error.__cause__ = cause
        return error

    def _build_segments(self, raw_segments: Iterable[Any]) -> list[Segment]:
        """Convert validated backend segments into the public model."""
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

        for index, seg in enumerate(raw_segments):
            try:
                start = float(seg.start)
                end = float(seg.end)
            except Exception as exc:  # noqa: BLE001 - arbitrary backend object
                raise self._segment_output_error(index, "required_shape", cause=exc) from exc

            if not (math.isfinite(start) and math.isfinite(end)):
                raise self._segment_output_error(index, "non_finite_timing")
            if start < 0 or end < 0:
                raise self._segment_output_error(index, "negative_timing")
            if end < start:
                raise self._segment_output_error(index, "end_before_start")

            try:
                raw_text = seg.text
            except Exception as exc:  # noqa: BLE001
                raise self._segment_output_error(index, "missing_text", cause=exc) from exc
            if not isinstance(raw_text, str):
                raise self._segment_output_error(index, "text_not_string")

            try:
                words: list[Word] | None = None
                if extract_words:
                    raw_words = getattr(seg, "words", None)
                    if raw_words:
                        words = self._build_words(raw_words, index)

                raw_tokens = getattr(seg, "tokens", None)
                tokens = list(raw_tokens) if raw_tokens is not None else None
                avg_logprob = float(getattr(seg, "avg_logprob", 0.0))
                compression_ratio = float(getattr(seg, "compression_ratio", 1.0))
                no_speech_prob = float(getattr(seg, "no_speech_prob", 0.0))
                temperature = float(getattr(seg, "temperature", 0.0))
                seek = int(getattr(seg, "seek", 0))
            except Exception as exc:  # noqa: BLE001
                raise self._segment_output_error(
                    index, "invalid_optional_fields", cause=exc
                ) from exc

            validated.append(
                (
                    start,
                    end,
                    raw_text.strip(),
                    words,
                    tokens,
                    avg_logprob,
                    compression_ratio,
                    no_speech_prob,
                    temperature,
                    seek,
                )
            )

        if not all(validated[i][0] <= validated[i + 1][0] for i in range(len(validated) - 1)):
            validated = sorted(validated, key=lambda segment: (segment[0], segment[1]))

        return [
            Segment(
                id=index,
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
            for index, (
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

    def _build_words(
        self,
        raw_words: Iterable[Any],
        segment_index: int,
    ) -> list[Word]:
        """Convert backend word timestamps, skipping malformed optional words."""
        words: list[Word] = []
        for word_index, raw_word in enumerate(raw_words):
            try:
                word_text = getattr(raw_word, "word", "")
                word_start = float(getattr(raw_word, "start", 0.0))
                word_end = float(getattr(raw_word, "end", 0.0))
                word_probability = float(getattr(raw_word, "probability", 1.0))
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Invalid word at segment %d, word %d: %s",
                    segment_index,
                    word_index,
                    exc,
                )
                continue

            if not (math.isfinite(word_start) and math.isfinite(word_end)):
                logger.warning(
                    "Skipping word with non-finite timing at segment %d, word %d",
                    segment_index,
                    word_index,
                )
                continue

            word_probability = max(0.0, min(1.0, word_probability))
            words.append(
                Word(
                    word=str(word_text) if word_text else "",
                    start=word_start,
                    end=word_end,
                    probability=word_probability,
                )
            )
        return words

    def transcribe_file(self, audio_path: Path) -> Transcript:
        """Transcribe one audio file without synthetic fallback output."""
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        if not audio_path.is_file():
            raise IsADirectoryError(f"Audio path is not a file: {audio_path}")

        logger.info(
            "Transcribing file: %s",
            audio_path.name,
            extra={"file": audio_path.name},
        )

        try:
            result = self._transcribe_with_model(audio_path)
        except ASRInferenceError:
            raise
        except Exception as exc:  # noqa: BLE001 - provider exception remains chained
            raise ASRInferenceError(
                "ASR inference failed",
                context={
                    "backend": "faster-whisper",
                    "phase": "transcribe",
                },
            ) from exc

        if not isinstance(result, tuple) or len(result) != 2:
            raise ASROutputError(
                "The ASR backend returned an invalid transcription result",
                context={"violation": "result_shape"},
            )

        raw_segments, info = result
        segments = self._materialize_segments(raw_segments)
        segment_objects = self._build_segments(segments)

        duration_after_vad: float | None = None
        if info is not None and hasattr(info, "duration_after_vad"):
            try:
                duration_after_vad = float(info.duration_after_vad)
            except (TypeError, ValueError):
                duration_after_vad = None

        transcript = Transcript(
            file_name=audio_path.name,
            language=self._normalize_language(info),
            segments=segment_objects,
            duration_after_vad=duration_after_vad,
        )
        asr_meta: dict[str, Any] = {
            "asr_backend": "faster-whisper",
            "asr_device": self.cfg.device,
            "asr_compute_type": self.cfg.compute_type or "unknown",
            "asr_model_load_attempts": [dict(attempt) for attempt in self.model_load_attempts],
        }
        if self.model_load_warnings:
            asr_meta["asr_model_load_warnings"] = list(self.model_load_warnings)
        transcript.meta = {**(transcript.meta or {}), **asr_meta}
        return transcript

    def transcribe_many(
        self,
        audio_files: Iterable[Path],
    ) -> Iterable[Transcript]:
        """Transcribe files sequentially, propagating typed ASR failures."""
        for path in audio_files:
            yield self.transcribe_file(path)
