# cSpell: ignore samplerate
import inspect
import math
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Protocol, cast

from .cache import CachePaths
from .config import AsrConfig
from .models import Segment, Transcript

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
    from faster_whisper import WhisperModel as _WhisperModel  # type: ignore[reportMissingTypeStubs]

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
            import soundfile as sf  # type: ignore[reportMissingTypeStubs]

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
        print(
            f"=== Step 2: Loading Whisper model "
            f"{cfg.model_name} on {cfg.device} ({cfg.compute_type}) ==="
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
                            print(
                                f"[warn] Whisper model load failed on {device}: {load_err}",
                                file=sys.stderr,
                            )
                            print(
                                f"[warn] Retrying on {next_device_label} with compute_type={next_compute_type}",
                                file=sys.stderr,
                            )
                        else:
                            print(
                                f"[warn] Whisper model load failed with compute_type={compute_type}: {load_err}",
                                file=sys.stderr,
                            )
                            print(
                                f"[warn] Retrying on {next_device_label} with compute_type={next_compute_type}",
                                file=sys.stderr,
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
            print(f"[warn] Using dummy Whisper model fallback: {e}", file=sys.stderr)
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
        print(
            f"[warn] faster-whisper transcribe() does not support {detail}",
            file=sys.stderr,
        )
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
        validated: list[tuple[float, float, str]] = []
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
            validated.append((start, end, text_str.strip()))

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
            )
            for idx, (start, end, text) in enumerate(validated)
        ]
        return seg_objs

    def transcribe_file(self, audio_path: Path) -> Transcript:
        """
        Transcribe a single audio file into a Transcript.
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        if not audio_path.is_file():
            raise IsADirectoryError(f"Audio path is not a file: {audio_path}")

        print(f"\n[transcribe] {audio_path.name}")
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
                print(
                    f"[warn] Whisper inference failed for {audio_path.name}: {exc}. "
                    "Falling back to dummy output.",
                    file=sys.stderr,
                )
                segments, fallback_info = DummyWhisperModel(self.cfg).transcribe(str(audio_path))
                info = info or fallback_info
                used_dummy = True
                fallback_reason = str(exc)

        try:
            seg_objs = self._build_segments(segments)
        except Exception as exc:
            if used_dummy:
                # Building segments for dummy output should never fail; bubble up
                raise
            print(
                f"[warn] Whisper produced invalid segments for {audio_path.name}: {exc}. "
                "Falling back to dummy output.",
                file=sys.stderr,
            )
            segments, fallback_info = DummyWhisperModel(self.cfg).transcribe(str(audio_path))
            info = info or fallback_info
            seg_objs = self._build_segments(segments)
            used_dummy = True
            fallback_reason = fallback_reason or str(exc)

        self.using_dummy = used_dummy

        transcript = Transcript(
            file_name=audio_path.name,
            language=self._normalize_language(info),
            segments=seg_objs,
        )
        actual_device = "cpu" if used_dummy else self.cfg.device
        actual_compute_type = "n/a" if used_dummy else (self.cfg.compute_type or "unknown")
        asr_meta: dict[str, str | list[str]] = {
            "asr_backend": "dummy" if used_dummy else "faster-whisper",
            "asr_device": actual_device,
            "asr_compute_type": actual_compute_type,
        }
        if self.model_load_warnings:
            # Surface model load retries (e.g., GPU→CPU) for downstream diagnostics.
            asr_meta["asr_model_load_warnings"] = list(self.model_load_warnings)
        if used_dummy:
            reason = fallback_reason or self.model_load_error
            if reason:
                asr_meta["asr_fallback_reason"] = reason

        transcript.meta = {**(transcript.meta or {}), **asr_meta}

        return transcript

    def transcribe_many(self, audio_files: Iterable[Path]) -> Iterable[Transcript]:
        """
        Convenience generator to transcribe multiple files.
        """
        for path in audio_files:
            yield self.transcribe_file(path)
