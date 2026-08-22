"""Process-runtime backend adapter for revision-aware streaming ASR.

The adapter owns only the temporary PCM-to-WAV boundary and text projection.
Model lifecycle, selected backend/device, inference capacity, and typed ASR
failure remain authoritative in :class:`transcription.service_runtime.ASRRuntime`.
"""

from __future__ import annotations

import asyncio
import tempfile
import wave
from pathlib import Path
from typing import Any, cast

from .config import TranscriptionConfig, WhisperTask
from .exceptions import ASROutputError, RuntimeNotReadyError
from .models import Transcript
from .service_runtime import ASRRuntime


class RuntimeIncrementalASRBackend:
    """Adapt the process-owned file engine to the incremental PCM backend port."""

    def __init__(self, runtime: ASRRuntime) -> None:
        self.runtime = runtime
        profile = runtime.profile
        self._runtime_config = TranscriptionConfig(
            model=profile.model,
            device=profile.device,
            compute_type=profile.compute_type,
            language=profile.language,
            task=cast(WhisperTask, profile.task),
            beam_size=profile.beam_size,
            vad_min_silence_ms=profile.vad_min_silence_ms,
            word_timestamps=profile.word_timestamps,
            skip_existing_json=False,
        )

    async def transcribe(
        self,
        pcm_s16le: bytes,
        *,
        sample_rate: int,
        start_sample: int,
        end_sample: int,
    ) -> str:
        """Transcribe one active-prefix snapshot through the owned runtime."""
        if not isinstance(pcm_s16le, bytes):
            raise TypeError("pcm_s16le must be bytes")
        if sample_rate != 16_000:
            raise ValueError("runtime incremental ASR requires a 16 kHz sample rate")
        if len(pcm_s16le) % 2 != 0:
            raise ValueError("pcm_s16le must contain complete 16-bit samples")
        if start_sample < 0 or end_sample < start_sample:
            raise ValueError("incremental sample bounds are invalid")
        if not self.runtime.ready:
            raise RuntimeNotReadyError(
                "The process-owned ASR runtime is not ready",
                context={"state": self.runtime.state.value},
            )

        with tempfile.TemporaryDirectory(prefix="slower-whisper-stream-") as temp_dir:
            root = Path(temp_dir)
            audio_path = root / "active-prefix.wav"
            await asyncio.to_thread(
                _write_pcm_wav,
                audio_path,
                pcm_s16le,
                sample_rate,
            )
            transcript = await self.runtime.transcribe_file(
                _transcribe_owned_engine,
                audio_path=audio_path,
                root=root,
                config=self._runtime_config,
            )

        return _transcript_text(transcript)


def _write_pcm_wav(
    path: Path,
    pcm_s16le: bytes,
    sample_rate: int,
) -> None:
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_s16le)


def _transcribe_owned_engine(
    audio_path: str | Path,
    _root: str | Path,
    _config: TranscriptionConfig,
    *,
    _engine: Any | None = None,
) -> Transcript:
    if _engine is None:
        raise RuntimeNotReadyError("The process-owned ASR engine is unavailable")
    return cast(Transcript, _engine.transcribe_file(Path(audio_path)))


def _transcript_text(transcript: Transcript) -> str:
    parts: list[str] = []
    for segment in transcript.segments:
        text = segment.text
        if not isinstance(text, str):
            raise ASROutputError(
                "Incremental ASR transcript segment text is invalid",
                context={"violation": "segment_text_not_string"},
            )
        normalized = text.strip()
        if normalized:
            parts.append(normalized)
    return " ".join(parts)
