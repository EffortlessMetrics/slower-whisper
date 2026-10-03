#!/usr/bin/env python3
"""Installed-artifact first-use probes for public capability profiles.

This script is repository-only acceptance tooling. Workflows copy it outside the
checkout and execute it with ``python -I`` after installing a wheel or sdist.
"""

from __future__ import annotations

import argparse
import io
import os
import struct
import tempfile
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


def _assert_site_packages(module: object) -> None:
    module_file = getattr(module, "__file__", None)
    assert module_file is not None, module
    path = Path(module_file).resolve()
    assert "site-packages" in path.parts, path


def _wav_bytes(*, frames: int = 160) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16_000)
        wav.writeframes(struct.pack(f"<{frames}h", *([0] * frames)))
    return buffer.getvalue()


def _probe_compatibility() -> None:
    import slower_whisper
    import transcription
    from slower_whisper import WhisperModel
    from transcription.models import Segment as InternalSegment
    from transcription.models import Transcript
    from transcription.models import Word as InternalWord

    _assert_site_packages(slower_whisper)
    _assert_site_packages(transcription)

    class CompatEngine:
        def __init__(self) -> None:
            self.cfg = SimpleNamespace(
                language=None,
                task="transcribe",
                beam_size=5,
                word_timestamps=False,
                vad_filter=True,
                vad_min_silence_ms=500,
                device="cpu",
                compute_type="int8",
            )

        def transcribe_file(self, audio_path: Path) -> Transcript:
            return Transcript(
                file_name=audio_path.name,
                language="en",
                segments=[
                    InternalSegment(
                        id=0,
                        start=0.0,
                        end=1.0,
                        text="hello",
                        words=[
                            InternalWord(
                                word=" hello",
                                start=0.0,
                                end=0.5,
                                probability=0.9,
                            )
                        ],
                        tokens=[1, 2],
                        avg_logprob=-0.1,
                        compression_ratio=1.0,
                        no_speech_prob=0.01,
                        temperature=0.0,
                        seek=0,
                    )
                ],
                duration_after_vad=0.9,
            )

    model = WhisperModel("tiny", device="cpu", compute_type="int8")
    model._engine = CompatEngine()

    with tempfile.TemporaryDirectory() as tmp:
        audio = Path(tmp) / "fixture.wav"
        audio.write_bytes(_wav_bytes())
        segments, info = model.transcribe(
            audio,
            language="en",
            task="transcribe",
            beam_size=3,
            word_timestamps=True,
            vad_filter=False,
        )

    assert isinstance(segments, list)
    assert len(segments) == 1
    segment = segments[0]
    assert len(segment) == 11
    assert tuple(segment)[4] == "hello"
    assert segment.text == "hello"
    assert segment.words is not None
    assert segment.words[0].word == " hello"
    assert segment.words[0].probability == 0.9
    assert info.language == "en"
    assert info.duration == 1.0
    assert info.duration_after_vad == 0.9
    assert info.transcription_options["beam_size"] == 3
    assert model.last_transcript is not None
    assert model.device == "cpu"
    assert model.compute_type == "int8"


def _probe_api() -> None:
    from fastapi.testclient import TestClient

    from transcription.config import AsrConfig, TranscriptionConfig
    from transcription.models import Transcript
    from transcription.service import create_app
    from transcription.service_runtime import ASRRuntime, RuntimeProfile

    class RuntimeEngine:
        def __init__(self, cfg: AsrConfig) -> None:
            self.cfg = cfg
            self.model_load_attempts = [
                {
                    "device": cfg.device,
                    "compute_type": cfg.compute_type or "unknown",
                    "outcome": "selected",
                    "reason_code": "ok",
                }
            ]
            self.close_count = 0

        def close(self) -> None:
            self.close_count += 1

    config = TranscriptionConfig(
        model="tiny",
        device="cpu",
        compute_type="int8",
        language="en",
        task="transcribe",
        beam_size=3,
        vad_min_silence_ms=400,
        word_timestamps=False,
    )
    created: list[RuntimeEngine] = []

    def engine_factory(cfg: AsrConfig) -> RuntimeEngine:
        engine = RuntimeEngine(cfg)
        created.append(engine)
        return engine

    runtime = ASRRuntime(
        RuntimeProfile.from_config(config),
        engine_factory=engine_factory,
    )
    app = create_app(runtime_factory=lambda: runtime)

    def deterministic_transcribe(
        audio_path: Path,
        _root: Path,
        _config: TranscriptionConfig,
        *,
        _engine: RuntimeEngine,
    ) -> Transcript:
        return Transcript(
            file_name=audio_path.name,
            language="en",
            segments=[],
            meta={
                "asr_backend": "faster-whisper",
                "asr_device": _engine.cfg.device,
                "asr_compute_type": _engine.cfg.compute_type,
                "asr_model_load_attempts": list(_engine.model_load_attempts),
            },
        )

    with (
        patch("transcription.service_transcribe.validate_audio_format"),
        patch(
            "transcription.service_transcribe.transcribe_file",
            side_effect=deterministic_transcribe,
        ),
        TestClient(app) as client,
    ):
        live = client.get("/health/live")
        ready = client.get("/health/ready")
        stream_config = client.get("/stream/config")
        response = client.post(
            "/transcribe",
            files={"audio": ("fixture.wav", _wav_bytes(), "audio/wav")},
        )

    assert live.status_code == 200
    ready_body = ready.json()
    assert ready_body["checks"]["runtime"]["ready"] is True
    assert ready_body["checks"]["resources"]["status"] == "ok"
    ffmpeg_status = ready_body["checks"]["ffmpeg"]["status"]
    if ffmpeg_status == "ok":
        assert ready.status_code == 200, ready.text
        assert ready_body["status"] == "ready"
        assert ready_body["healthy"] is True
    else:
        assert ffmpeg_status == "error", ready_body
        assert ready.status_code == 503, ready.text
        assert ready_body["status"] == "degraded"
        assert ready_body["healthy"] is False
    assert stream_config.status_code == 200
    assert stream_config.json()["supported_audio_formats"] == ["pcm_s16le"]
    assert stream_config.json()["optional_live_enrichment"] is False
    assert response.status_code == 200, response.text
    assert response.json()["segments"] == []
    assert len(created) == 1
    assert created[0].close_count == 1


def _probe_core() -> None:
    _probe_compatibility()
    _probe_api()


def _write_fixture(path: Path, *, seconds: float = 1.0) -> None:
    frames = int(16_000 * seconds)
    path.write_bytes(_wav_bytes(frames=frames))


def _probe_heavy() -> None:
    os.environ["SLOWER_WHISPER_PYANNOTE_MODE"] = "stub"

    import accelerate
    import numpy as np
    import torch
    import transformers

    from transcription.diarization import Diarizer
    from transcription.emotion import EMOTION_AVAILABLE, EmotionRecognizer
    from transcription.local_llm_provider import is_available
    from transcription.semantic_adapter import LocalLLMSemanticAdapter

    assert accelerate.__version__
    assert torch.__version__
    assert transformers.__version__

    with tempfile.TemporaryDirectory() as tmp:
        audio = Path(tmp) / "fixture.wav"
        _write_fixture(audio)
        turns = Diarizer(
            device="cpu",
            min_speakers=2,
            max_speakers=2,
        ).run(audio)

    assert [turn.speaker_id for turn in turns] == ["SPEAKER_00", "SPEAKER_01"]
    assert turns[0].start == 0.0
    assert turns[0].end <= turns[1].start
    assert turns[1].end >= 1.0

    assert EMOTION_AVAILABLE
    recognizer = EmotionRecognizer()
    samples = np.zeros(16_000, dtype=np.float32)
    normalized, valid = recognizer._validate_audio(samples, 16_000)
    assert valid
    assert normalized.dtype == np.float32
    assert normalized.shape == samples.shape
    assert recognizer._dimensional_model is None
    assert recognizer._categorical_model is None

    adapter = LocalLLMSemanticAdapter()
    assert adapter.model
    assert is_available()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("core", "heavy"), required=True)
    args = parser.parse_args()

    if args.profile == "core":
        _probe_core()
    else:
        _probe_heavy()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
