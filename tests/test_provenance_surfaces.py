"""Automatic receipt attachment across file, bytes, and REST surfaces."""

from __future__ import annotations

import io
import struct
import wave
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")

from fastapi.testclient import TestClient  # noqa: E402

from transcription import _build_info  # noqa: E402
from transcription.api import transcribe_bytes, transcribe_file  # noqa: E402
from transcription.config import AsrConfig, TranscriptionConfig  # noqa: E402
from transcription.models import Transcript  # noqa: E402
from transcription.service import create_app  # noqa: E402
from transcription.service_runtime import ASRRuntime, RuntimeProfile  # noqa: E402


class FakeEngine:
    instances = 0
    closes = 0

    def __init__(self, cfg: AsrConfig) -> None:
        type(self).instances += 1
        self.cfg = cfg
        self.model_load_attempts = [
            {
                "device": "cuda",
                "compute_type": "float16",
                "outcome": "failed",
                "reason_code": "asr_model_load_failed",
            },
            {
                "device": "cpu",
                "compute_type": "int8",
                "outcome": "selected",
                "reason_code": "ok",
            },
        ]
        self.cfg.device = "cpu"
        self.cfg.compute_type = "int8"

    def transcribe_file(self, audio_path: Path) -> Transcript:
        return Transcript(
            file_name=audio_path.name,
            language="en",
            segments=[],
            meta={
                "asr_backend": "faster-whisper",
                "asr_model": self.cfg.model_name,
                "asr_device": self.cfg.device,
                "asr_compute_type": self.cfg.compute_type,
                "asr_model_load_attempts": list(self.model_load_attempts),
            },
        )

    def close(self) -> None:
        type(self).closes += 1


def config() -> TranscriptionConfig:
    return TranscriptionConfig(
        model="tiny",
        device="cuda",
        compute_type="float16",
        language="en",
        task="transcribe",
        beam_size=3,
        vad_min_silence_ms=400,
        word_timestamps=False,
    )


def wav_bytes() -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16_000)
        wav.writeframes(struct.pack("h" * 160, *[0] * 160))
    return buffer.getvalue()


def install_fake_normalization(monkeypatch) -> None:
    def normalize_all(paths) -> None:
        paths.norm_dir.mkdir(parents=True, exist_ok=True)
        for raw_path in paths.raw_dir.iterdir():
            (paths.norm_dir / f"{raw_path.stem}.wav").write_bytes(wav_bytes())

    def normalize_single(_input: Path, output: Path) -> None:
        output.write_bytes(wav_bytes())

    monkeypatch.setattr("transcription.audio_io.normalize_all", normalize_all)
    monkeypatch.setattr("transcription.audio_io.normalize_single", normalize_single)
    monkeypatch.setattr(
        "transcription.api._get_wav_duration_seconds",
        lambda _path: 0.01,
    )
    monkeypatch.setattr(
        "transcription.api._maybe_run_diarization",
        lambda transcript, _path, _config: transcript,
    )
    monkeypatch.setattr(
        "transcription.api._maybe_build_chunks",
        lambda transcript, _config: transcript,
    )


def assert_receipt(receipt: dict) -> None:
    assert receipt["model"] == "tiny"
    assert receipt["backend"] == "faster-whisper"
    assert receipt["device"] == "cpu"
    assert receipt["compute_type"] == "int8"
    assert receipt["git_commit"] == "abcdef123456"
    assert receipt["build_id"] == "github-12345-1"
    assert [attempt["outcome"] for attempt in receipt["model_load_attempts"]] == [
        "failed",
        "selected",
    ]


def test_direct_file_transcription_attaches_receipt(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(_build_info, "SOURCE_COMMIT", "abcdef123456")
    monkeypatch.setattr(_build_info, "BUILD_ID", "github-12345-1")
    install_fake_normalization(monkeypatch)
    audio = tmp_path / "clip.wav"
    audio.write_bytes(wav_bytes())
    engine = FakeEngine(config().to_asr_config())

    transcript = transcribe_file(
        audio,
        tmp_path / "project",
        config(),
        _engine=engine,
    )

    assert_receipt(transcript.meta["receipt"])
    assert (tmp_path / "project" / "whisper_json" / "clip.json").is_file()


def test_bytes_transcription_attaches_receipt(
    monkeypatch,
) -> None:
    monkeypatch.setattr(_build_info, "SOURCE_COMMIT", "abcdef123456")
    monkeypatch.setattr(_build_info, "BUILD_ID", "github-12345-1")
    install_fake_normalization(monkeypatch)
    monkeypatch.setattr("transcription.asr_engine.TranscriptionEngine", FakeEngine)

    transcript = transcribe_bytes(
        wav_bytes(),
        config(),
        file_name="memory.wav",
    )

    assert transcript.file_name == "memory.wav"
    assert_receipt(transcript.meta["receipt"])


def test_rest_transcription_attaches_receipt_and_reuses_runtime(
    monkeypatch,
) -> None:
    FakeEngine.instances = 0
    FakeEngine.closes = 0
    monkeypatch.setattr(_build_info, "SOURCE_COMMIT", "abcdef123456")
    monkeypatch.setattr(_build_info, "BUILD_ID", "github-12345-1")
    install_fake_normalization(monkeypatch)
    monkeypatch.setattr(
        "transcription.service_transcribe.validate_audio_format",
        lambda _path: None,
    )

    runtime = ASRRuntime(
        RuntimeProfile.from_config(config()),
        engine_factory=FakeEngine,
    )
    app = create_app(runtime_factory=lambda: runtime)

    with TestClient(app) as client:
        first = client.post(
            "/transcribe",
            files={"audio": ("silence.wav", wav_bytes(), "audio/wav")},
        )
        second = client.post(
            "/transcribe",
            files={"audio": ("silence.wav", wav_bytes(), "audio/wav")},
        )

    assert first.status_code == 200, first.text
    assert second.status_code == 200, second.text
    assert_receipt(first.json()["meta"]["receipt"])
    assert_receipt(second.json()["meta"]["receipt"])
    assert FakeEngine.instances == 1
    assert FakeEngine.closes == 1
