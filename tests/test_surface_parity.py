"""Cross-surface transcript and receipt parity for canonical public APIs."""

from __future__ import annotations

import io
import json
import struct
import wave
from importlib import resources
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft7Validator, FormatChecker

pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")

from fastapi.testclient import TestClient  # noqa: E402

from transcription import _build_info  # noqa: E402
from transcription.api import transcribe_bytes, transcribe_file  # noqa: E402
from transcription.config import AsrConfig, Paths, TranscriptionConfig  # noqa: E402
from transcription.models import Transcript  # noqa: E402
from transcription.receipt import receipt_stable_projection  # noqa: E402
from transcription.service import create_app  # noqa: E402
from transcription.service_runtime import ASRRuntime, RuntimeProfile  # noqa: E402
from transcription.writers import write_json  # noqa: E402


class ParityEngine:
    """Deterministic fallback engine shared by every surface."""

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


def asr_config() -> AsrConfig:
    return AsrConfig(
        model_name="tiny",
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


def install_fake_audio_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    def normalize_all(paths: Any) -> None:
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
    monkeypatch.setattr(
        "transcription.service_transcribe.validate_audio_format",
        lambda _path: None,
    )


def load_schema(name: str) -> dict[str, Any]:
    resource = resources.files("transcription").joinpath("schemas", name)
    return json.loads(resource.read_text(encoding="utf-8"))


def validate_document(document: dict[str, Any]) -> None:
    transcript_errors = list(
        Draft7Validator(
            load_schema("transcript-v2.schema.json"),
            format_checker=FormatChecker(),
        ).iter_errors(document)
    )
    assert transcript_errors == []

    receipt = document["meta"]["receipt"]
    receipt_errors = list(
        Draft7Validator(
            load_schema("receipt-v1.schema.json"),
            format_checker=FormatChecker(),
        ).iter_errors(receipt)
    )
    assert receipt_errors == []


def semantic_projection(document: dict[str, Any]) -> dict[str, Any]:
    meta = document["meta"]
    return {
        "schema_version": document["schema_version"],
        "file": document["file"],
        "audio_file": meta["audio_file"],
        "language": document["language"],
        "segments": document["segments"],
        "runtime": {
            "backend": meta["asr_backend"],
            "model": meta["model_name"],
            "device": meta["device"],
            "compute_type": meta["compute_type"],
            "attempts": meta["asr_model_load_attempts"],
        },
        "receipt": receipt_stable_projection(meta["receipt"]),
    }


def test_file_bytes_and_rest_have_equivalent_transcript_truth(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ParityEngine.instances = 0
    ParityEngine.closes = 0
    monkeypatch.setattr(_build_info, "SOURCE_COMMIT", "abcdef123456")
    monkeypatch.setattr(_build_info, "BUILD_ID", "surface-parity-1")
    install_fake_audio_boundary(monkeypatch)

    source_name = "surface.mp3"
    audio_path = tmp_path / source_name
    audio_path.write_bytes(wav_bytes())

    file_engine = ParityEngine(asr_config())
    file_root = tmp_path / "file-project"
    file_result = transcribe_file(
        audio_path,
        file_root,
        config(),
        _engine=file_engine,
    )
    file_document = json.loads(
        (Paths(root=file_root).json_dir / "surface.json").read_text(
            encoding="utf-8"
        )
    )

    monkeypatch.setattr("transcription.asr_engine.TranscriptionEngine", ParityEngine)
    bytes_result = transcribe_bytes(
        wav_bytes(),
        config(),
        file_name=source_name,
    )
    bytes_path = tmp_path / "bytes.json"
    write_json(bytes_result, bytes_path)
    bytes_document = json.loads(bytes_path.read_text(encoding="utf-8"))

    runtime = ASRRuntime(
        RuntimeProfile.from_config(config()),
        engine_factory=ParityEngine,
    )
    app = create_app(runtime_factory=lambda: runtime)
    with TestClient(app) as client:
        response = client.post(
            "/transcribe",
            files={"audio": (source_name, wav_bytes(), "audio/mpeg")},
        )
    assert response.status_code == 200, response.text
    rest_document = response.json()
    assert rest_document["file"] == source_name
    assert rest_document["file_name"] == source_name
    assert rest_document["meta"]["audio_file"] == source_name

    documents = [file_document, bytes_document, rest_document]
    for document in documents:
        validate_document(document)
        assert document["file"] == source_name
        assert document["meta"]["audio_file"] == source_name

    assert "file_name" not in file_document
    assert "file_name" not in bytes_document

    projections = [semantic_projection(document) for document in documents]
    assert projections[1:] == projections[:-1]

    assert file_result.file_name == bytes_result.file_name == source_name
    assert projections[0]["runtime"] == {
        "backend": "faster-whisper",
        "model": "tiny",
        "device": "cpu",
        "compute_type": "int8",
        "attempts": [
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
        ],
    }
    assert ParityEngine.instances == 3
    assert ParityEngine.closes == 2


def test_operation_owned_file_engine_closes_without_closing_injected_engines(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ParityEngine.instances = 0
    ParityEngine.closes = 0
    install_fake_audio_boundary(monkeypatch)
    monkeypatch.setattr("transcription.asr_engine.TranscriptionEngine", ParityEngine)

    source_name = "owned.flac"
    audio_path = tmp_path / source_name
    audio_path.write_bytes(wav_bytes())

    result = transcribe_file(
        audio_path,
        tmp_path / "owned-project",
        config(),
    )

    assert result.file_name == source_name
    assert result.meta["audio_file"] == source_name
    assert ParityEngine.instances == 1
    assert ParityEngine.closes == 1
