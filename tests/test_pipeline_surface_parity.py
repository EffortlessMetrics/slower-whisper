"""Batch pipeline parity and operation-owned engine lifecycle."""

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

from transcription import _build_info
from transcription.api import transcribe_file
from transcription.config import AppConfig, AsrConfig, Paths, TranscriptionConfig
from transcription.exceptions import TranscriptionError
from transcription.models import Segment, Transcript, Word
from transcription.pipeline import run_pipeline
from transcription.receipt import receipt_stable_projection


class ParityEngine:
    """Deterministic fallback engine with observable lifecycle."""

    instances = 0
    closes = 0
    calls = 0
    fail_inference = False

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
        type(self).calls += 1
        if type(self).fail_inference:
            raise RuntimeError("backend failed")
        return Transcript(
            file_name=audio_path.name,
            language="en",
            segments=[
                Segment(
                    id=0,
                    start=0.0,
                    end=0.01,
                    text="hello",
                    words=[
                        Word(
                            word="hello",
                            start=0.0,
                            end=0.01,
                            probability=0.99,
                        ),
                    ],
                ),
            ],
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


def reset_engine() -> None:
    ParityEngine.instances = 0
    ParityEngine.closes = 0
    ParityEngine.calls = 0
    ParityEngine.fail_inference = False


def config() -> TranscriptionConfig:
    return TranscriptionConfig(
        model="tiny",
        device="cuda",
        compute_type="float16",
        language="en",
        task="transcribe",
        beam_size=3,
        vad_min_silence_ms=400,
        word_timestamps=True,
        skip_existing_json=False,
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
        word_timestamps=True,
    )


def wav_bytes() -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16_000)
        wav.writeframes(struct.pack("h" * 160, *[0] * 160))
    return buffer.getvalue()


def normalize_all(paths: Paths) -> None:
    paths.norm_dir.mkdir(parents=True, exist_ok=True)
    for raw_path in sorted(paths.raw_dir.iterdir()):
        if raw_path.is_file():
            (paths.norm_dir / f"{raw_path.stem}.wav").write_bytes(wav_bytes())


def prepare_project(root: Path, *source_names: str) -> Paths:
    paths = Paths(root=root)
    paths.raw_dir.mkdir(parents=True, exist_ok=True)
    for name in source_names:
        (paths.raw_dir / name).write_bytes(wav_bytes())
    return paths


def app_config(paths: Paths) -> AppConfig:
    return AppConfig(
        paths=paths,
        asr=asr_config(),
        skip_existing_json=False,
    )


def load_schema(name: str) -> dict[str, Any]:
    resource = resources.files("transcription").joinpath("schemas", name)
    return json.loads(resource.read_text(encoding="utf-8"))


def validate_document(document: dict[str, Any]) -> None:
    assert not list(
        Draft7Validator(
            load_schema("transcript-v2.schema.json"),
            format_checker=FormatChecker(),
        ).iter_errors(document)
    )
    assert not list(
        Draft7Validator(
            load_schema("receipt-v1.schema.json"),
            format_checker=FormatChecker(),
        ).iter_errors(document["meta"]["receipt"])
    )


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


def install_runtime_patches(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("transcription.pipeline.TranscriptionEngine", ParityEngine)
    monkeypatch.setattr("transcription.audio_io.normalize_all", normalize_all)
    monkeypatch.setattr("transcription.api._get_wav_duration_seconds", lambda _path: 0.01)
    monkeypatch.setattr(
        "transcription.api._maybe_run_diarization",
        lambda transcript, _path, _config: transcript,
    )
    monkeypatch.setattr(
        "transcription.api._maybe_build_chunks",
        lambda transcript, _config: transcript,
    )


def test_batch_matches_canonical_file_and_closes_one_engine(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reset_engine()
    monkeypatch.setattr(_build_info, "SOURCE_COMMIT", "abcdef123456")
    monkeypatch.setattr(_build_info, "BUILD_ID", "pipeline-parity-1")
    install_runtime_patches(monkeypatch)

    source_name = "surface.mp3"
    baseline_source = tmp_path / source_name
    baseline_source.write_bytes(wav_bytes())
    baseline_root = tmp_path / "baseline"
    transcribe_file(
        baseline_source,
        baseline_root,
        config(),
        _engine=ParityEngine(asr_config()),
    )
    baseline = json.loads(
        (Paths(root=baseline_root).json_dir / "surface.json").read_text(
            encoding="utf-8"
        )
    )

    batch_paths = prepare_project(tmp_path / "batch", source_name)
    result = run_pipeline(app_config(batch_paths), diarization_config=None)
    batch = json.loads(
        (batch_paths.json_dir / "surface.json").read_text(encoding="utf-8")
    )

    validate_document(baseline)
    validate_document(batch)
    assert semantic_projection(batch) == semantic_projection(baseline)
    assert result.processed == 1
    assert result.failed == 0
    assert result.file_results[0].file_name == source_name
    assert ParityEngine.instances == 2
    assert ParityEngine.calls == 2
    assert ParityEngine.closes == 1


def test_batch_rejects_ambiguous_raw_source_identity_before_normalization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reset_engine()
    install_runtime_patches(monkeypatch)
    paths = prepare_project(tmp_path, "surface.mp3", "Surface.flac")

    def unexpected_normalization(_paths: Paths) -> None:
        pytest.fail("ambiguous raw sources reached normalization")

    monkeypatch.setattr(
        "transcription.audio_io.normalize_all",
        unexpected_normalization,
    )

    with pytest.raises(
        TranscriptionError,
        match="would overwrite the same normalized WAV",
    ):
        run_pipeline(app_config(paths), diarization_config=None)

    assert ParityEngine.instances == 0
    assert ParityEngine.calls == 0
    assert ParityEngine.closes == 0


def test_batch_closes_engine_after_inference_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reset_engine()
    ParityEngine.fail_inference = True
    install_runtime_patches(monkeypatch)
    paths = prepare_project(tmp_path, "failure.wav")

    result = run_pipeline(app_config(paths), diarization_config=None)

    assert result.processed == 0
    assert result.failed == 1
    assert result.file_results[0].status == "error"
    assert result.file_results[0].error_message is not None
    assert ParityEngine.instances == 1
    assert ParityEngine.calls == 1
    assert ParityEngine.closes == 1


def test_batch_closes_engine_when_writer_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reset_engine()
    install_runtime_patches(monkeypatch)
    paths = prepare_project(tmp_path, "writer.wav")

    def fail_write(_transcript: Transcript, _path: Path) -> None:
        raise RuntimeError("writer failed")

    monkeypatch.setattr("transcription.pipeline.writers.write_json", fail_write)

    with pytest.raises(RuntimeError, match="writer failed"):
        run_pipeline(app_config(paths), diarization_config=None)

    assert ParityEngine.instances == 1
    assert ParityEngine.calls == 1
    assert ParityEngine.closes == 1


def test_empty_batch_does_not_construct_an_engine(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reset_engine()
    install_runtime_patches(monkeypatch)
    paths = Paths(root=tmp_path)

    result = run_pipeline(app_config(paths), diarization_config=None)

    assert result.total_files == 0
    assert ParityEngine.instances == 0
    assert ParityEngine.closes == 0
