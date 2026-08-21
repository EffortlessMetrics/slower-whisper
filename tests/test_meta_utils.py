"""Tests for canonical metadata and automatic receipt attachment."""

from __future__ import annotations

import json
from pathlib import Path

from transcription import _build_info
from transcription.meta_utils import build_generation_metadata
from transcription.models import Transcript
from transcription.receipt import (
    receipt_stable_projection,
    validate_receipt,
)


def build_meta(transcript: Transcript, **overrides):
    values = {
        "duration_sec": 12.34,
        "model_name": "large-v3",
        "config_device": "cuda",
        "config_compute_type": "float16",
        "beam_size": 5,
        "vad_min_silence_ms": 500,
        "language_hint": None,
        "task": "transcribe",
        "pipeline_version": "2.0.2",
        "root": Path("/private/caller/project"),
    }
    values.update(overrides)
    return build_generation_metadata(transcript, **values)


def test_build_generation_metadata_prefers_actual_runtime_and_attaches_receipt(
    monkeypatch,
) -> None:
    monkeypatch.setattr(_build_info, "SOURCE_COMMIT", "abcdef123456")
    monkeypatch.setattr(_build_info, "BUILD_ID", "github-12345-1")
    attempts = [
        {
            "device": "cuda",
            "compute_type": "float16",
            "outcome": "failed",
            "reason_code": "asr_model_load_failed",
            "private_error": "/srv/models/private",
        },
        {
            "device": "cpu",
            "compute_type": "int8",
            "outcome": "selected",
            "reason_code": "ok",
        },
    ]
    transcript = Transcript(
        file_name="clip.wav",
        language="en",
        segments=[],
        meta={
            "asr_backend": "faster-whisper",
            "asr_model": "large-v3",
            "asr_device": "cpu",
            "asr_compute_type": "int8",
            "asr_model_load_attempts": attempts,
        },
    )

    meta = build_meta(transcript)

    assert meta["device"] == "cpu"
    assert meta["compute_type"] == "int8"
    assert meta["asr_backend"] == "faster-whisper"
    receipt = meta["receipt"]
    assert receipt["model"] == "large-v3"
    assert receipt["backend"] == "faster-whisper"
    assert receipt["device"] == "cpu"
    assert receipt["compute_type"] == "int8"
    assert receipt["git_commit"] == "abcdef123456"
    assert receipt["build_id"] == "github-12345-1"
    assert receipt["model_load_attempts"] == [
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
    assert meta["asr_model_load_attempts"] == receipt["model_load_attempts"]
    assert validate_receipt(receipt) == []
    assert "/private/caller/project" not in json.dumps(receipt)
    assert "/srv/models/private" not in json.dumps(receipt)
    assert "/srv/models/private" not in json.dumps(meta)


def test_build_generation_metadata_falls_back_to_runtime_candidates() -> None:
    transcript = Transcript(file_name="clip.wav", language="en", segments=[], meta={})

    meta = build_meta(
        transcript,
        duration_sec=1.0,
        model_name="tiny",
        language_hint="en",
        runtime_device_candidates=("cpu",),
        runtime_compute_candidates=("int8",),
    )

    assert meta["device"] == "cpu"
    assert meta["compute_type"] == "int8"
    assert meta["audio_file"] == "clip.wav"
    assert meta["receipt"]["model_load_attempts"] == [
        {
            "device": "cpu",
            "compute_type": "int8",
            "outcome": "selected",
            "reason_code": "ok",
        }
    ]


def test_receipt_config_hash_is_stable_outside_volatile_fields(monkeypatch) -> None:
    monkeypatch.setattr(_build_info, "SOURCE_COMMIT", "abcdef123456")
    monkeypatch.setattr(_build_info, "BUILD_ID", "github-12345-1")
    transcript = Transcript(
        file_name="clip.wav",
        language="en",
        segments=[],
        meta={
            "asr_backend": "faster-whisper",
            "asr_device": "cpu",
            "asr_compute_type": "int8",
        },
    )

    first = build_meta(transcript)["receipt"]
    second = build_meta(transcript)["receipt"]

    assert first["config_hash"] == second["config_hash"]
    assert receipt_stable_projection(first) == receipt_stable_projection(second)
    assert first["run_id"] != second["run_id"]
