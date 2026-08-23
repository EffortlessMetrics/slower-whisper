"""Canonical receipt attachment at the shared JSON serialization seam."""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path

import pytest
from jsonschema import Draft7Validator, FormatChecker

from transcription import _build_info
from transcription.generation_receipt import ensure_generation_receipt
from transcription.meta_utils import build_generation_metadata
from transcription.models import Transcript
from transcription.receipt import receipt_stable_projection
from transcription.writers import write_json


def legacy_generation_meta() -> dict:
    return {
        "generated_at": "2026-08-22T12:00:00Z",
        "audio_file": "clip.wav",
        "audio_duration_sec": 1.0,
        "model_name": "tiny",
        "device": "cuda",
        "compute_type": "float16",
        "beam_size": 5,
        "vad_min_silence_ms": 500,
        "language_hint": "en",
        "task": "transcribe",
        "pipeline_version": "2.0.2",
        "root": "/private/caller/project",
        "asr_backend": "faster-whisper",
        "asr_model": "tiny",
        "asr_device": "cpu",
        "asr_compute_type": "int8",
        "asr_model_load_attempts": [
            {
                "device": "cuda",
                "compute_type": "float16",
                "outcome": "failed",
                "reason_code": "asr_model_load_failed",
                "private_error": "/srv/private/model",
            },
            {
                "device": "cpu",
                "compute_type": "int8",
                "outcome": "selected",
                "reason_code": "ok",
            },
        ],
    }


def canonical_meta() -> dict:
    transcript = Transcript(
        file_name="clip.wav",
        language="en",
        segments=[],
        meta={
            "asr_backend": "faster-whisper",
            "asr_model": "tiny",
            "asr_device": "cpu",
            "asr_compute_type": "int8",
            "asr_model_load_attempts": legacy_generation_meta()[
                "asr_model_load_attempts"
            ],
        },
    )
    return build_generation_metadata(
        transcript,
        duration_sec=1.0,
        model_name="tiny",
        config_device="cuda",
        config_compute_type="float16",
        beam_size=5,
        vad_min_silence_ms=500,
        language_hint="en",
        task="transcribe",
        pipeline_version="2.0.2",
        root=Path("/private/caller/project"),
    )


def test_legacy_generation_metadata_receipt_matches_canonical_surface(
    monkeypatch,
) -> None:
    monkeypatch.setattr(_build_info, "SOURCE_COMMIT", "abcdef123456")
    monkeypatch.setattr(_build_info, "BUILD_ID", "parity-620")

    legacy = ensure_generation_receipt(legacy_generation_meta())
    canonical = canonical_meta()

    assert receipt_stable_projection(legacy["receipt"]) == receipt_stable_projection(
        canonical["receipt"]
    )
    assert legacy["receipt"]["device"] == "cpu"
    assert legacy["receipt"]["compute_type"] == "int8"
    assert legacy["asr_model_load_attempts"] == legacy["receipt"][
        "model_load_attempts"
    ]
    serialized = json.dumps(legacy)
    assert "/srv/private/model" not in serialized
    assert "/private/caller/project" not in json.dumps(legacy["receipt"])


def test_incomplete_generation_metadata_remains_unknown() -> None:
    incomplete = {
        "model_name": "tiny",
        "device": "cpu",
        "compute_type": "int8",
    }

    assert ensure_generation_receipt(incomplete) == incomplete


def test_existing_valid_receipt_is_preserved(monkeypatch) -> None:
    monkeypatch.setattr(_build_info, "SOURCE_COMMIT", "abcdef123456")
    meta = ensure_generation_receipt(legacy_generation_meta())
    existing = dict(meta["receipt"])

    second = ensure_generation_receipt(meta)

    assert second["receipt"] == existing


def test_existing_invalid_receipt_fails_closed() -> None:
    with pytest.raises(ValueError, match="meta.receipt is invalid"):
        ensure_generation_receipt({"receipt": {"model": "tiny"}})
    with pytest.raises(ValueError, match="must be an object"):
        ensure_generation_receipt({"receipt": "not-an-object"})


def test_write_json_attaches_receipt_and_validates_complete_transcript(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(_build_info, "SOURCE_COMMIT", "abcdef123456")
    monkeypatch.setattr(_build_info, "BUILD_ID", "parity-620")
    transcript = Transcript(
        file_name="clip.wav",
        language="en",
        segments=[],
        meta=legacy_generation_meta(),
    )
    output = tmp_path / "clip.json"

    write_json(transcript, output)

    document = json.loads(output.read_text(encoding="utf-8"))
    receipt = document["meta"]["receipt"]
    assert receipt["git_commit"] == "abcdef123456"
    assert receipt["build_id"] == "parity-620"
    assert receipt["device"] == "cpu"
    assert receipt["compute_type"] == "int8"
    assert transcript.meta["receipt"] == receipt

    package_root = resources.files("transcription")
    receipt_schema = json.loads(
        package_root.joinpath("schemas/receipt-v1.schema.json").read_text(
            encoding="utf-8"
        )
    )
    transcript_schema = json.loads(
        package_root.joinpath("schemas/transcript-v2.schema.json").read_text(
            encoding="utf-8"
        )
    )
    assert not list(
        Draft7Validator(
            receipt_schema,
            format_checker=FormatChecker(),
        ).iter_errors(receipt)
    )
    assert not list(
        Draft7Validator(
            transcript_schema,
            format_checker=FormatChecker(),
        ).iter_errors(document)
    )


def test_write_json_leaves_non_generation_transcripts_unchanged(tmp_path: Path) -> None:
    transcript = Transcript(
        file_name="imported.wav",
        language="en",
        segments=[],
        meta={"source": "imported"},
    )
    output = tmp_path / "imported.json"

    write_json(transcript, output)

    document = json.loads(output.read_text(encoding="utf-8"))
    assert document["meta"] == {"source": "imported"}
    assert transcript.meta == {"source": "imported"}
