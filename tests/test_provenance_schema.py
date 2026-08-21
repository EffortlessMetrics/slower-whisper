"""Bundled receipt and transcript schema acceptance."""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path

from jsonschema import Draft7Validator, FormatChecker

from transcription import _build_info
from transcription.meta_utils import build_generation_metadata
from transcription.models import Transcript
from transcription.writers import write_json


def load_schema(name: str) -> dict:
    resource = resources.files("transcription").joinpath("schemas", name)
    return json.loads(resource.read_text(encoding="utf-8"))


def test_automatic_receipt_and_transcript_validate_against_bundled_schemas(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(_build_info, "SOURCE_COMMIT", "abcdef123456")
    monkeypatch.setattr(_build_info, "BUILD_ID", "github-12345-1")
    transcript = Transcript(
        file_name="silence.wav",
        language="en",
        segments=[],
        meta={
            "asr_backend": "faster-whisper",
            "asr_model": "tiny",
            "asr_device": "cpu",
            "asr_compute_type": "int8",
            "asr_model_load_attempts": [
                {
                    "device": "cpu",
                    "compute_type": "int8",
                    "outcome": "selected",
                    "reason_code": "ok",
                }
            ],
        },
    )
    transcript.meta = build_generation_metadata(
        transcript,
        duration_sec=0.01,
        model_name="tiny",
        config_device="cpu",
        config_compute_type="int8",
        beam_size=5,
        vad_min_silence_ms=500,
        language_hint="en",
        task="transcribe",
        pipeline_version="2.0.2",
        root=tmp_path,
    )

    receipt_schema = load_schema("receipt-v1.schema.json")
    receipt_errors = sorted(
        Draft7Validator(
            receipt_schema,
            format_checker=FormatChecker(),
        ).iter_errors(transcript.meta["receipt"]),
        key=lambda error: list(error.path),
    )
    assert receipt_errors == []

    output = tmp_path / "transcript.json"
    write_json(transcript, output)
    document = json.loads(output.read_text(encoding="utf-8"))
    transcript_schema = load_schema("transcript-v2.schema.json")
    transcript_errors = sorted(
        Draft7Validator(
            transcript_schema,
            format_checker=FormatChecker(),
        ).iter_errors(document),
        key=lambda error: list(error.path),
    )
    assert transcript_errors == []
    assert document["meta"]["receipt"]["git_commit"] == "abcdef123456"
    assert document["meta"]["receipt"]["build_id"] == "github-12345-1"
