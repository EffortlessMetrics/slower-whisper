"""Tests for metadata helpers used across API and pipeline."""

from pathlib import Path

from slower_whisper.pipeline.meta_utils import build_generation_metadata
from slower_whisper.pipeline.models import Transcript


def test_build_generation_metadata_prefers_asr_runtime():
    """ASR-emitted device/compute_type should override configured defaults."""
    transcript = Transcript(
        file_name="clip.wav",
        language="en",
        segments=[],
        meta={
            "asr_backend": "dummy",
            "asr_device": "cpu",
            "asr_compute_type": "n/a",
        },
    )

    meta = build_generation_metadata(
        transcript,
        duration_sec=12.34,
        model_name="large-v3",
        config_device="cuda",
        config_compute_type="float16",
        beam_size=5,
        vad_min_silence_ms=500,
        language_hint=None,
        task="transcribe",
        pipeline_version="1.1.0",
        root=Path("/tmp/project"),
    )

    assert meta["device"] == "cpu"
    assert meta["compute_type"] == "n/a"
    assert meta["asr_backend"] == "dummy"


def test_build_generation_metadata_falls_back_to_runtime_candidates():
    """Runtime candidates should be used when ASR metadata is absent."""
    transcript = Transcript(file_name="clip.wav", language="en", segments=[], meta={})

    meta = build_generation_metadata(
        transcript,
        duration_sec=1.0,
        model_name="tiny",
        config_device="cuda",
        config_compute_type="float16",
        beam_size=5,
        vad_min_silence_ms=500,
        language_hint="en",
        task="transcribe",
        pipeline_version="1.1.0",
        root="/tmp/project",
        runtime_device_candidates=("cpu",),
        runtime_compute_candidates=("int8",),
    )

    assert meta["device"] == "cpu"
    assert meta["compute_type"] == "int8"
    assert meta["audio_file"] == "clip.wav"
