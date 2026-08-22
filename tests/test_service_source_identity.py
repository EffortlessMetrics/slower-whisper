"""Public source identity must not expose the service's temporary path."""

from __future__ import annotations

import pytest

from transcription.service_runtime_transcribe import _safe_source_name


@pytest.mark.parametrize(
    ("submitted", "fallback_suffix", "expected"),
    [
        ("surface.wav", ".wav", "surface.wav"),
        ("../../surface.wav", ".wav", "surface.wav"),
        (r"..\..\surface.wav", ".wav", "surface.wav"),
        ("folder/subfolder/voice.flac", ".flac", "voice.flac"),
        ("voice\r\nforged.wav", ".wav", "voiceforged.wav"),
        (None, ".wav", "audio.wav"),
        ("", ".wav", "audio.wav"),
        ("..", ".wav", "audio.wav"),
    ],
)
def test_safe_source_name_is_a_bounded_basename(
    submitted: str | None,
    fallback_suffix: str,
    expected: str,
) -> None:
    assert (
        _safe_source_name(submitted, fallback_suffix=fallback_suffix)
        == expected
    )


def test_safe_source_name_bounds_untrusted_response_identity() -> None:
    result = _safe_source_name(
        f"{'x' * 400}.wav",
        fallback_suffix=".wav",
    )

    assert result.endswith(".wav")
    assert len(result) <= 255
