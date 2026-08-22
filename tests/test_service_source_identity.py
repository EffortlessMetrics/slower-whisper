"""Public source identity must not expose temporary or caller path state."""

from __future__ import annotations

import pytest

from transcription.source_identity import safe_source_name


@pytest.mark.parametrize(
    ("submitted", "fallback_suffix", "expected"),
    [
        ("surface.wav", ".wav", "surface.wav"),
        ("../../surface.wav", ".wav", "surface.wav"),
        (r"..\..\surface.wav", ".wav", "surface.wav"),
        ("folder/subfolder/voice.flac", ".flac", "voice.flac"),
        ("voice\r\nforged.wav", ".wav", "voiceforged.wav"),
        ("my recording.wav", ".wav", "my_recording.wav"),
        ("surface", ".wav", "surface.wav"),
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
    assert safe_source_name(submitted, fallback_suffix=fallback_suffix) == expected


def test_safe_source_name_bounds_untrusted_response_identity() -> None:
    result = safe_source_name(
        f"{'x' * 400}.wav",
        fallback_suffix=".wav",
    )

    assert result.endswith(".wav")
    assert len(result) <= 255


def test_safe_source_name_does_not_preserve_path_or_control_characters() -> None:
    result = safe_source_name(
        "../../private\nfolder/meeting.wav",
        fallback_suffix=".wav",
    )

    assert result == "meeting.wav"
    assert "/" not in result
    assert "\\" not in result
    assert "\n" not in result
