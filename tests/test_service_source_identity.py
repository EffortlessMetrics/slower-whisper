"""Public source identity must not expose temporary or caller path state."""

from __future__ import annotations

import pytest

from transcription.service_runtime_transcribe import _safe_audio_suffix
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


@pytest.mark.parametrize(
    "submitted",
    [
        f"{'x' * 400}.wav",
        f"clip.{'x' * 400}",
    ],
)
def test_safe_source_name_bounds_untrusted_response_identity(submitted: str) -> None:
    result = safe_source_name(
        submitted,
        fallback_suffix=".wav",
    )

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


def test_safe_audio_suffix_uses_normalized_source_identity() -> None:
    assert safe_source_name("voice.mp3 ") == "voice.mp3"
    assert _safe_audio_suffix("voice.mp3 ") == ".mp3"
    assert _safe_audio_suffix(r"..\folder\voice.MP3 ") == ".mp3"
    assert _safe_audio_suffix("voice.exe ") == ""
