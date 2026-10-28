"""Audio upload probing without loading an ASR model."""

import shutil
import subprocess
import wave
from pathlib import Path

import pytest
from fastapi import HTTPException

from transcription.service_validation import validate_audio_format


@pytest.mark.skipif(
    not shutil.which("ffmpeg") or not shutil.which("ffprobe"),
    reason="ffmpeg and ffprobe are required",
)
def test_probe_requires_audio_stream(tmp_path: Path) -> None:
    video = tmp_path / "video.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=16x16:d=1",
            "-c:v",
            "mpeg4",
            str(video),
        ],
        check=True,
    )
    with pytest.raises(HTTPException) as exc:
        validate_audio_format(video)
    assert exc.value.status_code == 400
    assert "no audio stream" in exc.value.detail

    audio = tmp_path / "audio.wav"
    with wave.open(str(audio), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\0\0" * 1600)
    validate_audio_format(audio)

    combined = tmp_path / "video-with-audio.mkv"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(video),
            "-i",
            str(audio),
            "-c",
            "copy",
            str(combined),
        ],
        check=True,
    )
    validate_audio_format(combined)
