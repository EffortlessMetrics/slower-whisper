"""Regression tests for bounded ASR VAD capability negotiation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from transcription.asr_engine import TranscriptionEngine
from transcription.config import AsrConfig
from transcription.exceptions import ASRInferenceError


class RepeatingVadTypeErrorModel:
    """Backend that keeps blaming a VAD kwarg after it has been removed."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def transcribe(self, _audio_path: str, **kwargs: Any) -> Any:
        self.calls.append(dict(kwargs))
        raise TypeError("backend rejected vad_filter during negotiation")


def test_vad_negotiation_stops_when_retry_makes_no_progress(tmp_path: Path) -> None:
    audio_path = tmp_path / "clip.wav"
    audio_path.write_bytes(b"injected backend does not read audio")
    model = RepeatingVadTypeErrorModel()
    engine = TranscriptionEngine(
        AsrConfig(model_name="tiny", device="cpu", compute_type="int8"),
        backend_factory=lambda *_args: model,
    )

    with pytest.raises(ASRInferenceError) as raised:
        engine.transcribe_file(audio_path)

    assert raised.value.context == {"phase": "vad_negotiation"}
    assert isinstance(raised.value.__cause__, TypeError)
    assert len(model.calls) == 2
    assert "vad_filter" in model.calls[0]
    assert "vad_parameters" in model.calls[0]
    assert "vad_filter" not in model.calls[1]
    assert "vad_parameters" in model.calls[1]
