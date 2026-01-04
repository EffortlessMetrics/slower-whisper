"""
Pytest configuration and fixtures for tests.

This module provides:
- Mock objects for optional dependencies
- Common test fixtures
- Test configuration
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from pytest import Item

# Ensure the project package is importable when running pytest as an installed script
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================================
# Mock unavailable dependencies to allow tests to run
# ============================================================================

# Default to auto mode; individual tests can opt into stub/missing.
os.environ.setdefault("SLOWER_WHISPER_PYANNOTE_MODE", "auto")


def mock_module(module_name: str, attrs: dict[str, Any] | None = None) -> MagicMock:
    """Create a mock module with optional attributes."""
    mock = MagicMock()
    if attrs:
        for attr_name, attr_value in attrs.items():
            setattr(mock, attr_name, attr_value)
    return mock


# Mock faster_whisper if not available
try:
    import faster_whisper  # noqa: F401
except Exception:
    sys.modules["faster_whisper"] = mock_module("faster_whisper", {"WhisperModel": MagicMock})


# Mock transformers if not available (for emotion tests)
try:
    import transformers  # noqa: F401
except Exception:
    sys.modules["transformers"] = mock_module(
        "transformers",
        {
            "AutoModelForAudioClassification": MagicMock,
            "Wav2Vec2Processor": MagicMock,
            "pipeline": MagicMock,
        },
    )


# Mock torch if not available (for emotion tests)
try:
    import torch  # noqa: F401
except Exception:
    sys.modules["torch"] = mock_module(
        "torch",
        {
            "cuda": mock_module("torch.cuda", {"is_available": lambda: False}),
            "no_grad": lambda: MagicMock(__enter__=lambda x: None, __exit__=lambda *args: None),
            "nn": mock_module(
                "torch.nn",
                {"functional": mock_module("torch.nn.functional", {"softmax": lambda x, dim: x})},
            ),
        },
    )


# Mock parselmouth if not available (for prosody tests)
try:
    import parselmouth  # noqa: F401
except Exception:
    sys.modules["parselmouth"] = mock_module(
        "parselmouth",
        {
            "Sound": MagicMock,
            "praat": mock_module("parselmouth.praat", {"call": MagicMock(return_value=None)}),
        },
    )


# Mock librosa if not available (for prosody tests)
try:
    import librosa  # noqa: F401
except Exception:
    sys.modules["librosa"] = mock_module(
        "librosa",
        {
            # Return near-silence energy so prosody tests treat audio as very quiet
            "feature": mock_module("librosa.feature", {"rms": MagicMock(return_value=[[1e-6]])}),
            "frames_to_time": lambda frames, sr, hop_length: frames * hop_length / sr,
        },
    )


# Mock soundfile if not available
SOUNDFILE_AVAILABLE = False
try:
    import soundfile  # noqa: F401

    SOUNDFILE_AVAILABLE = True
except Exception:
    import wave

    import numpy as np

    class MockSoundFile:
        def __init__(self, path: str, mode: str = "r") -> None:
            self._wf = wave.open(path, mode)
            self.samplerate: int = self._wf.getframerate()
            self.channels: int = self._wf.getnchannels()

        def __len__(self) -> int:
            # cast needed: wave.Wave_read.getnframes() returns Any in typeshed
            return int(self._wf.getnframes())

        def __enter__(self) -> MockSoundFile:
            return self

        def __exit__(self, *args: Any) -> None:
            self._wf.close()

        def seek(self, frame: int) -> None:
            self._wf.setpos(frame)

        def read(self, frames: int, dtype: str = "float32") -> np.ndarray:
            raw = self._wf.readframes(frames)
            data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
            return data

    def mock_read(path: str, dtype: str = "float32") -> tuple[np.ndarray, int]:
        with wave.open(path, "rb") as wf:
            raw = wf.readframes(wf.getnframes())
            data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
            return (data.astype(dtype), wf.getframerate())

    def mock_write(path: str | Path, data: Any, sr: int) -> None:
        pcm = (np.array(data, dtype=np.float32) * 32767).astype("<i2")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(pcm.tobytes())

    sys.modules["soundfile"] = mock_module(
        "soundfile", {"SoundFile": MockSoundFile, "read": mock_read, "write": mock_write}
    )


# Make sure numpy is available (it's usually in the base system)
try:
    import numpy  # noqa: F401
except ImportError:
    # If numpy is not available, we can't really run tests
    raise ImportError(
        "numpy is required for tests. Please install with: pip install numpy"
    ) from None


# Skip heavy diarization tests when pyannote.audio isn't available
PYANNOTE_AVAILABLE = True
if PYANNOTE_AVAILABLE:
    try:
        import pyannote.audio  # noqa: F401
    except Exception:
        sys.modules["pyannote"] = mock_module("pyannote")
        sys.modules["pyannote.audio"] = mock_module("pyannote.audio", {"Pipeline": MagicMock})


def pytest_runtest_setup(item: Item) -> None:
    """Skip tests based on marker requirements and dependency availability."""
    if item.get_closest_marker("requires_diarization") and not PYANNOTE_AVAILABLE:
        pytest.skip("pyannote.audio not available; skipping diarization tests")
