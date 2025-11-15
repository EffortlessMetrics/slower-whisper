"""Configuration for speaker diarization."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class DiarizationConfig:
    """
    Configuration for speaker diarization.

    Attributes:
        model_name: Pyannote model name (e.g., "pyannote/speaker-diarization-3.1")
        device: Device to use ("cuda" or "cpu")
        hf_token: HuggingFace token for downloading models (if None, checks env)
        num_speakers: Fixed number of speakers (None for automatic detection)
        min_speakers: Minimum number of speakers (for automatic detection)
        max_speakers: Maximum number of speakers (for automatic detection)
    """
    model_name: str = "pyannote/speaker-diarization-3.1"
    device: str = "cuda"
    hf_token: Optional[str] = None
    num_speakers: Optional[int] = None
    min_speakers: int = 1
    max_speakers: int = 10
