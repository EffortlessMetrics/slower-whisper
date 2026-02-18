"""Legacy internal configuration dataclasses.

This module contains the legacy "pipeline-facing" configuration dataclasses
that were originally part of the monolithic config.py. These are kept for
backward compatibility with existing pipeline code.

Classes:
    Paths: Resolves filesystem locations used by the pipeline.
    AsrConfig: Configuration for the ASR engine (faster-whisper).
    AppConfig: Top-level application configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .config_validation import auto_derive_compute_type, validate_model_name


@dataclass
class Paths:
    """
    Resolves all filesystem locations used by the pipeline from a single root.

    By default, the root is the current working directory. Subdirectories
    are derived properties to avoid mutable default pitfalls.
    """

    root: Path = Path()

    @property
    def raw_dir(self) -> Path:
        return self.root / "raw_audio"

    @property
    def norm_dir(self) -> Path:
        return self.root / "input_audio"

    @property
    def transcripts_dir(self) -> Path:
        return self.root / "transcripts"

    @property
    def json_dir(self) -> Path:
        return self.root / "whisper_json"


@dataclass
class AsrConfig:
    """
    Configuration for the ASR engine (faster-whisper).
    """

    model_name: str = "large-v3"
    device: str = "cuda"
    # None means "auto-select based on device" (float16 for CUDA, int8 for CPU)
    compute_type: str | None = None
    vad_min_silence_ms: int = 500
    beam_size: int = 5
    # Optional language and task; if language is None, auto-detect is used.
    language: str | None = None  # e.g. "en"
    task: str = "transcribe"  # or "translate"
    # Word-level timestamps (v1.8+)
    word_timestamps: bool = False

    def __post_init__(self):
        """Validate model name and auto-detect compute_type based on device if using default."""
        validate_model_name(self.model_name)
        self.compute_type = auto_derive_compute_type(self.device, self.compute_type)


@dataclass
class AppConfig:
    """
    Top-level application configuration.

    Attributes:
        paths: Filesystem paths and directory layout.
        asr: ASR engine configuration.
        skip_existing_json: If True, skip transcription for files that
            already have a JSON output.
    """

    paths: Paths = field(default_factory=Paths)
    asr: AsrConfig = field(default_factory=AsrConfig)
    skip_existing_json: bool = False
