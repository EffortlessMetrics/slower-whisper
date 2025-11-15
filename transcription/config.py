from dataclasses import dataclass, field
from pathlib import Path


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
    compute_type: str = "float16"
    vad_min_silence_ms: int = 500
    beam_size: int = 5
    # Optional language and task; if language is None, auto-detect is used.
    language: str | None = None  # e.g. "en"
    task: str = "transcribe"  # or "translate"


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
