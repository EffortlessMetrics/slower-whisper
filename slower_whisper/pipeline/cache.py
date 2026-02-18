"""Cache management for slower-whisper model downloads.

This module provides centralized control over where all heavy model weights
are downloaded and cached. By default, all models live under:

    $SLOWER_WHISPER_CACHE_ROOT (default: ~/.cache/slower-whisper)

This ensures:
1. One-time downloads of multi-GB models (Whisper, pyannote, emotion)
2. Reuse across virtualenvs and runs
3. Explicit control over cache location and cleanup

Environment variables respected:
- SLOWER_WHISPER_CACHE_ROOT: Root cache directory
- HF_HOME: Hugging Face cache (defaults to $ROOT/hf)
- TORCH_HOME: PyTorch cache (defaults to $ROOT/torch)
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_ROOT = Path.home() / ".cache" / "slower-whisper"

# Guard to prevent duplicate environment variable configuration
_CACHE_ENV_CONFIGURED = False


@dataclass
class CachePaths:
    """Centralized cache paths for all slower-whisper models and artifacts.

    Attributes:
        root: Root cache directory for all slower-whisper data
        hf_home: Hugging Face hub cache (HF_HOME)
        torch_home: PyTorch cache (TORCH_HOME)
        whisper_root: Whisper model weights (faster-whisper download_root)
        emotion_root: Emotion recognition models (transformers cache_dir)
        diarization_root: Pyannote diarization models (pipeline cache_dir)
        samples_root: Sample datasets for testing (synthetic + mini datasets)
        benchmarks_root: Benchmark evaluation datasets (AMI, IEMOCAP, LibriCSS, etc.)
    """

    root: Path
    hf_home: Path
    torch_home: Path
    whisper_root: Path
    emotion_root: Path
    diarization_root: Path
    samples_root: Path
    benchmarks_root: Path

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> CachePaths:
        """Construct cache paths from environment variables.

        Args:
            env: Environment dict (defaults to os.environ)

        Returns:
            CachePaths with resolved paths
        """
        env = env or os.environ
        root = Path(env.get("SLOWER_WHISPER_CACHE_ROOT", DEFAULT_ROOT)).expanduser()

        # Respect HF_HOME / TORCH_HOME if user set them explicitly
        hf_home = Path(env.get("HF_HOME", root / "hf")).expanduser()
        torch_home = Path(env.get("TORCH_HOME", root / "torch")).expanduser()

        # Model-specific subdirectories
        whisper_root = root / "whisper"
        emotion_root = root / "emotion"
        diarization_root = root / "diarization"

        # Dataset directories
        samples_root = root / "samples"
        benchmarks_root = root / "benchmarks"

        return cls(
            root=root,
            hf_home=hf_home,
            torch_home=torch_home,
            whisper_root=whisper_root,
            emotion_root=emotion_root,
            diarization_root=diarization_root,
            samples_root=samples_root,
            benchmarks_root=benchmarks_root,
        )

    def ensure_dirs(self) -> CachePaths:
        """Create all cache directories if they don't exist.

        Returns:
            Self for method chaining
        """
        for path in [
            self.root,
            self.hf_home,
            self.torch_home,
            self.whisper_root,
            self.emotion_root,
            self.diarization_root,
            self.samples_root,
            self.benchmarks_root,
        ]:
            path.mkdir(parents=True, exist_ok=True)
        return self


def configure_global_cache_env(env: MutableMapping[str, str] | None = None) -> CachePaths:
    """Configure global cache environment variables for all model downloads.

    This function:
    1. Resolves cache paths from environment
    2. Creates cache directories if missing
    3. Sets HF_HOME, TORCH_HOME, HF_HUB_CACHE if not already set

    Call this once at startup (e.g., in transcription/__init__.py or CLI main)
    to ensure all model downloads respect slower-whisper's cache layout.

    Subsequent calls are safe (idempotent) but skip environment variable setting.

    Args:
        env: Environment dict (defaults to os.environ)

    Returns:
        Configured CachePaths

    Example:
        >>> from transcription.cache import configure_global_cache_env
        >>> paths = configure_global_cache_env()
        >>> # Now all HF / torch model loads will use these paths
    """
    global _CACHE_ENV_CONFIGURED

    if env is None:
        env = os.environ

    paths = CachePaths.from_env(env).ensure_dirs()

    # Only configure environment once to avoid overwrites
    if not _CACHE_ENV_CONFIGURED:
        # Only set if not already set, so users can override
        env.setdefault("HF_HOME", str(paths.hf_home))
        env.setdefault("TORCH_HOME", str(paths.torch_home))
        env.setdefault("HF_HUB_CACHE", str(paths.hf_home / "hub"))
        _CACHE_ENV_CONFIGURED = True

    return paths
