from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

WhisperTask = Literal["transcribe", "translate"]


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


# ============================================================================
# Public API Configuration Classes
# ============================================================================


@dataclass(slots=True)
class TranscriptionConfig:
    """
    High-level transcription configuration for public API and CLI.

    This is the stable, public-facing configuration for Stage 1 transcription.
    Internally, it can be converted to AsrConfig/AppConfig for backward
    compatibility with existing pipeline code.
    """

    # Whisper / faster-whisper settings
    model: str = "large-v3"
    device: str = "cuda"  # "cuda" or "cpu"
    compute_type: str = "float16"
    language: str | None = None  # None = auto-detect
    task: WhisperTask = "transcribe"

    # Behavior
    skip_existing_json: bool = True

    # Advanced options
    vad_min_silence_ms: int = 500
    beam_size: int = 5

    # v1.1+ diarization (L2) — opt-in
    enable_diarization: bool = False
    diarization_device: str = "auto"  # "cuda" | "cpu" | "auto"
    min_speakers: int | None = None
    max_speakers: int | None = None
    overlap_threshold: float = 0.3  # internal; not exposed in CLI yet

    # Internal field to track which config values were explicitly set
    # Used by CLI merge logic to implement correct precedence
    _source_fields: set[str] = field(default_factory=set, init=False, repr=False)

    @classmethod
    def from_file(cls, path: str | Path) -> TranscriptionConfig:
        """
        Load configuration from a JSON file.

        Only fields present in the JSON file are set. Missing fields remain
        at their default values, which allows proper config layering in the CLI.

        Args:
            path: Path to JSON configuration file.

        Returns:
            TranscriptionConfig instance with loaded settings. The returned
            instance includes a _source_fields attribute (set of field names
            that were explicitly loaded from the file).

        Raises:
            FileNotFoundError: If the configuration file doesn't exist.
            ValueError: If the JSON is invalid or contains invalid field values.
            json.JSONDecodeError: If the file contains malformed JSON.

        Example JSON file:
            {
                "model": "large-v3",
                "device": "cuda",
                "compute_type": "float16",
                "language": "en",
                "task": "transcribe",
                "skip_existing_json": true,
                "vad_min_silence_ms": 500,
                "beam_size": 5
            }
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(
                f"Invalid JSON in configuration file {path}: {e.msg}",
                e.doc,
                e.pos,
            ) from e

        if not isinstance(data, dict):
            raise ValueError(
                f"Configuration file must contain a JSON object, got {type(data).__name__}"
            )

        # Validate task if present
        if "task" in data and data["task"] not in ("transcribe", "translate"):
            raise ValueError(
                f"Invalid task value: {data['task']}. Must be 'transcribe' or 'translate'"
            )

        # Validate device if present
        if "device" in data and data["device"] not in ("cuda", "cpu"):
            raise ValueError(f"Invalid device value: {data['device']}. Must be 'cuda' or 'cpu'")

        # Validate numeric fields
        if "vad_min_silence_ms" in data:
            if not isinstance(data["vad_min_silence_ms"], int) or data["vad_min_silence_ms"] < 0:
                raise ValueError(
                    f"vad_min_silence_ms must be a non-negative integer, got {data['vad_min_silence_ms']}"
                )

        if "beam_size" in data:
            if not isinstance(data["beam_size"], int) or data["beam_size"] < 1:
                raise ValueError(f"beam_size must be a positive integer, got {data['beam_size']}")

        # Filter out unknown fields
        valid_fields = {
            "model",
            "device",
            "compute_type",
            "language",
            "task",
            "skip_existing_json",
            "vad_min_silence_ms",
            "beam_size",
            "enable_diarization",
            "diarization_device",
            "min_speakers",
            "max_speakers",
            "overlap_threshold",
        }
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}

        try:
            config = cls(**filtered_data)
            # Track which fields were explicitly set from file
            # This is used by _merge_configs to implement correct precedence
            config._source_fields = set(filtered_data.keys())
            return config
        except TypeError as e:
            raise ValueError(f"Invalid configuration data: {e}") from e

    @classmethod
    def from_env(cls, prefix: str = "SLOWER_WHISPER_") -> TranscriptionConfig:
        """
        Load configuration from environment variables.

        Environment variables are mapped to config fields by converting:
        - {prefix}MODEL -> model
        - {prefix}DEVICE -> device
        - {prefix}COMPUTE_TYPE -> compute_type
        - {prefix}LANGUAGE -> language
        - {prefix}TASK -> task
        - {prefix}SKIP_EXISTING_JSON -> skip_existing_json
        - {prefix}VAD_MIN_SILENCE_MS -> vad_min_silence_ms
        - {prefix}BEAM_SIZE -> beam_size

        Args:
            prefix: Environment variable prefix (default: "SLOWER_WHISPER_")

        Returns:
            TranscriptionConfig instance with values from environment.

        Raises:
            ValueError: If environment variables contain invalid values.

        Example:
            export SLOWER_WHISPER_MODEL=large-v3
            export SLOWER_WHISPER_DEVICE=cuda
            export SLOWER_WHISPER_LANGUAGE=en
            export SLOWER_WHISPER_TASK=transcribe
        """
        config_dict: dict[str, Any] = {}

        # String fields
        if model := os.getenv(f"{prefix}MODEL"):
            config_dict["model"] = model

        if device := os.getenv(f"{prefix}DEVICE"):
            if device not in ("cuda", "cpu"):
                raise ValueError(f"Invalid {prefix}DEVICE: {device}. Must be 'cuda' or 'cpu'")
            config_dict["device"] = device

        if compute_type := os.getenv(f"{prefix}COMPUTE_TYPE"):
            config_dict["compute_type"] = compute_type

        if language := os.getenv(f"{prefix}LANGUAGE"):
            config_dict["language"] = language if language.lower() != "none" else None

        if task := os.getenv(f"{prefix}TASK"):
            if task not in ("transcribe", "translate"):
                raise ValueError(
                    f"Invalid {prefix}TASK: {task}. Must be 'transcribe' or 'translate'"
                )
            config_dict["task"] = task

        # Boolean fields
        if skip_existing := os.getenv(f"{prefix}SKIP_EXISTING_JSON"):
            skip_existing_lower = skip_existing.lower()
            if skip_existing_lower in ("true", "1", "yes", "on"):
                config_dict["skip_existing_json"] = True
            elif skip_existing_lower in ("false", "0", "no", "off"):
                config_dict["skip_existing_json"] = False
            else:
                raise ValueError(
                    f"Invalid {prefix}SKIP_EXISTING_JSON: {skip_existing}. "
                    "Must be true/false, 1/0, yes/no, or on/off"
                )

        # Integer fields
        if vad_silence := os.getenv(f"{prefix}VAD_MIN_SILENCE_MS"):
            try:
                vad_value = int(vad_silence)
                if vad_value < 0:
                    raise ValueError(
                        f"{prefix}VAD_MIN_SILENCE_MS must be non-negative, got {vad_value}"
                    )
                config_dict["vad_min_silence_ms"] = vad_value
            except ValueError as e:
                raise ValueError(
                    f"Invalid {prefix}VAD_MIN_SILENCE_MS: {vad_silence}. Must be an integer."
                ) from e

        if beam_size := os.getenv(f"{prefix}BEAM_SIZE"):
            try:
                beam_value = int(beam_size)
                if beam_value < 1:
                    raise ValueError(f"{prefix}BEAM_SIZE must be positive, got {beam_value}")
                config_dict["beam_size"] = beam_value
            except ValueError as e:
                raise ValueError(
                    f"Invalid {prefix}BEAM_SIZE: {beam_size}. Must be a positive integer."
                ) from e

        config = cls(**config_dict)
        # Track which fields were explicitly set from environment
        # This is used by _merge_configs to implement correct precedence
        config._source_fields = set(config_dict.keys())
        return config


@dataclass(slots=True)
class EnrichmentConfig:
    """
    High-level enrichment configuration for public API and CLI.

    Controls prosody/emotion extraction and device selection for Stage 2.
    """

    # What to enrich
    skip_existing: bool = True
    enable_prosody: bool = True
    enable_emotion: bool = True
    enable_categorical_emotion: bool = False

    # Runtime
    device: str = "cpu"  # "cpu" or "cuda"

    # Optional model overrides
    dimensional_model_name: str | None = None
    categorical_model_name: str | None = None

    # Internal field to track which config values were explicitly set
    _source_fields: set[str] = field(default_factory=set, init=False, repr=False)

    @classmethod
    def from_file(cls, path: str | Path) -> EnrichmentConfig:
        """
        Load configuration from a JSON file.

        Args:
            path: Path to JSON configuration file.

        Returns:
            EnrichmentConfig instance with loaded settings.

        Raises:
            FileNotFoundError: If the configuration file doesn't exist.
            ValueError: If the JSON is invalid or contains invalid field values.
            json.JSONDecodeError: If the file contains malformed JSON.

        Example JSON file:
            {
                "skip_existing": true,
                "enable_prosody": true,
                "enable_emotion": true,
                "enable_categorical_emotion": false,
                "device": "cpu",
                "dimensional_model_name": null,
                "categorical_model_name": null
            }
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(
                f"Invalid JSON in configuration file {path}: {e.msg}",
                e.doc,
                e.pos,
            ) from e

        if not isinstance(data, dict):
            raise ValueError(
                f"Configuration file must contain a JSON object, got {type(data).__name__}"
            )

        # Validate device if present
        if "device" in data and data["device"] not in ("cuda", "cpu"):
            raise ValueError(f"Invalid device value: {data['device']}. Must be 'cuda' or 'cpu'")

        # Validate boolean fields
        bool_fields = [
            "skip_existing",
            "enable_prosody",
            "enable_emotion",
            "enable_categorical_emotion",
        ]
        for field_name in bool_fields:
            if field_name in data and not isinstance(data[field_name], bool):
                raise ValueError(
                    f"{field_name} must be a boolean, got {type(data[field_name]).__name__}"
                )

        # Filter out unknown fields
        valid_fields = {
            "skip_existing",
            "enable_prosody",
            "enable_emotion",
            "enable_categorical_emotion",
            "device",
            "dimensional_model_name",
            "categorical_model_name",
        }
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}

        try:
            config = cls(**filtered_data)
            # Track which fields were explicitly set from file
            config._source_fields = set(filtered_data.keys())
            return config
        except TypeError as e:
            raise ValueError(f"Invalid configuration data: {e}") from e

    @classmethod
    def from_env(cls, prefix: str = "SLOWER_WHISPER_ENRICH_") -> EnrichmentConfig:
        """
        Load configuration from environment variables.

        Environment variables are mapped to config fields by converting:
        - {prefix}SKIP_EXISTING -> skip_existing
        - {prefix}ENABLE_PROSODY -> enable_prosody
        - {prefix}ENABLE_EMOTION -> enable_emotion
        - {prefix}ENABLE_CATEGORICAL_EMOTION -> enable_categorical_emotion
        - {prefix}DEVICE -> device
        - {prefix}DIMENSIONAL_MODEL_NAME -> dimensional_model_name
        - {prefix}CATEGORICAL_MODEL_NAME -> categorical_model_name

        Args:
            prefix: Environment variable prefix (default: "SLOWER_WHISPER_ENRICH_")

        Returns:
            EnrichmentConfig instance with values from environment.

        Raises:
            ValueError: If environment variables contain invalid values.

        Example:
            export SLOWER_WHISPER_ENRICH_SKIP_EXISTING=true
            export SLOWER_WHISPER_ENRICH_ENABLE_PROSODY=true
            export SLOWER_WHISPER_ENRICH_ENABLE_EMOTION=true
            export SLOWER_WHISPER_ENRICH_DEVICE=cuda
        """
        config_dict: dict[str, Any] = {}

        # Boolean fields
        bool_mapping = {
            "SKIP_EXISTING": "skip_existing",
            "ENABLE_PROSODY": "enable_prosody",
            "ENABLE_EMOTION": "enable_emotion",
            "ENABLE_CATEGORICAL_EMOTION": "enable_categorical_emotion",
        }

        for env_suffix, field_name in bool_mapping.items():
            if value := os.getenv(f"{prefix}{env_suffix}"):
                value_lower = value.lower()
                if value_lower in ("true", "1", "yes", "on"):
                    config_dict[field_name] = True
                elif value_lower in ("false", "0", "no", "off"):
                    config_dict[field_name] = False
                else:
                    raise ValueError(
                        f"Invalid {prefix}{env_suffix}: {value}. "
                        "Must be true/false, 1/0, yes/no, or on/off"
                    )

        # Device field
        if device := os.getenv(f"{prefix}DEVICE"):
            if device not in ("cuda", "cpu"):
                raise ValueError(f"Invalid {prefix}DEVICE: {device}. Must be 'cuda' or 'cpu'")
            config_dict["device"] = device

        # String fields (model names)
        if dim_model := os.getenv(f"{prefix}DIMENSIONAL_MODEL_NAME"):
            config_dict["dimensional_model_name"] = (
                dim_model if dim_model.lower() != "none" else None
            )

        if cat_model := os.getenv(f"{prefix}CATEGORICAL_MODEL_NAME"):
            config_dict["categorical_model_name"] = (
                cat_model if cat_model.lower() != "none" else None
            )

        config = cls(**config_dict)
        # Track which fields were explicitly set from environment
        config._source_fields = set(config_dict.keys())
        return config
