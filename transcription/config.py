"""Configuration management for transcription and enrichment pipelines.

This module defines the configuration dataclasses used throughout slower-whisper:

- TranscriptionConfig: Public API for Stage 1 (ASR) settings
- EnrichmentConfig: Public API for Stage 2 (audio enrichment) settings
- AppConfig, AsrConfig, Paths: Legacy internal configs for backward compatibility

Configuration can be created programmatically or loaded from JSON files.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from .exceptions import ConfigurationError

logger = logging.getLogger(__name__)

WhisperTask = Literal["transcribe", "translate"]
ALLOWED_COMPUTE_TYPES: set[str] = {
    "float16",
    "float32",
    "int16",
    "int8",
    "int8_float16",
    "int8_float32",
}
ALLOWED_WHISPER_MODELS: set[str] = {
    "tiny",
    "base",
    "small",
    "medium",
    "large-v1",
    "large-v2",
    "large-v3",
    "large-v3-turbo",
}


def validate_model_name(model_name: str) -> str:
    """
    Validate that model_name is a supported Whisper model.

    Args:
        model_name: Requested Whisper model name.

    Returns:
        The validated model name (unchanged).

    Raises:
        ConfigurationError: If model_name is not one of the allowed values.
    """
    if model_name not in ALLOWED_WHISPER_MODELS:
        allowed = ", ".join(sorted(ALLOWED_WHISPER_MODELS))
        raise ConfigurationError(f"Invalid model name '{model_name}'. Must be one of: {allowed}")

    return model_name


def validate_compute_type(compute_type: str | None) -> str | None:
    """
    Validate that compute_type is supported by faster-whisper.

    Args:
        compute_type: Requested compute type (case-insensitive) or None for auto.

    Returns:
        Normalized compute type (lowercase) or None if not provided.

    Raises:
        ConfigurationError: If compute_type is not one of the allowed values.
    """
    if compute_type is None:
        return None

    normalized = compute_type.lower()
    if normalized not in ALLOWED_COMPUTE_TYPES:
        allowed = ", ".join(sorted(ALLOWED_COMPUTE_TYPES))
        raise ConfigurationError(
            f"Invalid compute_type '{compute_type}'. Must be one of: {allowed}"
        )

    return normalized


def auto_derive_compute_type(device: str, compute_type: str | None) -> str:
    """
    Auto-derive compute_type based on device if not explicitly provided.

    If compute_type is None, selects sensible defaults:
    - "int8" for CPU (optimal for CPU inference)
    - "float16" for CUDA (uses Tensor Cores on modern GPUs)

    Args:
        device: Target device ("cpu" or "cuda").
        compute_type: Requested compute type or None for auto-selection.

    Returns:
        Normalized compute type (lowercase, validated).

    Raises:
        ConfigurationError: If compute_type is not supported by faster-whisper.
    """
    if compute_type is None:
        # Use sensible defaults: int8 for CPU, float16 for CUDA
        compute_type = "int8" if device == "cpu" else "float16"

    # Validate and normalize
    return validate_compute_type(compute_type) or compute_type


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
    # None means "auto-select based on device" (float16 for CUDA, int8 for CPU)
    compute_type: str | None = None
    language: str | None = None  # None = auto-detect
    task: WhisperTask = "transcribe"

    # Behavior
    skip_existing_json: bool = True

    # Advanced options
    vad_min_silence_ms: int = 500
    beam_size: int = 5

    # Chunking (v1.3 preview)
    enable_chunking: bool = False
    chunk_target_duration_s: float = 30.0
    chunk_max_duration_s: float = 45.0
    chunk_target_tokens: int = 400
    chunk_pause_split_threshold_s: float = 1.5

    def __post_init__(self):
        """Validate model name and auto-detect compute_type based on device if using default."""
        # Validate model name
        validate_model_name(self.model)

        # Auto-detect and validate compute_type
        validated_compute_type = auto_derive_compute_type(self.device, self.compute_type)
        object.__setattr__(self, "compute_type", validated_compute_type)

        if self.chunk_target_duration_s <= 0 or self.chunk_max_duration_s <= 0:
            raise ConfigurationError("Chunk durations must be positive")
        if self.chunk_max_duration_s < self.chunk_target_duration_s:
            object.__setattr__(self, "chunk_max_duration_s", self.chunk_target_duration_s)
        if self.chunk_target_tokens <= 0:
            raise ConfigurationError("chunk_target_tokens must be positive")
        if self.chunk_pause_split_threshold_s < 0:
            raise ConfigurationError("chunk_pause_split_threshold_s cannot be negative")

    # v1.1+ diarization (L2) — opt-in
    enable_diarization: bool = False
    diarization_device: str = "auto"  # "cuda" | "cpu" | "auto"
    min_speakers: int | None = None
    max_speakers: int | None = None
    overlap_threshold: float = 0.3  # exposed via --overlap-threshold

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

        # Validate model if present
        if "model" in data:
            try:
                validate_model_name(data["model"])
            except ConfigurationError as e:
                raise ValueError(str(e)) from e

        # Validate task if present
        if "task" in data and data["task"] not in ("transcribe", "translate"):
            raise ValueError(
                f"Invalid task value: {data['task']}. Must be 'transcribe' or 'translate'"
            )

        # Validate device if present
        if "device" in data and data["device"] not in ("cuda", "cpu"):
            raise ValueError(f"Invalid device value: {data['device']}. Must be 'cuda' or 'cpu'")

        # Validate diarization_device if present
        if "diarization_device" in data:
            if data["diarization_device"] not in ("cuda", "cpu", "auto"):
                raise ValueError(
                    f"Invalid diarization_device value: {data['diarization_device']}. "
                    "Must be 'cuda', 'cpu', or 'auto'"
                )

        # Validate boolean fields
        bool_fields = [
            "skip_existing_json",
            "enable_diarization",
        ]
        for field_name in bool_fields:
            if field_name in data and not isinstance(data[field_name], bool):
                raise ValueError(
                    f"{field_name} must be a boolean, got {type(data[field_name]).__name__}"
                )

        # Validate numeric fields
        if "vad_min_silence_ms" in data:
            vad_value = data["vad_min_silence_ms"]
            if isinstance(vad_value, bool) or not isinstance(vad_value, int) or vad_value < 0:
                raise ValueError(
                    f"vad_min_silence_ms must be a non-negative integer, got {data['vad_min_silence_ms']}"
                )

        if "beam_size" in data:
            beam_value = data["beam_size"]
            if isinstance(beam_value, bool) or not isinstance(beam_value, int) or beam_value < 1:
                raise ValueError(f"beam_size must be a positive integer, got {data['beam_size']}")

        # Validate compute_type (case-insensitive)
        compute_type_value = data.get("compute_type")
        try:
            normalized_compute_type = validate_compute_type(compute_type_value)
        except ConfigurationError as e:
            raise ValueError(str(e)) from e
        if normalized_compute_type is not None:
            data["compute_type"] = normalized_compute_type

        # Validate diarization fields (type and range)
        min_speakers = data.get("min_speakers")
        max_speakers = data.get("max_speakers")
        overlap_threshold = data.get("overlap_threshold")

        try:
            validate_diarization_settings(min_speakers, max_speakers, overlap_threshold)
        except ConfigurationError as e:
            # Surface config file errors as ValueError for consistency with other validations
            raise ValueError(str(e)) from e

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
            "enable_chunking",
            "chunk_target_duration_s",
            "chunk_max_duration_s",
            "chunk_target_tokens",
            "chunk_pause_split_threshold_s",
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
        - {prefix}ENABLE_DIARIZATION -> enable_diarization
        - {prefix}DIARIZATION_DEVICE -> diarization_device
        - {prefix}MIN_SPEAKERS -> min_speakers
        - {prefix}MAX_SPEAKERS -> max_speakers
        - {prefix}OVERLAP_THRESHOLD -> overlap_threshold
        - {prefix}ENABLE_CHUNKING -> enable_chunking
        - {prefix}CHUNK_TARGET_DURATION_S -> chunk_target_duration_s
        - {prefix}CHUNK_MAX_DURATION_S -> chunk_max_duration_s
        - {prefix}CHUNK_TARGET_TOKENS -> chunk_target_tokens
        - {prefix}CHUNK_PAUSE_SPLIT_THRESHOLD_S -> chunk_pause_split_threshold_s

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
            export SLOWER_WHISPER_ENABLE_DIARIZATION=true
            export SLOWER_WHISPER_DIARIZATION_DEVICE=cpu
            export SLOWER_WHISPER_MIN_SPEAKERS=2
            export SLOWER_WHISPER_MAX_SPEAKERS=4
            export SLOWER_WHISPER_OVERLAP_THRESHOLD=0.3
        """
        config_dict: dict[str, Any] = {}

        # String fields
        if model := os.getenv(f"{prefix}MODEL"):
            try:
                validate_model_name(model)
            except ConfigurationError as e:
                raise ValueError(str(e)) from e
            config_dict["model"] = model

        if device := os.getenv(f"{prefix}DEVICE"):
            if device not in ("cuda", "cpu"):
                raise ValueError(f"Invalid {prefix}DEVICE: {device}. Must be 'cuda' or 'cpu'")
            config_dict["device"] = device

        if compute_type := os.getenv(f"{prefix}COMPUTE_TYPE"):
            try:
                normalized_compute_type = validate_compute_type(compute_type)
            except ConfigurationError as e:
                raise ValueError(str(e)) from e
            config_dict["compute_type"] = normalized_compute_type

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
        if enable_chunking := os.getenv(f"{prefix}ENABLE_CHUNKING"):
            enable_lower = enable_chunking.lower()
            if enable_lower in ("true", "1", "yes", "on"):
                config_dict["enable_chunking"] = True
            elif enable_lower in ("false", "0", "no", "off"):
                config_dict["enable_chunking"] = False
            else:
                raise ValueError(
                    f"Invalid {prefix}ENABLE_CHUNKING: {enable_chunking}. "
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
        if chunk_tokens := os.getenv(f"{prefix}CHUNK_TARGET_TOKENS"):
            try:
                token_value = int(chunk_tokens)
                if token_value <= 0:
                    raise ValueError(
                        f"{prefix}CHUNK_TARGET_TOKENS must be positive, got {token_value}"
                    )
                config_dict["chunk_target_tokens"] = token_value
            except ValueError as e:
                raise ValueError(
                    f"Invalid {prefix}CHUNK_TARGET_TOKENS: {chunk_tokens}. Must be a positive integer."
                ) from e

        # Diarization fields
        if enable_diar := os.getenv(f"{prefix}ENABLE_DIARIZATION"):
            enable_lower = enable_diar.lower()
            if enable_lower in ("true", "1", "yes", "on"):
                config_dict["enable_diarization"] = True
            elif enable_lower in ("false", "0", "no", "off"):
                config_dict["enable_diarization"] = False
            else:
                raise ValueError(
                    f"Invalid {prefix}ENABLE_DIARIZATION: {enable_diar}. "
                    "Must be true/false, 1/0, yes/no, or on/off"
                )

        if diar_device := os.getenv(f"{prefix}DIARIZATION_DEVICE"):
            if diar_device not in ("cuda", "cpu", "auto"):
                raise ValueError(
                    f"Invalid {prefix}DIARIZATION_DEVICE: {diar_device}. "
                    "Must be 'cuda', 'cpu', or 'auto'"
                )
            config_dict["diarization_device"] = diar_device

        if min_speakers := os.getenv(f"{prefix}MIN_SPEAKERS"):
            try:
                if min_speakers.lower() in ("none", "null", ""):
                    config_dict["min_speakers"] = None
                else:
                    min_value = int(min_speakers)
                    if min_value < 1:
                        raise ValueError(
                            f"{prefix}MIN_SPEAKERS must be a positive integer, got {min_value}"
                        )
                    config_dict["min_speakers"] = min_value
            except ValueError as e:
                raise ValueError(
                    f"Invalid {prefix}MIN_SPEAKERS: {min_speakers}. Must be a positive integer."
                ) from e

        if max_speakers := os.getenv(f"{prefix}MAX_SPEAKERS"):
            try:
                if max_speakers.lower() in ("none", "null", ""):
                    config_dict["max_speakers"] = None
                else:
                    max_value = int(max_speakers)
                    if max_value < 1:
                        raise ValueError(
                            f"{prefix}MAX_SPEAKERS must be a positive integer, got {max_value}"
                        )
                    config_dict["max_speakers"] = max_value
            except ValueError as e:
                raise ValueError(
                    f"Invalid {prefix}MAX_SPEAKERS: {max_speakers}. Must be a positive integer."
                ) from e

        if overlap := os.getenv(f"{prefix}OVERLAP_THRESHOLD"):
            try:
                overlap_value = float(overlap)
                if not 0.0 <= overlap_value <= 1.0:
                    raise ValueError(
                        f"{prefix}OVERLAP_THRESHOLD must be between 0.0 and 1.0, got {overlap_value}"
                    )
                config_dict["overlap_threshold"] = overlap_value
            except ValueError as e:
                raise ValueError(
                    f"Invalid {prefix}OVERLAP_THRESHOLD: {overlap}. "
                    "Must be a floating point value between 0.0 and 1.0"
                ) from e
        if chunk_target := os.getenv(f"{prefix}CHUNK_TARGET_DURATION_S"):
            try:
                target_value = float(chunk_target)
                if target_value <= 0:
                    raise ValueError(
                        f"{prefix}CHUNK_TARGET_DURATION_S must be positive, got {target_value}"
                    )
                config_dict["chunk_target_duration_s"] = target_value
            except ValueError as e:
                raise ValueError(
                    f"Invalid {prefix}CHUNK_TARGET_DURATION_S: {chunk_target}. Must be a positive float."
                ) from e

        if chunk_max := os.getenv(f"{prefix}CHUNK_MAX_DURATION_S"):
            try:
                chunk_max_val = float(chunk_max)
                if chunk_max_val <= 0:
                    raise ValueError(
                        f"{prefix}CHUNK_MAX_DURATION_S must be positive, got {chunk_max_val}"
                    )
                config_dict["chunk_max_duration_s"] = chunk_max_val
            except ValueError as e:
                raise ValueError(
                    f"Invalid {prefix}CHUNK_MAX_DURATION_S: {chunk_max}. Must be a positive float."
                ) from e

        if pause_thresh := os.getenv(f"{prefix}CHUNK_PAUSE_SPLIT_THRESHOLD_S"):
            try:
                pause_value = float(pause_thresh)
                if pause_value < 0:
                    raise ValueError(
                        f"{prefix}CHUNK_PAUSE_SPLIT_THRESHOLD_S cannot be negative, got {pause_value}"
                    )
                config_dict["chunk_pause_split_threshold_s"] = pause_value
            except ValueError as e:
                raise ValueError(
                    f"Invalid {prefix}CHUNK_PAUSE_SPLIT_THRESHOLD_S: {pause_thresh}. Must be a float."
                ) from e

        config = cls(**config_dict)
        # Track which fields were explicitly set from environment
        # This is used by _merge_configs to implement correct precedence
        config._source_fields = set(config_dict.keys())
        return config

    @classmethod
    def from_sources(
        cls,
        env_prefix: str = "SLOWER_WHISPER_",
        config_file: str | Path | None = None,
        **overrides: Any,
    ) -> TranscriptionConfig:
        """
        Load configuration from multiple sources with proper precedence.

        Precedence order (highest to lowest):
        1. Explicit keyword arguments (overrides parameter)
        2. Config file (if config_file provided)
        3. Environment variables (with env_prefix)
        4. Defaults

        This method implements the same precedence logic as the CLI, making it
        available for programmatic use without needing to go through argparse.

        The compute_type field has special handling: if not explicitly set by any source,
        it defaults based on the final device value ("int8" for CPU, "float16" for CUDA).

        Args:
            env_prefix: Environment variable prefix (default: "SLOWER_WHISPER_").
                Used to look up env vars like {prefix}MODEL, {prefix}DEVICE, etc.
            config_file: Optional path to JSON configuration file.
            **overrides: Explicit configuration values that override all other sources.
                Only non-None values are applied. All TranscriptionConfig fields are
                supported (model, device, compute_type, language, task, etc.).

        Returns:
            TranscriptionConfig with merged settings from all sources.

        Raises:
            FileNotFoundError: If config_file is specified but doesn't exist.
            ValueError: If config file or environment variables contain invalid values.
            ConfigurationError: If configuration values fail validation.

        Examples:
            # Use defaults
            config = TranscriptionConfig.from_sources()

            # Load from environment variables only
            config = TranscriptionConfig.from_sources()

            # Load from config file and environment
            config = TranscriptionConfig.from_sources(
                config_file=Path("config.json")
            )

            # Override specific values
            config = TranscriptionConfig.from_sources(
                config_file=Path("config.json"),
                device="cpu",
                model="base"
            )

            # Custom environment prefix
            config = TranscriptionConfig.from_sources(
                env_prefix="MY_APP_",
                model="medium"
            )

            # Full precedence example
            # 1. Defaults: device="cuda", compute_type=None -> "float16"
            # 2. Env: SLOWER_WHISPER_DEVICE=cpu -> device="cpu", compute_type="int8"
            # 3. File: {"model": "base"} -> model="base", device="cpu", compute_type="int8"
            # 4. CLI: device="cuda" -> model="base", device="cuda", compute_type="float16"
            config = TranscriptionConfig.from_sources(
                config_file="config.json",
                device="cuda"  # Overrides env and file
            )
        """
        # Step 1: Start with defaults
        config = cls()

        # Step 2: Override with environment variables
        env_config = cls.from_env(prefix=env_prefix)
        env_compute_type_explicit = "compute_type" in getattr(env_config, "_source_fields", set())
        config = _merge_configs(config, env_config)

        # Step 3: Override with config file if provided
        file_compute_type_explicit = False
        if config_file is not None:
            file_config = cls.from_file(config_file)
            file_compute_type_explicit = "compute_type" in getattr(
                file_config,
                "_source_fields",
                set(),
            )
            config = _merge_configs(config, file_config)

        # Step 4: Override with explicit keyword arguments (only if not None)
        # Filter out None values to distinguish between "not set" and "set to None"
        filtered_overrides = {k: v for k, v in overrides.items() if v is not None}

        if filtered_overrides:
            override_compute_type_explicit = "compute_type" in filtered_overrides

            # Create a new config with overrides applied
            override_config = cls(
                model=filtered_overrides.get("model", config.model),
                device=filtered_overrides.get("device", config.device),
                compute_type=filtered_overrides.get("compute_type", config.compute_type),
                language=filtered_overrides.get("language", config.language),
                task=filtered_overrides.get("task", config.task),
                vad_min_silence_ms=filtered_overrides.get(
                    "vad_min_silence_ms", config.vad_min_silence_ms
                ),
                beam_size=filtered_overrides.get("beam_size", config.beam_size),
                skip_existing_json=filtered_overrides.get(
                    "skip_existing_json", config.skip_existing_json
                ),
                enable_chunking=filtered_overrides.get("enable_chunking", config.enable_chunking),
                chunk_target_duration_s=filtered_overrides.get(
                    "chunk_target_duration_s", config.chunk_target_duration_s
                ),
                chunk_max_duration_s=filtered_overrides.get(
                    "chunk_max_duration_s", config.chunk_max_duration_s
                ),
                chunk_target_tokens=filtered_overrides.get(
                    "chunk_target_tokens", config.chunk_target_tokens
                ),
                chunk_pause_split_threshold_s=filtered_overrides.get(
                    "chunk_pause_split_threshold_s", config.chunk_pause_split_threshold_s
                ),
                enable_diarization=filtered_overrides.get(
                    "enable_diarization", config.enable_diarization
                ),
                diarization_device=filtered_overrides.get(
                    "diarization_device", config.diarization_device
                ),
                min_speakers=filtered_overrides.get("min_speakers", config.min_speakers),
                max_speakers=filtered_overrides.get("max_speakers", config.max_speakers),
                overlap_threshold=filtered_overrides.get(
                    "overlap_threshold", config.overlap_threshold
                ),
            )
            config = override_config
        else:
            override_compute_type_explicit = False

        # If compute_type wasn't explicitly provided by any source,
        # re-derive the default based on the final device selection.
        compute_type_explicit = (
            override_compute_type_explicit
            or env_compute_type_explicit
            or file_compute_type_explicit
        )
        if not compute_type_explicit:
            # Use object.__setattr__ because of slots=True
            object.__setattr__(
                config, "compute_type", "int8" if config.device == "cpu" else "float16"
            )

        # Validate final configuration
        validate_diarization_settings(
            config.min_speakers,
            config.max_speakers,
            config.overlap_threshold,
        )

        return config


def validate_diarization_settings(
    min_speakers: int | None,
    max_speakers: int | None,
    overlap_threshold: float | None = None,
) -> None:
    """
    Validate diarization-related configuration values.

    Ensures provided speaker bounds are positive integers, overlap thresholds
    are within [0.0, 1.0], and that min_speakers does not exceed max_speakers
    when both are set.

    Raises:
        ConfigurationError: If any constraint is violated.
    """
    if min_speakers is not None:
        if isinstance(min_speakers, bool) or not isinstance(min_speakers, int):
            raise ConfigurationError("min_speakers must be a positive integer when provided.")
        if min_speakers < 1:
            raise ConfigurationError("min_speakers must be a positive integer when provided.")

    if max_speakers is not None:
        if isinstance(max_speakers, bool) or not isinstance(max_speakers, int):
            raise ConfigurationError("max_speakers must be a positive integer when provided.")
        if max_speakers < 1:
            raise ConfigurationError("max_speakers must be a positive integer when provided.")

    if min_speakers is not None and max_speakers is not None and min_speakers > max_speakers:
        raise ConfigurationError(
            f"min_speakers ({min_speakers}) cannot be greater than max_speakers ({max_speakers})."
        )

    if overlap_threshold is not None:
        if isinstance(overlap_threshold, bool) or not isinstance(
            overlap_threshold,
            int | float,
        ):
            raise ConfigurationError("overlap_threshold must be between 0.0 and 1.0.")
        if not 0.0 <= float(overlap_threshold) <= 1.0:
            raise ConfigurationError("overlap_threshold must be between 0.0 and 1.0.")


def _merge_configs(base: TranscriptionConfig, override: TranscriptionConfig) -> TranscriptionConfig:
    """
    Merge two TranscriptionConfig instances, with override taking precedence.

    This function implements proper config layering by checking which fields
    were explicitly set in the override config (via _source_fields attribute).
    Only explicitly-set fields override the base config, preventing the bug where
    a file value that equals the default would be ignored.

    The merge strategy:
    - If field was explicitly set in override (in _source_fields): use override value
    - Otherwise: keep base value

    This correctly handles the case where:
    1. Env sets device="cpu"
    2. File explicitly sets device="cuda" (which happens to equal the default)
    3. Result is "cuda" (file overrides env) ✓

    And also handles:
    1. Env sets device="cpu"
    2. File doesn't mention device (so device="cuda" is just a default)
    3. Result is "cpu" (env value preserved) ✓

    Args:
        base: Base configuration (lower precedence).
        override: Override configuration (higher precedence).

    Returns:
        New TranscriptionConfig with merged values.
    """
    # Check if override has _source_fields (set by from_file/from_env)
    # If not, or if it's empty, fall back to comparing values (for backward compatibility)
    source_fields = getattr(override, "_source_fields", None)

    if source_fields is None or not source_fields:
        # No source tracking - fall back to value comparison
        # This shouldn't happen in normal CLI usage, but provides safety
        return TranscriptionConfig(
            model=override.model if override.model != base.model else base.model,
            device=override.device if override.device != base.device else base.device,
            compute_type=override.compute_type
            if override.compute_type != base.compute_type
            else base.compute_type,
            language=override.language if override.language != base.language else base.language,
            task=override.task if override.task != base.task else base.task,
            vad_min_silence_ms=override.vad_min_silence_ms
            if override.vad_min_silence_ms != base.vad_min_silence_ms
            else base.vad_min_silence_ms,
            beam_size=override.beam_size
            if override.beam_size != base.beam_size
            else base.beam_size,
            skip_existing_json=override.skip_existing_json
            if override.skip_existing_json != base.skip_existing_json
            else base.skip_existing_json,
            enable_chunking=override.enable_chunking
            if override.enable_chunking != base.enable_chunking
            else base.enable_chunking,
            chunk_target_duration_s=override.chunk_target_duration_s
            if override.chunk_target_duration_s != base.chunk_target_duration_s
            else base.chunk_target_duration_s,
            chunk_max_duration_s=override.chunk_max_duration_s
            if override.chunk_max_duration_s != base.chunk_max_duration_s
            else base.chunk_max_duration_s,
            chunk_target_tokens=override.chunk_target_tokens
            if override.chunk_target_tokens != base.chunk_target_tokens
            else base.chunk_target_tokens,
            chunk_pause_split_threshold_s=override.chunk_pause_split_threshold_s
            if override.chunk_pause_split_threshold_s != base.chunk_pause_split_threshold_s
            else base.chunk_pause_split_threshold_s,
            enable_diarization=override.enable_diarization
            if override.enable_diarization != base.enable_diarization
            else base.enable_diarization,
            diarization_device=override.diarization_device
            if override.diarization_device != base.diarization_device
            else base.diarization_device,
            min_speakers=override.min_speakers
            if override.min_speakers != base.min_speakers
            else base.min_speakers,
            max_speakers=override.max_speakers
            if override.max_speakers != base.max_speakers
            else base.max_speakers,
            overlap_threshold=override.overlap_threshold
            if override.overlap_threshold != base.overlap_threshold
            else base.overlap_threshold,
        )

    # Use source_fields to determine which values to override
    return TranscriptionConfig(
        model=override.model if "model" in source_fields else base.model,
        device=override.device if "device" in source_fields else base.device,
        compute_type=override.compute_type
        if "compute_type" in source_fields
        else base.compute_type,
        language=override.language if "language" in source_fields else base.language,
        task=override.task if "task" in source_fields else base.task,
        vad_min_silence_ms=override.vad_min_silence_ms
        if "vad_min_silence_ms" in source_fields
        else base.vad_min_silence_ms,
        beam_size=override.beam_size if "beam_size" in source_fields else base.beam_size,
        skip_existing_json=override.skip_existing_json
        if "skip_existing_json" in source_fields
        else base.skip_existing_json,
        enable_chunking=override.enable_chunking
        if "enable_chunking" in source_fields
        else base.enable_chunking,
        chunk_target_duration_s=override.chunk_target_duration_s
        if "chunk_target_duration_s" in source_fields
        else base.chunk_target_duration_s,
        chunk_max_duration_s=override.chunk_max_duration_s
        if "chunk_max_duration_s" in source_fields
        else base.chunk_max_duration_s,
        chunk_target_tokens=override.chunk_target_tokens
        if "chunk_target_tokens" in source_fields
        else base.chunk_target_tokens,
        chunk_pause_split_threshold_s=override.chunk_pause_split_threshold_s
        if "chunk_pause_split_threshold_s" in source_fields
        else base.chunk_pause_split_threshold_s,
        enable_diarization=override.enable_diarization
        if "enable_diarization" in source_fields
        else base.enable_diarization,
        diarization_device=override.diarization_device
        if "diarization_device" in source_fields
        else base.diarization_device,
        min_speakers=override.min_speakers
        if "min_speakers" in source_fields
        else base.min_speakers,
        max_speakers=override.max_speakers
        if "max_speakers" in source_fields
        else base.max_speakers,
        overlap_threshold=override.overlap_threshold
        if "overlap_threshold" in source_fields
        else base.overlap_threshold,
    )


def _merge_enrich_configs(base: EnrichmentConfig, override: EnrichmentConfig) -> EnrichmentConfig:
    """
    Merge two EnrichmentConfig instances, with override taking precedence.

    This function implements proper config layering by checking which fields
    were explicitly set in the override config (via _source_fields attribute).
    Only explicitly-set fields override the base config.

    See _merge_configs() for detailed explanation of the merge strategy.

    Args:
        base: Base configuration (lower precedence).
        override: Override configuration (higher precedence).

    Returns:
        New EnrichmentConfig with merged values.
    """
    # Check if override has _source_fields (set by from_file/from_env)
    # If not, or if it's empty, fall back to comparing values (for backward compatibility)
    source_fields = getattr(override, "_source_fields", None)

    if source_fields is None or not source_fields:
        # No source tracking - fall back to value comparison
        return EnrichmentConfig(
            skip_existing=override.skip_existing
            if override.skip_existing != base.skip_existing
            else base.skip_existing,
            enable_prosody=override.enable_prosody
            if override.enable_prosody != base.enable_prosody
            else base.enable_prosody,
            enable_emotion=override.enable_emotion
            if override.enable_emotion != base.enable_emotion
            else base.enable_emotion,
            enable_categorical_emotion=override.enable_categorical_emotion
            if override.enable_categorical_emotion != base.enable_categorical_emotion
            else base.enable_categorical_emotion,
            enable_semantic_annotator=override.enable_semantic_annotator
            if override.enable_semantic_annotator != base.enable_semantic_annotator
            else base.enable_semantic_annotator,
            enable_turn_metadata=override.enable_turn_metadata
            if override.enable_turn_metadata != base.enable_turn_metadata
            else base.enable_turn_metadata,
            enable_speaker_stats=override.enable_speaker_stats
            if override.enable_speaker_stats != base.enable_speaker_stats
            else base.enable_speaker_stats,
            device=override.device if override.device != base.device else base.device,
            dimensional_model_name=override.dimensional_model_name
            if override.dimensional_model_name != base.dimensional_model_name
            else base.dimensional_model_name,
            categorical_model_name=override.categorical_model_name
            if override.categorical_model_name != base.categorical_model_name
            else base.categorical_model_name,
            pause_threshold=override.pause_threshold
            if override.pause_threshold != base.pause_threshold
            else base.pause_threshold,
            semantic_annotator=override.semantic_annotator or base.semantic_annotator,
        )

    # Use source_fields to determine which values to override
    return EnrichmentConfig(
        skip_existing=override.skip_existing
        if "skip_existing" in source_fields
        else base.skip_existing,
        enable_prosody=override.enable_prosody
        if "enable_prosody" in source_fields
        else base.enable_prosody,
        enable_emotion=override.enable_emotion
        if "enable_emotion" in source_fields
        else base.enable_emotion,
        enable_categorical_emotion=override.enable_categorical_emotion
        if "enable_categorical_emotion" in source_fields
        else base.enable_categorical_emotion,
        enable_semantic_annotator=override.enable_semantic_annotator
        if "enable_semantic_annotator" in source_fields
        else base.enable_semantic_annotator,
        enable_turn_metadata=override.enable_turn_metadata
        if "enable_turn_metadata" in source_fields
        else base.enable_turn_metadata,
        enable_speaker_stats=override.enable_speaker_stats
        if "enable_speaker_stats" in source_fields
        else base.enable_speaker_stats,
        device=override.device if "device" in source_fields else base.device,
        dimensional_model_name=override.dimensional_model_name
        if "dimensional_model_name" in source_fields
        else base.dimensional_model_name,
        categorical_model_name=override.categorical_model_name
        if "categorical_model_name" in source_fields
        else base.categorical_model_name,
        pause_threshold=override.pause_threshold
        if "pause_threshold" in source_fields
        else base.pause_threshold,
        semantic_annotator=override.semantic_annotator or base.semantic_annotator,
    )


@dataclass(slots=True)
class EnrichmentConfig:
    """
    High-level enrichment configuration for public API and CLI.

    Controls prosody/emotion extraction and device selection for Stage 2.
    Also gates optional turn/speaker analytics (v1.2).
    """

    # What to enrich
    skip_existing: bool = True
    enable_prosody: bool = True
    enable_emotion: bool = True
    enable_categorical_emotion: bool = False
    enable_turn_metadata: bool = True
    enable_speaker_stats: bool = True
    enable_semantic_annotator: bool = False
    semantic_annotator: Any | None = field(default=None, repr=False, compare=False)

    # Runtime
    device: str = "cpu"  # "cpu" or "cuda"

    # Optional model overrides
    dimensional_model_name: str | None = None
    categorical_model_name: str | None = None

    # Turn building configuration
    pause_threshold: float | None = None  # Minimum pause (seconds) to split turns

    # Internal field to track which config values were explicitly set
    _source_fields: set[str] = field(default_factory=set, init=False, repr=False)

    def __post_init__(self):
        """Validate configuration values."""
        if self.pause_threshold is not None and self.pause_threshold < 0.0:
            raise ConfigurationError(f"pause_threshold must be >= 0.0, got {self.pause_threshold}")

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
                "enable_turn_metadata": true,
                "enable_speaker_stats": true,
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
            "enable_turn_metadata",
            "enable_speaker_stats",
            "enable_semantic_annotator",
        ]
        for field_name in bool_fields:
            if field_name in data and not isinstance(data[field_name], bool):
                raise ValueError(
                    f"{field_name} must be a boolean, got {type(data[field_name]).__name__}"
                )

        # Validate pause_threshold if present
        if "pause_threshold" in data:
            pause_val = data["pause_threshold"]
            if pause_val is not None:
                if isinstance(pause_val, bool) or not isinstance(pause_val, (int, float)):
                    raise ValueError(
                        f"pause_threshold must be a number or null, got {type(pause_val).__name__}"
                    )
                if pause_val < 0.0:
                    raise ValueError(f"pause_threshold must be >= 0.0, got {pause_val}")

        # Filter out unknown fields
        valid_fields = {
            "skip_existing",
            "enable_prosody",
            "enable_emotion",
            "enable_categorical_emotion",
            "enable_turn_metadata",
            "enable_speaker_stats",
            "enable_semantic_annotator",
            "device",
            "dimensional_model_name",
            "categorical_model_name",
            "pause_threshold",
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
        - {prefix}ENABLE_TURN_METADATA -> enable_turn_metadata
        - {prefix}ENABLE_SPEAKER_STATS -> enable_speaker_stats
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
            "ENABLE_TURN_METADATA": "enable_turn_metadata",
            "ENABLE_SPEAKER_STATS": "enable_speaker_stats",
            "ENABLE_SEMANTIC_ANNOTATOR": "enable_semantic_annotator",
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

        # Float field (pause_threshold)
        if pause_thresh := os.getenv(f"{prefix}PAUSE_THRESHOLD"):
            try:
                if pause_thresh.lower() in ("none", "null", ""):
                    config_dict["pause_threshold"] = None
                else:
                    pause_value = float(pause_thresh)
                    if pause_value < 0.0:
                        raise ValueError(
                            f"{prefix}PAUSE_THRESHOLD must be >= 0.0, got {pause_value}"
                        )
                    config_dict["pause_threshold"] = pause_value
            except ValueError as e:
                raise ValueError(
                    f"Invalid {prefix}PAUSE_THRESHOLD: {pause_thresh}. Must be a non-negative float."
                ) from e

        config = cls(**config_dict)
        # Track which fields were explicitly set from environment
        config._source_fields = set(config_dict.keys())
        return config

    @classmethod
    def from_sources(
        cls,
        env_prefix: str = "SLOWER_WHISPER_ENRICH_",
        config_file: str | Path | None = None,
        **overrides: Any,
    ) -> EnrichmentConfig:
        """
        Load configuration from multiple sources with proper precedence.

        Precedence order (highest to lowest):
        1. Explicit keyword arguments (overrides parameter)
        2. Config file (if config_file provided)
        3. Environment variables (with env_prefix)
        4. Defaults

        This method implements the same precedence logic as the CLI, making it
        available for programmatic use without needing to go through argparse.

        Args:
            env_prefix: Environment variable prefix (default: "SLOWER_WHISPER_ENRICH_").
                Used to look up env vars like {prefix}DEVICE, {prefix}ENABLE_PROSODY, etc.
            config_file: Optional path to JSON configuration file.
            **overrides: Explicit configuration values that override all other sources.
                Only non-None values are applied. All EnrichmentConfig fields are
                supported (device, enable_prosody, enable_emotion, etc.).

        Returns:
            EnrichmentConfig with merged settings from all sources.

        Raises:
            FileNotFoundError: If config_file is specified but doesn't exist.
            ValueError: If config file or environment variables contain invalid values.
            ConfigurationError: If configuration values fail validation.

        Examples:
            # Use defaults
            config = EnrichmentConfig.from_sources()

            # Load from environment variables only
            config = EnrichmentConfig.from_sources()

            # Load from config file and environment
            config = EnrichmentConfig.from_sources(
                config_file=Path("enrich_config.json")
            )

            # Override specific values
            config = EnrichmentConfig.from_sources(
                config_file=Path("enrich_config.json"),
                device="cuda",
                enable_prosody=True
            )

            # Custom environment prefix
            config = EnrichmentConfig.from_sources(
                env_prefix="MY_APP_ENRICH_",
                device="cpu"
            )

            # Full precedence example
            # 1. Defaults: device="cpu", enable_prosody=True
            # 2. Env: SLOWER_WHISPER_ENRICH_DEVICE=cuda -> device="cuda"
            # 3. File: {"enable_prosody": false} -> device="cuda", enable_prosody=False
            # 4. CLI: enable_emotion=False -> device="cuda", enable_prosody=False, enable_emotion=False
            config = EnrichmentConfig.from_sources(
                config_file="enrich_config.json",
                enable_emotion=False  # Overrides env and file
            )
        """
        # Step 1: Start with defaults
        config = cls()

        # Step 2: Override with environment variables
        env_config = cls.from_env(prefix=env_prefix)
        config = _merge_enrich_configs(config, env_config)

        # Step 3: Override with config file if provided
        if config_file is not None:
            file_config = cls.from_file(config_file)
            config = _merge_enrich_configs(config, file_config)

        # Step 4: Override with explicit keyword arguments (only if not None)
        # Filter out None values to distinguish between "not set" and "set to None"
        filtered_overrides = {k: v for k, v in overrides.items() if v is not None}

        if filtered_overrides:
            # Create a new config with overrides applied
            override_config = cls(
                skip_existing=filtered_overrides.get("skip_existing", config.skip_existing),
                enable_prosody=filtered_overrides.get("enable_prosody", config.enable_prosody),
                enable_emotion=filtered_overrides.get("enable_emotion", config.enable_emotion),
                enable_categorical_emotion=filtered_overrides.get(
                    "enable_categorical_emotion", config.enable_categorical_emotion
                ),
                enable_turn_metadata=filtered_overrides.get(
                    "enable_turn_metadata", config.enable_turn_metadata
                ),
                enable_speaker_stats=filtered_overrides.get(
                    "enable_speaker_stats", config.enable_speaker_stats
                ),
                enable_semantic_annotator=filtered_overrides.get(
                    "enable_semantic_annotator", config.enable_semantic_annotator
                ),
                device=filtered_overrides.get("device", config.device),
                dimensional_model_name=filtered_overrides.get(
                    "dimensional_model_name", config.dimensional_model_name
                ),
                categorical_model_name=filtered_overrides.get(
                    "categorical_model_name", config.categorical_model_name
                ),
                pause_threshold=filtered_overrides.get("pause_threshold", config.pause_threshold),
                semantic_annotator=filtered_overrides.get(
                    "semantic_annotator", config.semantic_annotator
                ),
            )
            config = override_config

        return config
