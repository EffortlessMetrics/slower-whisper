"""Transcription configuration for the slower-whisper pipeline.

This module contains the TranscriptionConfig dataclass, which is the public API
for Stage 1 (ASR) settings. The configuration can be created programmatically or
loaded from JSON files and environment variables.

The module provides three loader methods:
- from_file(): Load configuration from a JSON file
- from_env(): Load configuration from environment variables
- from_sources(): Load configuration from multiple sources with proper precedence

Precedence order (highest to lowest):
1. Explicit keyword arguments
2. Config file
3. Environment variables
4. Defaults
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config_merge import _merge_configs
from .config_validation import (
    WhisperTask,
    auto_derive_compute_type,
    validate_compute_type,
    validate_diarization_settings,
    validate_model_name,
)
from .exceptions import ConfigurationError

logger = logging.getLogger(__name__)


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

    # Word-level alignment (v1.8+)
    word_timestamps: bool = False

    # Chunking (v1.3 preview)
    enable_chunking: bool = False
    chunk_target_duration_s: float = 30.0
    chunk_max_duration_s: float = 45.0
    chunk_target_tokens: int = 400
    chunk_pause_split_threshold_s: float = 1.5

    # v1.1+ diarization (L2) — opt-in
    enable_diarization: bool = False
    diarization_device: str = "auto"  # "cuda" | "cpu" | "auto"
    min_speakers: int | None = None
    max_speakers: int | None = None
    overlap_threshold: float = 0.3  # exposed via --overlap-threshold

    # Internal field to track which config values were explicitly set
    # Used by CLI merge logic to implement correct precedence
    _source_fields: set[str] = field(default_factory=set, init=False, repr=False)

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
            "word_timestamps",
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
            "word_timestamps",
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
        if word_timestamps := os.getenv(f"{prefix}WORD_TIMESTAMPS"):
            word_timestamps_lower = word_timestamps.lower()
            if word_timestamps_lower in ("true", "1", "yes", "on"):
                config_dict["word_timestamps"] = True
            elif word_timestamps_lower in ("false", "0", "no", "off"):
                config_dict["word_timestamps"] = False
            else:
                raise ValueError(
                    f"Invalid {prefix}WORD_TIMESTAMPS: {word_timestamps}. "
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
                word_timestamps=filtered_overrides.get("word_timestamps", config.word_timestamps),
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
