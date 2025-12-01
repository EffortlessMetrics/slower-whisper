from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from .exceptions import ConfigurationError

WhisperTask = Literal["transcribe", "translate"]
ALLOWED_COMPUTE_TYPES: set[str] = {
    "float16",
    "float32",
    "int16",
    "int8",
    "int8_float16",
    "int8_float32",
}


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
        """Auto-detect compute_type based on device if using default."""
        if self.compute_type is None:
            # Use sensible defaults: int8 for CPU, float16 for CUDA
            self.compute_type = "int8" if self.device == "cpu" else "float16"

        self.compute_type = validate_compute_type(self.compute_type) or self.compute_type


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
        """Auto-detect compute_type based on device if using default."""
        compute_type = self.compute_type
        if compute_type is None:
            # Use sensible defaults: int8 for CPU, float16 for CUDA
            compute_type = "int8" if self.device == "cpu" else "float16"

        validated = validate_compute_type(compute_type) or compute_type
        object.__setattr__(self, "compute_type", validated)

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

        config = cls(**config_dict)
        # Track which fields were explicitly set from environment
        config._source_fields = set(config_dict.keys())
        return config
