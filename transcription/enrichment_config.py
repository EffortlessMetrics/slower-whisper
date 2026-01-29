"""Enrichment configuration for audio enrichment pipelines.

This module defines the EnrichmentConfig dataclass used for Stage 2 (audio enrichment)
settings. Configuration can be created programmatically or loaded from JSON files and
environment variables.

The EnrichmentConfig controls:
- Prosody and emotion extraction
- Device selection for enrichment models
- Optional turn/speaker analytics
- Turn building configuration
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config_merge import _merge_enrich_configs
from .exceptions import ConfigurationError


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

    # Conversation Intelligence features (v2.0)
    enable_safety_layer: bool = False  # Combined PII/moderation/formatting
    enable_role_inference: bool = False  # Agent/customer/facilitator detection
    enable_topic_segmentation: bool = False  # Topic boundary detection
    enable_prosody_v2: bool = False  # Extended prosody (boundary tone, monotony)
    enable_environment_classifier: bool = False  # Audio environment classification
    turn_taking_policy: str = "balanced"  # aggressive/balanced/conservative

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
        if self.turn_taking_policy not in ("aggressive", "balanced", "conservative"):
            raise ConfigurationError(
                f"turn_taking_policy must be aggressive/balanced/conservative, got {self.turn_taking_policy}"
            )

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
            "enable_safety_layer",
            "enable_role_inference",
            "enable_topic_segmentation",
            "enable_prosody_v2",
            "enable_environment_classifier",
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
                if isinstance(pause_val, bool) or not isinstance(pause_val, int | float):
                    raise ValueError(
                        f"pause_threshold must be a number or null, got {type(pause_val).__name__}"
                    )
                if pause_val < 0.0:
                    raise ValueError(f"pause_threshold must be >= 0.0, got {pause_val}")

        # Validate turn_taking_policy if present
        if "turn_taking_policy" in data:
            policy = data["turn_taking_policy"]
            if policy not in ("aggressive", "balanced", "conservative"):
                raise ValueError(
                    f"turn_taking_policy must be aggressive/balanced/conservative, got {policy}"
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
            "enable_safety_layer",
            "enable_role_inference",
            "enable_topic_segmentation",
            "enable_prosody_v2",
            "enable_environment_classifier",
            "turn_taking_policy",
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
            "ENABLE_SAFETY_LAYER": "enable_safety_layer",
            "ENABLE_ROLE_INFERENCE": "enable_role_inference",
            "ENABLE_TOPIC_SEGMENTATION": "enable_topic_segmentation",
            "ENABLE_PROSODY_V2": "enable_prosody_v2",
            "ENABLE_ENVIRONMENT_CLASSIFIER": "enable_environment_classifier",
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

        # String field (turn_taking_policy)
        if policy := os.getenv(f"{prefix}TURN_TAKING_POLICY"):
            if policy.lower() not in ("aggressive", "balanced", "conservative"):
                raise ValueError(
                    f"Invalid {prefix}TURN_TAKING_POLICY: {policy}. "
                    "Must be aggressive/balanced/conservative"
                )
            config_dict["turn_taking_policy"] = policy.lower()

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
                enable_safety_layer=filtered_overrides.get(
                    "enable_safety_layer", config.enable_safety_layer
                ),
                enable_role_inference=filtered_overrides.get(
                    "enable_role_inference", config.enable_role_inference
                ),
                enable_topic_segmentation=filtered_overrides.get(
                    "enable_topic_segmentation", config.enable_topic_segmentation
                ),
                enable_prosody_v2=filtered_overrides.get(
                    "enable_prosody_v2", config.enable_prosody_v2
                ),
                enable_environment_classifier=filtered_overrides.get(
                    "enable_environment_classifier", config.enable_environment_classifier
                ),
                turn_taking_policy=filtered_overrides.get(
                    "turn_taking_policy", config.turn_taking_policy
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
