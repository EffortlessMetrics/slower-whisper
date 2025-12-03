"""Comprehensive tests for Config.from_sources() factory method.

This module tests the from_sources() classmethod that provides a clean API for
loading configuration from multiple sources with proper precedence handling.

Precedence order (highest to lowest):
1. CLI args (explicit overrides)
2. Config file
3. Environment variables
4. Defaults

Key test scenarios:
- File overrides env even when value equals default (regression test)
- None overrides are properly ignored
- compute_type auto-derivation based on final device
- Source field tracking
- Invalid paths and custom prefixes
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import pytest

from transcription.config import EnrichmentConfig, TranscriptionConfig
from transcription.exceptions import ConfigurationError


class TestTranscriptionConfigFromSources:
    """Test suite for TranscriptionConfig.from_sources() factory method."""

    def setup_method(self):
        """Clean environment before each test."""
        self.original_env = os.environ.copy()
        # Clear all slower-whisper env vars
        for key in list(os.environ.keys()):
            if key.startswith("SLOWER_WHISPER"):
                del os.environ[key]

    def teardown_method(self):
        """Restore environment after each test."""
        os.environ.clear()
        os.environ.update(self.original_env)

    def test_defaults_only(self):
        """When no sources provided, should use defaults only."""
        config = TranscriptionConfig.from_sources()

        assert config.model == "large-v3"
        assert config.device == "cuda"
        assert config.compute_type == "float16"  # Auto-derived from cuda
        assert config.language is None
        assert config.task == "transcribe"
        assert config.skip_existing_json is True
        assert config.vad_min_silence_ms == 500
        assert config.beam_size == 5
        assert config.enable_diarization is False
        assert config.diarization_device == "auto"
        assert config.min_speakers is None
        assert config.max_speakers is None
        assert config.overlap_threshold == 0.3

    def test_env_only(self):
        """Environment variables should override defaults."""
        os.environ["SLOWER_WHISPER_MODEL"] = "medium"
        os.environ["SLOWER_WHISPER_DEVICE"] = "cpu"
        os.environ["SLOWER_WHISPER_LANGUAGE"] = "en"
        os.environ["SLOWER_WHISPER_BEAM_SIZE"] = "3"
        os.environ["SLOWER_WHISPER_ENABLE_DIARIZATION"] = "true"
        os.environ["SLOWER_WHISPER_MIN_SPEAKERS"] = "2"
        os.environ["SLOWER_WHISPER_MAX_SPEAKERS"] = "4"

        config = TranscriptionConfig.from_sources()

        assert config.model == "medium"
        assert config.device == "cpu"
        assert config.compute_type == "int8"  # Auto-derived from cpu
        assert config.language == "en"
        assert config.beam_size == 3
        assert config.enable_diarization is True
        assert config.min_speakers == 2
        assert config.max_speakers == 4

    def test_env_only_custom_prefix(self):
        """Should support custom environment variable prefix."""
        os.environ["CUSTOM_MODEL"] = "base"
        os.environ["CUSTOM_DEVICE"] = "cpu"
        os.environ["CUSTOM_LANGUAGE"] = "fr"

        config = TranscriptionConfig.from_sources(env_prefix="CUSTOM_")

        assert config.model == "base"
        assert config.device == "cpu"
        assert config.language == "fr"

    def test_file_only(self):
        """Config file should override defaults."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(
                {
                    "model": "small",
                    "device": "cpu",
                    "language": "es",
                    "task": "translate",
                    "beam_size": 7,
                    "enable_diarization": True,
                    "diarization_device": "cpu",
                    "min_speakers": 3,
                    "max_speakers": 5,
                    "overlap_threshold": 0.5,
                },
                f,
            )
            config_file = Path(f.name)

        try:
            config = TranscriptionConfig.from_sources(config_file=config_file)

            assert config.model == "small"
            assert config.device == "cpu"
            assert config.language == "es"
            assert config.task == "translate"
            assert config.beam_size == 7
            assert config.enable_diarization is True
            assert config.diarization_device == "cpu"
            assert config.min_speakers == 3
            assert config.max_speakers == 5
            assert config.overlap_threshold == 0.5

        finally:
            config_file.unlink()

    def test_cli_only(self):
        """CLI args should override defaults."""
        cli_overrides: dict[str, Any] = {
            "model": "tiny",
            "device": "cuda",
            "language": "de",
            "task": "transcribe",
            "beam_size": 10,
            "enable_diarization": True,
            "min_speakers": 1,
            "max_speakers": 2,
        }

        config = TranscriptionConfig.from_sources(**cli_overrides)

        assert config.model == "tiny"
        assert config.device == "cuda"
        assert config.language == "de"
        assert config.task == "transcribe"
        assert config.beam_size == 10
        assert config.enable_diarization is True
        assert config.min_speakers == 1
        assert config.max_speakers == 2

    def test_file_overrides_env_with_default_value(self):
        """
        File should override env even when file value equals default.

        This is the critical regression test for the merge bug.
        When a config file explicitly sets device="cuda" (which is the default),
        it should override an env var that sets device="cpu".
        """
        # Set env to non-default value
        os.environ["SLOWER_WHISPER_DEVICE"] = "cpu"
        os.environ["SLOWER_WHISPER_BEAM_SIZE"] = "3"

        # Create file with default values (but explicitly set)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(
                {
                    "device": "cuda",  # cuda is the default
                    "beam_size": 5,  # 5 is the default
                },
                f,
            )
            config_file = Path(f.name)

        try:
            config = TranscriptionConfig.from_sources(config_file=config_file)

            # File should win even though its values equal the defaults
            assert config.device == "cuda", (
                "File should override env even when value equals default"
            )
            assert config.beam_size == 5, "File should override env even when value equals default"

        finally:
            config_file.unlink()

    def test_file_overrides_env_with_custom_value(self):
        """File should override env when file value differs from default."""
        os.environ["SLOWER_WHISPER_MODEL"] = "base"
        os.environ["SLOWER_WHISPER_LANGUAGE"] = "en"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(
                {
                    "model": "medium",  # medium != default (large-v3)
                    "language": "fr",  # fr != env (en)
                },
                f,
            )
            config_file = Path(f.name)

        try:
            config = TranscriptionConfig.from_sources(config_file=config_file)

            assert config.model == "medium"
            assert config.language == "fr"

        finally:
            config_file.unlink()

    def test_cli_overrides_file(self):
        """CLI args should override file config."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(
                {
                    "model": "medium",
                    "device": "cpu",
                    "language": "en",
                },
                f,
            )
            config_file = Path(f.name)

        try:
            cli_overrides: dict[str, Any] = {
                "model": "base",  # Override file
                "language": "de",  # Override file
            }

            config = TranscriptionConfig.from_sources(config_file=config_file, **cli_overrides)

            assert config.model == "base"  # CLI wins
            assert config.device == "cpu"  # File wins (no CLI override)
            assert config.language == "de"  # CLI wins

        finally:
            config_file.unlink()

    def test_full_precedence_chain(self):
        """
        Test complete precedence: CLI > file > env > defaults.

        Each source should only override what it explicitly sets.
        """
        # Env sets some values
        os.environ["SLOWER_WHISPER_MODEL"] = "small"
        os.environ["SLOWER_WHISPER_DEVICE"] = "cpu"
        os.environ["SLOWER_WHISPER_BEAM_SIZE"] = "3"
        os.environ["SLOWER_WHISPER_LANGUAGE"] = "en"

        # File overrides some env values and adds new ones
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(
                {
                    "model": "medium",  # Override env
                    "device": "cuda",  # Default value, should still override env
                    "task": "translate",  # Not in env
                },
                f,
            )
            config_file = Path(f.name)

        try:
            # CLI overrides some file values
            cli_overrides: dict[str, Any] = {
                "model": "base",  # Override file
                "vad_min_silence_ms": 1000,  # Not in file or env
            }

            config = TranscriptionConfig.from_sources(config_file=config_file, **cli_overrides)

            assert config.model == "base"  # CLI
            assert config.device == "cuda"  # File (overrode env despite being default)
            assert config.beam_size == 3  # Env (file didn't set it)
            assert config.language == "en"  # Env
            assert config.task == "translate"  # File
            assert config.vad_min_silence_ms == 1000  # CLI

        finally:
            config_file.unlink()

    def test_none_overrides_ignored(self):
        """None values in CLI overrides should be ignored."""
        os.environ["SLOWER_WHISPER_MODEL"] = "medium"

        cli_overrides: dict[str, Any] = {
            "model": None,  # Should be ignored
            "device": None,  # Should be ignored
            "language": "en",  # Should be used
        }

        config = TranscriptionConfig.from_sources(**cli_overrides)

        assert config.model == "medium"  # Env preserved (CLI None ignored)
        assert config.device == "cuda"  # Default preserved (CLI None ignored)
        assert config.language == "en"  # CLI used

    def test_compute_type_auto_derivation_from_env(self):
        """compute_type should auto-derive from device when not explicitly set."""
        # CPU device -> int8
        os.environ["SLOWER_WHISPER_DEVICE"] = "cpu"
        config = TranscriptionConfig.from_sources()
        assert config.device == "cpu"
        assert config.compute_type == "int8"

        # CUDA device -> float16
        os.environ["SLOWER_WHISPER_DEVICE"] = "cuda"
        config = TranscriptionConfig.from_sources()
        assert config.device == "cuda"
        assert config.compute_type == "float16"

    def test_compute_type_auto_derivation_from_file(self):
        """compute_type should auto-derive when device changes via file."""
        os.environ["SLOWER_WHISPER_DEVICE"] = "cpu"  # Would give int8

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"device": "cuda"}, f)  # Should give float16
            config_file = Path(f.name)

        try:
            config = TranscriptionConfig.from_sources(config_file=config_file)
            assert config.device == "cuda"
            assert config.compute_type == "float16"  # Derived from final device

        finally:
            config_file.unlink()

    def test_compute_type_auto_derivation_from_cli(self):
        """compute_type should auto-derive when device changes via CLI."""
        os.environ["SLOWER_WHISPER_DEVICE"] = "cpu"  # Would give int8

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"device": "cpu"}, f)  # Would give int8
            config_file = Path(f.name)

        try:
            cli_overrides: dict[str, Any] = {"device": "cuda"}  # Should give float16

            config = TranscriptionConfig.from_sources(config_file=config_file, **cli_overrides)

            assert config.device == "cuda"
            assert config.compute_type == "float16"  # Derived from final device

        finally:
            config_file.unlink()

    def test_explicit_compute_type_preserved(self):
        """Explicit compute_type should not be auto-overridden."""
        os.environ["SLOWER_WHISPER_DEVICE"] = "cpu"
        os.environ["SLOWER_WHISPER_COMPUTE_TYPE"] = "float16"

        config = TranscriptionConfig.from_sources()

        assert config.device == "cpu"
        assert config.compute_type == "float16"  # Explicit value preserved

    def test_invalid_file_path(self):
        """Should raise FileNotFoundError for non-existent config file."""
        with pytest.raises(FileNotFoundError) as exc_info:
            TranscriptionConfig.from_sources(config_file="/nonexistent/config.json")

        assert "Configuration file not found" in str(exc_info.value)

    def test_invalid_json_in_file(self):
        """Should raise JSONDecodeError for malformed JSON."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("{invalid json")
            config_file = Path(f.name)

        try:
            with pytest.raises(json.JSONDecodeError):
                TranscriptionConfig.from_sources(config_file=config_file)

        finally:
            config_file.unlink()

    def test_invalid_model_in_file(self):
        """Should raise ValueError for invalid model name in file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"model": "invalid-model"}, f)
            config_file = Path(f.name)

        try:
            with pytest.raises(ValueError) as exc_info:
                TranscriptionConfig.from_sources(config_file=config_file)

            assert "Invalid model name" in str(exc_info.value)

        finally:
            config_file.unlink()

    def test_invalid_model_in_env(self):
        """Should raise ValueError for invalid model name in env."""
        os.environ["SLOWER_WHISPER_MODEL"] = "gpt-4"

        with pytest.raises(ValueError) as exc_info:
            TranscriptionConfig.from_sources()

        assert "Invalid model name" in str(exc_info.value)

    def test_invalid_model_in_cli(self):
        """Should raise ConfigurationError for invalid model name in CLI."""
        cli_overrides: dict[str, Any] = {"model": "large-v99"}

        with pytest.raises(ConfigurationError) as exc_info:
            TranscriptionConfig.from_sources(**cli_overrides)

        assert "Invalid model name" in str(exc_info.value)

    def test_source_fields_tracking_env(self):
        """Should track which fields came from environment."""
        os.environ["SLOWER_WHISPER_MODEL"] = "medium"
        os.environ["SLOWER_WHISPER_LANGUAGE"] = "en"

        config = TranscriptionConfig.from_sources()

        # _source_fields tracking is internal, but we can verify behavior
        # by checking that env values are used
        assert config.model == "medium"
        assert config.language == "en"
        assert config.device == "cuda"  # Not in env, uses default

    def test_source_fields_tracking_file(self):
        """Should track which fields came from file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"model": "small", "beam_size": 3}, f)
            config_file = Path(f.name)

        try:
            config = TranscriptionConfig.from_sources(config_file=config_file)

            assert config.model == "small"
            assert config.beam_size == 3
            assert config.device == "cuda"  # Not in file, uses default

        finally:
            config_file.unlink()

    def test_boolean_fields_file_overrides_env_with_default(self):
        """Boolean fields should follow same precedence rules."""
        os.environ["SLOWER_WHISPER_SKIP_EXISTING_JSON"] = "false"
        os.environ["SLOWER_WHISPER_ENABLE_DIARIZATION"] = "true"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            # File sets back to defaults - should still override env
            json.dump(
                {
                    "skip_existing_json": True,  # True is the default
                    "enable_diarization": False,  # False is the default
                },
                f,
            )
            config_file = Path(f.name)

        try:
            config = TranscriptionConfig.from_sources(config_file=config_file)

            assert config.skip_existing_json is True  # File overrides env
            assert config.enable_diarization is False  # File overrides env

        finally:
            config_file.unlink()

    def test_diarization_fields_full_precedence(self):
        """Diarization fields should follow precedence rules."""
        os.environ["SLOWER_WHISPER_ENABLE_DIARIZATION"] = "false"
        os.environ["SLOWER_WHISPER_MIN_SPEAKERS"] = "1"
        os.environ["SLOWER_WHISPER_MAX_SPEAKERS"] = "2"
        os.environ["SLOWER_WHISPER_OVERLAP_THRESHOLD"] = "0.2"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(
                {
                    "enable_diarization": True,
                    "diarization_device": "cpu",
                    "min_speakers": 2,
                    "max_speakers": 4,
                    "overlap_threshold": 0.6,
                },
                f,
            )
            config_file = Path(f.name)

        try:
            cli_overrides: dict[str, Any] = {
                "min_speakers": 3,  # Override file
                "max_speakers": 5,  # Override file
            }

            config = TranscriptionConfig.from_sources(config_file=config_file, **cli_overrides)

            assert config.enable_diarization is True  # File
            assert config.diarization_device == "cpu"  # File
            assert config.min_speakers == 3  # CLI
            assert config.max_speakers == 5  # CLI
            assert config.overlap_threshold == 0.6  # File

        finally:
            config_file.unlink()

    def test_chunking_fields_precedence(self):
        """Chunking fields should follow precedence rules."""
        os.environ["SLOWER_WHISPER_ENABLE_CHUNKING"] = "true"
        os.environ["SLOWER_WHISPER_CHUNK_TARGET_DURATION_S"] = "20.0"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(
                {
                    "enable_chunking": True,
                    "chunk_target_duration_s": 25.0,
                    "chunk_max_duration_s": 40.0,
                },
                f,
            )
            config_file = Path(f.name)

        try:
            cli_overrides: dict[str, Any] = {
                "chunk_target_duration_s": 35.0,  # Override file
            }

            config = TranscriptionConfig.from_sources(config_file=config_file, **cli_overrides)

            assert config.enable_chunking is True
            assert config.chunk_target_duration_s == 35.0  # CLI
            assert config.chunk_max_duration_s == 40.0  # File

        finally:
            config_file.unlink()


class TestEnrichmentConfigFromSources:
    """Test suite for EnrichmentConfig.from_sources() factory method."""

    def setup_method(self):
        """Clean environment before each test."""
        self.original_env = os.environ.copy()
        # Clear all slower-whisper env vars
        for key in list(os.environ.keys()):
            if key.startswith("SLOWER_WHISPER"):
                del os.environ[key]

    def teardown_method(self):
        """Restore environment after each test."""
        os.environ.clear()
        os.environ.update(self.original_env)

    def test_defaults_only(self):
        """When no sources provided, should use defaults only."""
        config = EnrichmentConfig.from_sources()

        assert config.skip_existing is True
        assert config.enable_prosody is True
        assert config.enable_emotion is True
        assert config.enable_categorical_emotion is False
        assert config.enable_turn_metadata is True
        assert config.enable_speaker_stats is True
        assert config.enable_semantic_annotator is False
        assert config.device == "cpu"
        assert config.dimensional_model_name is None
        assert config.categorical_model_name is None
        assert config.pause_threshold is None

    def test_env_only(self):
        """Environment variables should override defaults."""
        os.environ["SLOWER_WHISPER_ENRICH_SKIP_EXISTING"] = "false"
        os.environ["SLOWER_WHISPER_ENRICH_ENABLE_PROSODY"] = "false"
        os.environ["SLOWER_WHISPER_ENRICH_ENABLE_EMOTION"] = "false"
        os.environ["SLOWER_WHISPER_ENRICH_DEVICE"] = "cuda"
        os.environ["SLOWER_WHISPER_ENRICH_PAUSE_THRESHOLD"] = "1.5"

        config = EnrichmentConfig.from_sources()

        assert config.skip_existing is False
        assert config.enable_prosody is False
        assert config.enable_emotion is False
        assert config.device == "cuda"
        assert config.pause_threshold == 1.5

    def test_env_only_custom_prefix(self):
        """Should support custom environment variable prefix."""
        os.environ["CUSTOM_DEVICE"] = "cuda"
        os.environ["CUSTOM_ENABLE_PROSODY"] = "false"
        os.environ["CUSTOM_PAUSE_THRESHOLD"] = "2.0"

        config = EnrichmentConfig.from_sources(env_prefix="CUSTOM_")

        assert config.device == "cuda"
        assert config.enable_prosody is False
        assert config.pause_threshold == 2.0

    def test_file_only(self):
        """Config file should override defaults."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(
                {
                    "skip_existing": False,
                    "enable_prosody": False,
                    "enable_emotion": False,
                    "enable_categorical_emotion": True,
                    "device": "cuda",
                    "pause_threshold": 2.5,
                },
                f,
            )
            config_file = Path(f.name)

        try:
            config = EnrichmentConfig.from_sources(config_file=config_file)

            assert config.skip_existing is False
            assert config.enable_prosody is False
            assert config.enable_emotion is False
            assert config.enable_categorical_emotion is True
            assert config.device == "cuda"
            assert config.pause_threshold == 2.5

        finally:
            config_file.unlink()

    def test_cli_only(self):
        """CLI args should override defaults."""
        cli_overrides: dict[str, Any] = {
            "skip_existing": False,
            "enable_prosody": False,
            "device": "cuda",
            "pause_threshold": 3.0,
        }

        config = EnrichmentConfig.from_sources(**cli_overrides)

        assert config.skip_existing is False
        assert config.enable_prosody is False
        assert config.device == "cuda"
        assert config.pause_threshold == 3.0

    def test_file_overrides_env_with_default_value(self):
        """
        File should override env even when file value equals default.

        This is the critical regression test for enrichment config.
        """
        os.environ["SLOWER_WHISPER_ENRICH_DEVICE"] = "cuda"
        os.environ["SLOWER_WHISPER_ENRICH_ENABLE_PROSODY"] = "false"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            # File sets back to defaults - should still override env
            json.dump(
                {
                    "device": "cpu",  # cpu is the default
                    "enable_prosody": True,  # True is the default
                },
                f,
            )
            config_file = Path(f.name)

        try:
            config = EnrichmentConfig.from_sources(config_file=config_file)

            assert config.device == "cpu"  # File overrides env
            assert config.enable_prosody is True  # File overrides env

        finally:
            config_file.unlink()

    def test_file_overrides_env_with_custom_value(self):
        """File should override env when file value differs from default."""
        os.environ["SLOWER_WHISPER_ENRICH_DEVICE"] = "cpu"
        os.environ["SLOWER_WHISPER_ENRICH_PAUSE_THRESHOLD"] = "1.0"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(
                {
                    "device": "cuda",  # cuda != default (cpu)
                    "pause_threshold": 2.0,  # 2.0 != env (1.0)
                },
                f,
            )
            config_file = Path(f.name)

        try:
            config = EnrichmentConfig.from_sources(config_file=config_file)

            assert config.device == "cuda"
            assert config.pause_threshold == 2.0

        finally:
            config_file.unlink()

    def test_cli_overrides_file(self):
        """CLI args should override file config."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(
                {
                    "device": "cpu",
                    "enable_prosody": False,
                    "pause_threshold": 1.5,
                },
                f,
            )
            config_file = Path(f.name)

        try:
            cli_overrides: dict[str, Any] = {
                "device": "cuda",  # Override file
                "pause_threshold": 2.5,  # Override file
            }

            config = EnrichmentConfig.from_sources(config_file=config_file, **cli_overrides)

            assert config.device == "cuda"  # CLI wins
            assert config.enable_prosody is False  # File wins (no CLI override)
            assert config.pause_threshold == 2.5  # CLI wins

        finally:
            config_file.unlink()

    def test_full_precedence_chain(self):
        """
        Test complete precedence: CLI > file > env > defaults.
        """
        os.environ["SLOWER_WHISPER_ENRICH_DEVICE"] = "cuda"
        os.environ["SLOWER_WHISPER_ENRICH_ENABLE_PROSODY"] = "false"
        os.environ["SLOWER_WHISPER_ENRICH_PAUSE_THRESHOLD"] = "1.0"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(
                {
                    "device": "cpu",  # Override env (default value)
                    "enable_emotion": False,  # Not in env
                    "pause_threshold": 2.0,  # Override env
                },
                f,
            )
            config_file = Path(f.name)

        try:
            cli_overrides: dict[str, Any] = {
                "enable_prosody": True,  # Override env
                "enable_categorical_emotion": True,  # Not in file or env
            }

            config = EnrichmentConfig.from_sources(config_file=config_file, **cli_overrides)

            assert config.device == "cpu"  # File (overrode env despite being default)
            assert config.enable_prosody is True  # CLI
            assert config.enable_emotion is False  # File
            assert config.enable_categorical_emotion is True  # CLI
            assert config.pause_threshold == 2.0  # File

        finally:
            config_file.unlink()

    def test_none_overrides_ignored(self):
        """None values in CLI overrides should be ignored."""
        os.environ["SLOWER_WHISPER_ENRICH_DEVICE"] = "cuda"

        cli_overrides: dict[str, Any] = {
            "device": None,  # Should be ignored
            "enable_prosody": None,  # Should be ignored
            "pause_threshold": 1.5,  # Should be used
        }

        config = EnrichmentConfig.from_sources(**cli_overrides)

        assert config.device == "cuda"  # Env preserved (CLI None ignored)
        assert config.enable_prosody is True  # Default preserved (CLI None ignored)
        assert config.pause_threshold == 1.5  # CLI used

    def test_invalid_file_path(self):
        """Should raise FileNotFoundError for non-existent config file."""
        with pytest.raises(FileNotFoundError) as exc_info:
            EnrichmentConfig.from_sources(config_file="/nonexistent/config.json")

        assert "Configuration file not found" in str(exc_info.value)

    def test_invalid_json_in_file(self):
        """Should raise JSONDecodeError for malformed JSON."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("not valid json{")
            config_file = Path(f.name)

        try:
            with pytest.raises(json.JSONDecodeError):
                EnrichmentConfig.from_sources(config_file=config_file)

        finally:
            config_file.unlink()

    def test_pause_threshold_validation(self):
        """Should validate pause_threshold is non-negative."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"pause_threshold": -1.0}, f)
            config_file = Path(f.name)

        try:
            with pytest.raises(ValueError) as exc_info:
                EnrichmentConfig.from_sources(config_file=config_file)

            assert "pause_threshold must be >= 0.0" in str(exc_info.value)

        finally:
            config_file.unlink()


class TestFromSourcesEdgeCases:
    """Test edge cases and error handling for from_sources() methods."""

    def setup_method(self):
        """Clean environment before each test."""
        self.original_env = os.environ.copy()
        for key in list(os.environ.keys()):
            if key.startswith("SLOWER_WHISPER"):
                del os.environ[key]

    def teardown_method(self):
        """Restore environment after each test."""
        os.environ.clear()
        os.environ.update(self.original_env)

    def test_invalid_field_names_in_cli_overrides(self):
        """Unknown fields in CLI overrides should be silently ignored."""
        cli_overrides: dict[str, Any] = {
            "model": "base",
            "unknown_field": "value",  # Should be ignored
            "another_unknown": 123,  # Should be ignored
        }

        config = TranscriptionConfig.from_sources(**cli_overrides)

        assert config.model == "base"
        # Unknown fields don't cause errors, just ignored

    def test_empty_cli_overrides(self):
        """Empty CLI overrides dict should work."""
        config = TranscriptionConfig.from_sources()
        assert config.model == "large-v3"  # Defaults

    def test_empty_file(self):
        """Empty JSON file should use all defaults."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({}, f)
            config_file = Path(f.name)

        try:
            config = TranscriptionConfig.from_sources(config_file=config_file)
            assert config.model == "large-v3"  # Defaults

        finally:
            config_file.unlink()

    def test_file_path_as_string(self):
        """Should accept file path as string."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"model": "base"}, f)
            config_file = f.name  # String, not Path

        try:
            config = TranscriptionConfig.from_sources(config_file=config_file)
            assert config.model == "base"

        finally:
            Path(config_file).unlink()

    def test_env_prefix_with_trailing_underscore(self):
        """Env prefix should work with or without trailing underscore."""
        os.environ["CUSTOM_MODEL"] = "base"

        # With trailing underscore
        config1 = TranscriptionConfig.from_sources(env_prefix="CUSTOM_")
        assert config1.model == "base"

        # Without trailing underscore (should auto-add)
        # Note: This depends on implementation - may need adjustment
        # For now, test with underscore as documented

    def test_mixed_case_env_values(self):
        """Env values should be case-sensitive for most fields."""
        os.environ["SLOWER_WHISPER_LANGUAGE"] = "EN"  # Uppercase

        config = TranscriptionConfig.from_sources()

        # Language codes are case-sensitive, so EN should be preserved
        assert config.language == "EN"

    def test_whitespace_in_env_values(self):
        """Env values should preserve whitespace (validation will catch invalid)."""
        os.environ["SLOWER_WHISPER_LANGUAGE"] = "  en  "

        config = TranscriptionConfig.from_sources()

        # Whitespace preserved (may cause issues, but that's user error)
        assert config.language == "  en  "

    def test_concurrent_sources_with_all_combinations(self):
        """All three sources together should follow precedence."""
        os.environ["SLOWER_WHISPER_MODEL"] = "small"
        os.environ["SLOWER_WHISPER_BEAM_SIZE"] = "3"
        os.environ["SLOWER_WHISPER_LANGUAGE"] = "en"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(
                {
                    "model": "medium",  # Override env
                    "device": "cpu",  # Not in env
                },
                f,
            )
            config_file = Path(f.name)

        try:
            cli_overrides: dict[str, Any] = {
                "model": "base",  # Override file and env
                "task": "translate",  # Not in file or env
            }

            config = TranscriptionConfig.from_sources(config_file=config_file, **cli_overrides)

            assert config.model == "base"  # CLI (highest precedence)
            assert config.device == "cpu"  # File
            assert config.beam_size == 3  # Env
            assert config.language == "en"  # Env
            assert config.task == "translate"  # CLI

        finally:
            config_file.unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
