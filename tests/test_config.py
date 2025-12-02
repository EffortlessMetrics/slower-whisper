"""Tests for configuration validation and model name checking.

This module tests the model name validation and configuration error handling
in transcription.config module.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from transcription.config import (
    ALLOWED_WHISPER_MODELS,
    AsrConfig,
    TranscriptionConfig,
    validate_model_name,
)
from transcription.exceptions import ConfigurationError


class TestModelNameValidation:
    """Test suite for Whisper model name validation."""

    def test_validate_model_name_with_valid_models(self):
        """All allowed model names should pass validation."""
        for model in ALLOWED_WHISPER_MODELS:
            # Should not raise
            result = validate_model_name(model)
            assert result == model

    def test_validate_model_name_with_invalid_model(self):
        """Invalid model names should raise ConfigurationError."""
        invalid_models = [
            "large",  # Missing version
            "large-v4",  # Non-existent version
            "xlarge",  # Non-existent size
            "tiny-v3",  # Wrong format
            "LARGE-V3",  # Case-sensitive
            "",  # Empty string
            "gpt-4",  # Wrong model type
        ]

        for invalid_model in invalid_models:
            with pytest.raises(ConfigurationError) as exc_info:
                validate_model_name(invalid_model)

            error_msg = str(exc_info.value)
            assert "Invalid model name" in error_msg
            assert invalid_model in error_msg
            assert "Must be one of:" in error_msg

    def test_validation_error_includes_all_allowed_models(self):
        """Error message should list all allowed models."""
        with pytest.raises(ConfigurationError) as exc_info:
            validate_model_name("invalid-model")

        error_msg = str(exc_info.value)
        # Check that common models are mentioned
        assert "tiny" in error_msg
        assert "base" in error_msg
        assert "small" in error_msg
        assert "medium" in error_msg
        assert "large-v3" in error_msg


class TestTranscriptionConfigValidation:
    """Test model name validation in TranscriptionConfig."""

    def test_config_with_valid_model(self):
        """TranscriptionConfig should accept valid model names."""
        for model in ["tiny", "base", "small", "medium", "large-v3", "large-v3-turbo"]:
            config = TranscriptionConfig(model=model)
            assert config.model == model

    def test_config_with_invalid_model(self):
        """TranscriptionConfig should reject invalid model names."""
        with pytest.raises(ConfigurationError) as exc_info:
            TranscriptionConfig(model="invalid-model")

        assert "Invalid model name" in str(exc_info.value)

    def test_config_default_model_is_valid(self):
        """Default model should be valid."""
        config = TranscriptionConfig()
        assert config.model == "large-v3"
        # Should not raise when validating
        validate_model_name(config.model)

    def test_config_from_file_with_valid_model(self):
        """Loading config from file should validate model name."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"model": "medium"}, f)
            config_file = Path(f.name)

        try:
            config = TranscriptionConfig.from_file(config_file)
            assert config.model == "medium"
        finally:
            config_file.unlink()

    def test_config_from_file_with_invalid_model(self):
        """Loading config from file should reject invalid model names."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"model": "large-v4"}, f)
            config_file = Path(f.name)

        try:
            with pytest.raises(ValueError) as exc_info:
                TranscriptionConfig.from_file(config_file)

            assert "Invalid model name" in str(exc_info.value)
            assert "large-v4" in str(exc_info.value)
        finally:
            config_file.unlink()

    def test_config_from_env_with_valid_model(self):
        """Loading config from environment should validate model name."""
        original_env = os.environ.copy()
        try:
            os.environ["SLOWER_WHISPER_MODEL"] = "small"
            config = TranscriptionConfig.from_env()
            assert config.model == "small"
        finally:
            os.environ.clear()
            os.environ.update(original_env)

    def test_config_from_env_with_invalid_model(self):
        """Loading config from environment should reject invalid model names."""
        original_env = os.environ.copy()
        try:
            os.environ["SLOWER_WHISPER_MODEL"] = "gpt-4"

            with pytest.raises(ValueError) as exc_info:
                TranscriptionConfig.from_env()

            assert "Invalid model name" in str(exc_info.value)
            assert "gpt-4" in str(exc_info.value)
        finally:
            os.environ.clear()
            os.environ.update(original_env)

    def test_allowed_models_constant_matches_validation(self):
        """ALLOWED_WHISPER_MODELS constant should match validation behavior."""
        # All models in the constant should validate
        for model in ALLOWED_WHISPER_MODELS:
            validate_model_name(model)  # Should not raise

        # A model not in the constant should fail
        with pytest.raises(ConfigurationError):
            validate_model_name("not-in-constant")


class TestAsrConfigValidation:
    """Test model name validation in legacy AsrConfig."""

    def test_asr_config_with_valid_model(self):
        """AsrConfig should accept valid model names."""
        for model in ["tiny", "base", "small", "medium", "large-v3"]:
            config = AsrConfig(model_name=model)
            assert config.model_name == model

    def test_asr_config_with_invalid_model(self):
        """AsrConfig should reject invalid model names."""
        with pytest.raises(ConfigurationError) as exc_info:
            AsrConfig(model_name="large-v99")

        assert "Invalid model name" in str(exc_info.value)

    def test_asr_config_default_model_is_valid(self):
        """AsrConfig default model should be valid."""
        config = AsrConfig()
        assert config.model_name == "large-v3"
        validate_model_name(config.model_name)  # Should not raise


class TestModelNameErrorMessages:
    """Test that error messages are helpful and informative."""

    def test_error_message_format(self):
        """Error message should be clear and actionable."""
        with pytest.raises(ConfigurationError) as exc_info:
            validate_model_name("unknown-model")

        error_msg = str(exc_info.value)

        # Check message structure
        assert error_msg.startswith("Invalid model name")
        assert "unknown-model" in error_msg
        assert "Must be one of:" in error_msg

        # Check that models are listed in sorted order
        models_str = error_msg.split("Must be one of:")[1].strip()
        models_list = [m.strip() for m in models_str.split(",")]

        # Verify sorted order
        assert models_list == sorted(models_list)

    def test_error_from_config_file_is_clear(self):
        """Error from config file should clearly indicate the problem."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"model": "wrong-model", "device": "cuda"}, f)
            config_file = Path(f.name)

        try:
            with pytest.raises(ValueError) as exc_info:
                TranscriptionConfig.from_file(config_file)

            error_msg = str(exc_info.value)
            assert "Invalid model name" in error_msg
            assert "wrong-model" in error_msg
        finally:
            config_file.unlink()

    def test_error_from_env_is_clear(self):
        """Error from environment variable should clearly indicate the problem."""
        original_env = os.environ.copy()
        try:
            os.environ["SLOWER_WHISPER_MODEL"] = "bad-model"

            with pytest.raises(ValueError) as exc_info:
                TranscriptionConfig.from_env()

            error_msg = str(exc_info.value)
            assert "Invalid model name" in error_msg
            assert "bad-model" in error_msg
        finally:
            os.environ.clear()
            os.environ.update(original_env)


class TestAllowedModelsConstant:
    """Test the ALLOWED_WHISPER_MODELS constant."""

    def test_all_expected_models_present(self):
        """Verify all expected Whisper models are in the allowed list."""
        expected_models = {
            "tiny",
            "base",
            "small",
            "medium",
            "large-v1",
            "large-v2",
            "large-v3",
            "large-v3-turbo",
        }

        assert ALLOWED_WHISPER_MODELS == expected_models

    def test_models_are_strings(self):
        """All model names should be strings."""
        for model in ALLOWED_WHISPER_MODELS:
            assert isinstance(model, str)

    def test_models_are_lowercase(self):
        """All model names should be lowercase (convention)."""
        for model in ALLOWED_WHISPER_MODELS:
            assert model == model.lower()


class TestInvalidModelNameRaisesError:
    """Test that invalid model names raise ConfigurationError appropriately."""

    def test_invalid_model_name_raises_error(self):
        """Test typo like 'large-v4' raises ConfigurationError."""
        invalid_typos = [
            "large-v4",  # Non-existent version (off-by-one typo)
            "large-v0",  # Wrong version number
            "large-v3-turob",  # Typo in "turbo"
            "xlarge",  # Typo/non-existent size
            "medium-v2",  # Medium doesn't have versions
        ]

        for invalid_model in invalid_typos:
            with pytest.raises(ConfigurationError) as exc_info:
                validate_model_name(invalid_model)

            error_msg = str(exc_info.value)
            assert "Invalid model name" in error_msg
            assert invalid_model in error_msg
            assert "Must be one of:" in error_msg

    def test_invalid_model_names_comprehensive(self):
        """Test a comprehensive set of invalid model names."""
        invalid_models = [
            "large",  # Missing version
            "large-v4",  # Non-existent version
            "xlarge",  # Non-existent size
            "tiny-v3",  # Wrong format
            "LARGE-V3",  # Case-sensitive (uppercase not allowed)
            "large-V3",  # Case-sensitive (mixed case)
            "",  # Empty string
            "gpt-4",  # Wrong model type
            "whisper-large",  # Wrong naming convention
            "large_v3",  # Underscore instead of hyphen
            "large- v3",  # Space in name
            "123",  # Numeric only
            "large-v3 ",  # Trailing space
            " large-v3",  # Leading space
        ]

        for invalid_model in invalid_models:
            with pytest.raises(ConfigurationError) as exc_info:
                validate_model_name(invalid_model)

            assert "Invalid model name" in str(exc_info.value)

    def test_model_validation_error_contains_suggestions(self):
        """Verify error message lists allowed models for user guidance."""
        with pytest.raises(ConfigurationError) as exc_info:
            validate_model_name("large-v4")

        error_msg = str(exc_info.value)
        # Should suggest valid alternatives
        assert "large-v3" in error_msg
        assert "large-v1" in error_msg
        assert "large-v2" in error_msg


class TestValidModelNamesAccepted:
    """Test that all valid model names are properly accepted."""

    def test_valid_model_names_accepted(self):
        """Test all allowed models pass validation."""
        valid_models = [
            "tiny",
            "base",
            "small",
            "medium",
            "large-v1",
            "large-v2",
            "large-v3",
            "large-v3-turbo",
        ]

        for model in valid_models:
            # Should not raise
            result = validate_model_name(model)
            assert result == model, f"Model {model} should validate and return unchanged"

    def test_valid_models_match_allowed_constant(self):
        """Verify all ALLOWED_WHISPER_MODELS validate successfully."""
        for model in ALLOWED_WHISPER_MODELS:
            result = validate_model_name(model)
            assert result == model

    def test_transcription_config_accepts_all_valid_models(self):
        """TranscriptionConfig should accept all allowed model names."""
        for model in ALLOWED_WHISPER_MODELS:
            config = TranscriptionConfig(model=model)
            assert config.model == model

    def test_asr_config_accepts_all_valid_models(self):
        """AsrConfig (legacy) should accept all allowed model names."""
        for model in ALLOWED_WHISPER_MODELS:
            config = AsrConfig(model_name=model)
            assert config.model_name == model


class TestModelNameValidationInFromFile:
    """Test model name validation when loading from config files."""

    def test_model_name_validation_in_from_file(self):
        """Validate model name when loading TranscriptionConfig from file."""
        # Test valid model
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"model": "large-v3"}, f)
            config_file = Path(f.name)

        try:
            config = TranscriptionConfig.from_file(config_file)
            assert config.model == "large-v3"
        finally:
            config_file.unlink()

        # Test invalid model
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"model": "large-v4"}, f)
            config_file = Path(f.name)

        try:
            with pytest.raises(ValueError) as exc_info:
                TranscriptionConfig.from_file(config_file)
            assert "Invalid model name" in str(exc_info.value)
            assert "large-v4" in str(exc_info.value)
        finally:
            config_file.unlink()

    def test_from_file_with_all_valid_models(self):
        """Test loading all valid models from file."""
        for model in ALLOWED_WHISPER_MODELS:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                json.dump({"model": model}, f)
                config_file = Path(f.name)

            try:
                config = TranscriptionConfig.from_file(config_file)
                assert config.model == model
            finally:
                config_file.unlink()

    def test_from_file_with_invalid_models_comprehensive(self):
        """Test that from_file rejects various invalid models."""
        invalid_models = [
            "large-v4",
            "xlarge",
            "LARGE-V3",
            "large- v3",
        ]

        for invalid_model in invalid_models:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                json.dump({"model": invalid_model}, f)
                config_file = Path(f.name)

            try:
                with pytest.raises(ValueError) as exc_info:
                    TranscriptionConfig.from_file(config_file)
                assert "Invalid model name" in str(exc_info.value)
            finally:
                config_file.unlink()

    def test_from_file_with_other_valid_fields(self):
        """Test model validation works with other config fields."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(
                {"model": "small", "device": "cpu", "language": "en", "task": "transcribe"}, f
            )
            config_file = Path(f.name)

        try:
            config = TranscriptionConfig.from_file(config_file)
            assert config.model == "small"
            assert config.device == "cpu"
            assert config.language == "en"
        finally:
            config_file.unlink()

    def test_from_file_missing_model_uses_default(self):
        """When model is omitted from file, default should be used."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"device": "cuda"}, f)
            config_file = Path(f.name)

        try:
            config = TranscriptionConfig.from_file(config_file)
            assert config.model == "large-v3"  # Default value
        finally:
            config_file.unlink()


class TestModelNameValidationInFromEnv:
    """Test model name validation when loading from environment variables."""

    def test_model_name_validation_in_from_env(self):
        """Validate model name when loading TranscriptionConfig from environment."""
        original_env = os.environ.copy()
        try:
            # Test valid model
            os.environ["SLOWER_WHISPER_MODEL"] = "base"
            config = TranscriptionConfig.from_env()
            assert config.model == "base"

            # Test invalid model
            os.environ["SLOWER_WHISPER_MODEL"] = "large-v4"
            with pytest.raises(ValueError) as exc_info:
                TranscriptionConfig.from_env()
            assert "Invalid model name" in str(exc_info.value)
            assert "large-v4" in str(exc_info.value)
        finally:
            os.environ.clear()
            os.environ.update(original_env)

    def test_from_env_with_all_valid_models(self):
        """Test loading all valid models from environment."""
        original_env = os.environ.copy()
        try:
            for model in ALLOWED_WHISPER_MODELS:
                os.environ["SLOWER_WHISPER_MODEL"] = model
                config = TranscriptionConfig.from_env()
                assert config.model == model
                # Clear for next iteration
                del os.environ["SLOWER_WHISPER_MODEL"]
        finally:
            os.environ.clear()
            os.environ.update(original_env)

    def test_from_env_with_invalid_models_comprehensive(self):
        """Test that from_env rejects various invalid models."""
        invalid_models = [
            "large-v4",
            "xlarge",
            "LARGE-V3",
            "whisper-large",
        ]

        original_env = os.environ.copy()
        try:
            for invalid_model in invalid_models:
                os.environ["SLOWER_WHISPER_MODEL"] = invalid_model
                with pytest.raises(ValueError) as exc_info:
                    TranscriptionConfig.from_env()
                assert "Invalid model name" in str(exc_info.value)
                # Clean up for next iteration
                del os.environ["SLOWER_WHISPER_MODEL"]
        finally:
            os.environ.clear()
            os.environ.update(original_env)

    def test_from_env_with_other_valid_settings(self):
        """Test model validation works with other env vars."""
        original_env = os.environ.copy()
        try:
            os.environ["SLOWER_WHISPER_MODEL"] = "medium"
            os.environ["SLOWER_WHISPER_DEVICE"] = "cuda"
            os.environ["SLOWER_WHISPER_LANGUAGE"] = "en"
            os.environ["SLOWER_WHISPER_TASK"] = "transcribe"

            config = TranscriptionConfig.from_env()
            assert config.model == "medium"
            assert config.device == "cuda"
            assert config.language == "en"
            assert config.task == "transcribe"
        finally:
            os.environ.clear()
            os.environ.update(original_env)

    def test_from_env_missing_model_uses_default(self):
        """When SLOWER_WHISPER_MODEL is not set, default should be used."""
        original_env = os.environ.copy()
        try:
            # Make sure the model env var is not set
            os.environ.pop("SLOWER_WHISPER_MODEL", None)
            config = TranscriptionConfig.from_env()
            assert config.model == "large-v3"  # Default value
        finally:
            os.environ.clear()
            os.environ.update(original_env)

    def test_from_env_error_message_is_informative(self):
        """Error message from env validation should be clear."""
        original_env = os.environ.copy()
        try:
            os.environ["SLOWER_WHISPER_MODEL"] = "gpt-4"
            with pytest.raises(ValueError) as exc_info:
                TranscriptionConfig.from_env()

            error_msg = str(exc_info.value)
            assert "Invalid model name" in error_msg
            assert "gpt-4" in error_msg
            assert "Must be one of:" in error_msg
        finally:
            os.environ.clear()
            os.environ.update(original_env)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
