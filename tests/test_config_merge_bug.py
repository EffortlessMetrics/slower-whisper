"""
Tests for configuration merge bug fix.

This module tests the fix for the critical bug where config file values
that equal defaults would be incorrectly overridden by environment variables.

Bug description:
- When a config file sets a value that happens to equal the default (e.g., device="cuda"),
  the old merge logic would incorrectly use the environment variable instead.
- This violated the documented precedence: CLI > file > env > defaults

Fix:
- Added _source_fields tracking to TranscriptionConfig and EnrichmentConfig
- Modified _merge_configs() and _merge_enrich_configs() to check _source_fields
- Only fields explicitly present in the source override lower-precedence values
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from transcription.cli import _config_from_transcribe_args, _merge_configs, _merge_enrich_configs
from transcription.config import AsrConfig, EnrichmentConfig, TranscriptionConfig


class TestConfigMergeBugFix:
    """Test suite for configuration merge bug fix."""

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

    def test_env_loads_diarization_fields(self):
        """Environment variables should populate diarization settings."""
        os.environ["SLOWER_WHISPER_ENABLE_DIARIZATION"] = "true"
        os.environ["SLOWER_WHISPER_DIARIZATION_DEVICE"] = "cpu"
        os.environ["SLOWER_WHISPER_MIN_SPEAKERS"] = "2"
        os.environ["SLOWER_WHISPER_MAX_SPEAKERS"] = "4"
        os.environ["SLOWER_WHISPER_OVERLAP_THRESHOLD"] = "0.45"

        env_config = TranscriptionConfig.from_env()

        assert env_config.enable_diarization is True
        assert env_config.diarization_device == "cpu"
        assert env_config.min_speakers == 2
        assert env_config.max_speakers == 4
        assert env_config.overlap_threshold == 0.45
        assert {
            "enable_diarization",
            "diarization_device",
            "min_speakers",
            "max_speakers",
            "overlap_threshold",
        }.issubset(env_config._source_fields)

    def test_file_overrides_env_when_value_equals_default(self):
        """
        Test the main bug: file should override env even when value equals default.

        This was the core bug - when a config file explicitly set device="cuda"
        (which is the default), it would be ignored in favor of the env var.
        """
        # Set env to non-default value
        os.environ["SLOWER_WHISPER_DEVICE"] = "cpu"

        # Create temp config file with default value
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"device": "cuda"}, f)  # cuda is the default
            config_file = Path(f.name)

        try:
            # Merge: defaults -> env -> file
            config = TranscriptionConfig()
            env_config = TranscriptionConfig.from_env()
            config = _merge_configs(config, env_config)
            file_config = TranscriptionConfig.from_file(config_file)
            config = _merge_configs(config, file_config)

            # File should win even though its value equals the default
            assert config.device == "cuda", (
                "File should override env even when value equals default"
            )

        finally:
            config_file.unlink()

    def test_file_overrides_env_when_value_differs_from_default(self):
        """Test that file overrides env when value differs from default."""
        os.environ["SLOWER_WHISPER_MODEL"] = "base"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"model": "medium"}, f)  # medium != default (large-v3)
            config_file = Path(f.name)

        try:
            config = TranscriptionConfig()
            env_config = TranscriptionConfig.from_env()
            config = _merge_configs(config, env_config)
            file_config = TranscriptionConfig.from_file(config_file)
            config = _merge_configs(config, file_config)

            assert config.model == "medium"

        finally:
            config_file.unlink()

    def test_env_preserved_when_file_doesnt_set_field(self):
        """Test that env value is preserved when file doesn't mention that field."""
        os.environ["SLOWER_WHISPER_DEVICE"] = "cpu"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"model": "medium"}, f)  # File doesn't set device
            config_file = Path(f.name)

        try:
            config = TranscriptionConfig()
            env_config = TranscriptionConfig.from_env()
            config = _merge_configs(config, env_config)
            file_config = TranscriptionConfig.from_file(config_file)
            config = _merge_configs(config, file_config)

            # Env value should be preserved since file didn't set device
            assert config.device == "cpu"
            # File value should be used
            assert config.model == "medium"

        finally:
            config_file.unlink()

    def test_boolean_field_merge_with_default_value(self):
        """
        Test that boolean fields work correctly with the fix.

        This is important because skip_existing_json defaults to True,
        so a file setting it to True should override an env var setting it to False.
        """
        os.environ["SLOWER_WHISPER_SKIP_EXISTING_JSON"] = "false"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"skip_existing_json": True}, f)  # True is the default
            config_file = Path(f.name)

        try:
            config = TranscriptionConfig()
            env_config = TranscriptionConfig.from_env()
            config = _merge_configs(config, env_config)
            file_config = TranscriptionConfig.from_file(config_file)
            config = _merge_configs(config, file_config)

            # File's True should override env's False, even though True is the default
            assert config.skip_existing_json is True

        finally:
            config_file.unlink()

    def test_multiple_fields_mixed_defaults_and_custom(self):
        """Test complex scenario with multiple fields, some default, some custom."""
        os.environ["SLOWER_WHISPER_MODEL"] = "small"
        os.environ["SLOWER_WHISPER_DEVICE"] = "cpu"
        os.environ["SLOWER_WHISPER_BEAM_SIZE"] = "3"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(
                {
                    "model": "medium",  # Override env
                    "device": "cuda",  # Default value, should still override env
                    "beam_size": 5,  # Default value, should still override env
                },
                f,
            )
            config_file = Path(f.name)

        try:
            config = TranscriptionConfig()
            env_config = TranscriptionConfig.from_env()
            config = _merge_configs(config, env_config)
            file_config = TranscriptionConfig.from_file(config_file)
            config = _merge_configs(config, file_config)

            assert config.model == "medium"
            assert config.device == "cuda"
            assert config.beam_size == 5

        finally:
            config_file.unlink()

    def test_source_fields_tracking_from_file(self):
        """Test that _source_fields correctly tracks which fields came from file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"model": "medium", "device": "cpu"}, f)
            config_file = Path(f.name)

        try:
            file_config = TranscriptionConfig.from_file(config_file)

            assert hasattr(file_config, "_source_fields")
            assert "model" in file_config._source_fields
            assert "device" in file_config._source_fields
            assert "language" not in file_config._source_fields

        finally:
            config_file.unlink()

    def test_source_fields_tracking_from_env(self):
        """Test that _source_fields correctly tracks which fields came from env."""
        os.environ["SLOWER_WHISPER_MODEL"] = "base"
        os.environ["SLOWER_WHISPER_LANGUAGE"] = "en"

        env_config = TranscriptionConfig.from_env()

        assert hasattr(env_config, "_source_fields")
        assert "model" in env_config._source_fields
        assert "language" in env_config._source_fields
        assert "device" not in env_config._source_fields

    def test_enrichment_config_merge_bug_fix(self):
        """Test that EnrichmentConfig has the same bug fix."""
        os.environ["SLOWER_WHISPER_ENRICH_DEVICE"] = "cuda"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"device": "cpu"}, f)  # cpu is the default for EnrichmentConfig
            config_file = Path(f.name)

        try:
            config = EnrichmentConfig()
            env_config = EnrichmentConfig.from_env()
            config = _merge_enrich_configs(config, env_config)
            file_config = EnrichmentConfig.from_file(config_file)
            config = _merge_enrich_configs(config, file_config)

            # File should override env even though value equals default
            assert config.device == "cpu"

        finally:
            config_file.unlink()

    def test_enrichment_boolean_fields_with_defaults(self):
        """Test EnrichmentConfig boolean fields that match defaults."""
        # Set env to override defaults
        os.environ["SLOWER_WHISPER_ENRICH_ENABLE_PROSODY"] = "false"
        os.environ["SLOWER_WHISPER_ENRICH_ENABLE_EMOTION"] = "false"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            # File sets back to defaults - should still override env
            json.dump({"enable_prosody": True, "enable_emotion": True}, f)
            config_file = Path(f.name)

        try:
            config = EnrichmentConfig()
            env_config = EnrichmentConfig.from_env()
            config = _merge_enrich_configs(config, env_config)
            file_config = EnrichmentConfig.from_file(config_file)
            config = _merge_enrich_configs(config, file_config)

            assert config.enable_prosody is True
            assert config.enable_emotion is True

        finally:
            config_file.unlink()

    def test_backward_compatibility_when_source_fields_missing(self):
        """
        Test that merge still works when _source_fields is not present.

        This ensures backward compatibility if someone creates a config manually
        without using from_file() or from_env().
        """
        # Create configs manually (no _source_fields)
        base = TranscriptionConfig(device="cpu", model="large-v3")
        override = TranscriptionConfig(model="medium", device="cpu")

        # Clear _source_fields if they exist
        base._source_fields = set()
        override._source_fields = set()

        # This should fall back to value comparison
        result = _merge_configs(base, override)

        # Since override.model != base.model, it should use override
        assert result.model == "medium"
        # Since override.device == base.device, it should keep base (or override, same value)
        assert result.device == "cpu"

    def test_full_precedence_chain_with_bug_scenario(self):
        """
        Test complete precedence chain: CLI > file > env > defaults

        Specifically test the bug scenario at each level.
        """
        # Env sets device to non-default
        os.environ["SLOWER_WHISPER_DEVICE"] = "cpu"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            # File sets back to default
            json.dump({"device": "cuda", "model": "medium"}, f)
            config_file = Path(f.name)

        try:
            # Simulate CLI args (using TranscriptionConfig directly for simplicity)
            from argparse import Namespace

            from transcription.cli import _config_from_transcribe_args

            args = Namespace(
                config=config_file,
                model="base",  # CLI override
                device=None,  # No CLI override
                language=None,
                task=None,
                compute_type=None,
                vad_min_silence_ms=None,
                beam_size=None,
                skip_existing_json=None,
                enable_diarization=None,  # Diarization fields added in v1.1
                diarization_device=None,
                min_speakers=None,
                max_speakers=None,
                overlap_threshold=None,
            )

            config = _config_from_transcribe_args(args)

            # CLI overrides file
            assert config.model == "base"
            # File overrides env (even though file value equals default)
            assert config.device == "cuda"

        finally:
            config_file.unlink()

    def test_compute_type_defaults_follow_final_device(self):
        """
        compute_type should follow the final device when not explicitly set.

        If env forces device=cpu (which makes compute_type=int8 by default) but the CLI
        overrides device to cuda, compute_type should return to the CUDA default (float16).
        """
        os.environ["SLOWER_WHISPER_DEVICE"] = "cpu"

        from argparse import Namespace

        args = Namespace(
            config=None,
            model=None,
            device="cuda",  # CLI override
            language=None,
            task=None,
            compute_type=None,  # Not explicitly set anywhere
            vad_min_silence_ms=None,
            beam_size=None,
            skip_existing_json=None,
            enable_diarization=None,
            diarization_device=None,
            min_speakers=None,
            max_speakers=None,
            overlap_threshold=None,
        )

        config = _config_from_transcribe_args(args)

        assert config.device == "cuda"
        assert config.compute_type == "float16"

    def test_explicit_compute_type_preserved(self):
        """
        Explicit compute_type should not be auto-overridden based on device.

        Regression test for bug where compute_type was forced to int8 on CPU even
        when explicitly set to float16.
        """
        cfg = TranscriptionConfig(device="cpu", compute_type="float16")
        assert cfg.compute_type == "float16"

        asr_cfg = AsrConfig(device="cpu", compute_type="float16")
        assert asr_cfg.compute_type == "float16"

    def test_diarization_file_overrides_env_and_preserves_threshold(self):
        """
        File diarization settings should override env and survive CLI rebuild.

        This guards against regression where overlap_threshold was dropped when
        reconstructing the TranscriptionConfig in _config_from_transcribe_args.
        """
        os.environ["SLOWER_WHISPER_ENABLE_DIARIZATION"] = "false"
        os.environ["SLOWER_WHISPER_MIN_SPEAKERS"] = "1"
        os.environ["SLOWER_WHISPER_MAX_SPEAKERS"] = "1"
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
            from argparse import Namespace

            args = Namespace(
                config=config_file,
                model=None,
                device=None,
                language=None,
                task=None,
                compute_type=None,
                vad_min_silence_ms=None,
                beam_size=None,
                skip_existing_json=None,
                enable_diarization=None,
                diarization_device=None,
                min_speakers=None,
                max_speakers=None,
                overlap_threshold=None,
            )

            config = _config_from_transcribe_args(args)

            assert config.enable_diarization is True
            assert config.diarization_device == "cpu"
            assert config.min_speakers == 2
            assert config.max_speakers == 4
            assert config.overlap_threshold == 0.6

        finally:
            config_file.unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
