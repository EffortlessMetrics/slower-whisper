"""Environment boolean values must be explicit and diagnosable."""

import pytest

from transcription.config import EnrichmentConfig, TranscriptionConfig


@pytest.mark.parametrize(
    ("config_type", "suffix"),
    [
        (TranscriptionConfig, "SKIP_EXISTING_JSON"),
        (TranscriptionConfig, "WORD_TIMESTAMPS"),
        (TranscriptionConfig, "ENABLE_CHUNKING"),
        (TranscriptionConfig, "ENABLE_DIARIZATION"),
        (EnrichmentConfig, "SKIP_EXISTING"),
        (EnrichmentConfig, "ENABLE_PROSODY"),
        (EnrichmentConfig, "ENABLE_EMOTION"),
        (EnrichmentConfig, "ENABLE_CATEGORICAL_EMOTION"),
        (EnrichmentConfig, "ENABLE_TURN_METADATA"),
        (EnrichmentConfig, "ENABLE_SPEAKER_STATS"),
        (EnrichmentConfig, "ENABLE_SEMANTIC_ANNOTATOR"),
        (EnrichmentConfig, "ENABLE_SAFETY_LAYER"),
        (EnrichmentConfig, "ENABLE_ROLE_INFERENCE"),
        (EnrichmentConfig, "ENABLE_TOPIC_SEGMENTATION"),
        (EnrichmentConfig, "ENABLE_PROSODY_V2"),
        (EnrichmentConfig, "ENABLE_ENVIRONMENT_CLASSIFIER"),
    ],
)
def test_empty_boolean_environment_value_is_rejected(monkeypatch, config_type, suffix):
    name = f"CUSTOM_{suffix}"
    monkeypatch.setenv(name, "")
    with pytest.raises(ValueError, match=f"Invalid {name}: ''"):
        config_type.from_env(prefix="CUSTOM_")


def test_boolean_environment_value_is_case_insensitive(monkeypatch):
    monkeypatch.setenv("CUSTOM_SKIP_EXISTING_JSON", "FaLsE")
    assert TranscriptionConfig.from_env(prefix="CUSTOM_").skip_existing_json is False
    monkeypatch.delenv("CUSTOM_SKIP_EXISTING_JSON")
    assert TranscriptionConfig.from_env(prefix="CUSTOM_").skip_existing_json is True


def test_empty_boolean_is_rejected_through_from_sources(monkeypatch):
    monkeypatch.setenv("CUSTOM_SKIP_EXISTING_JSON", "")
    with pytest.raises(ValueError, match="CUSTOM_SKIP_EXISTING_JSON"):
        TranscriptionConfig.from_sources(env_prefix="CUSTOM_")
