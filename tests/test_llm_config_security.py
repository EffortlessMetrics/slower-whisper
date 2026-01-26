from transcription.historian.llm_client import LLMConfig


def test_llm_config_repr_masks_api_key():
    """Test that LLMConfig.__repr__ masks the API key."""
    # Test with a long key
    secret_key = "sk-ant-secret-key-12345"
    config = LLMConfig(provider="anthropic", api_key=secret_key, model="claude-3-opus")

    repr_str = repr(config)

    assert secret_key not in repr_str
    assert "sk-...***" in repr_str
    assert "anthropic" in repr_str
    assert "claude-3-opus" in repr_str

    # Test with a short key
    short_key = "12345"
    config_short = LLMConfig(provider="openai", api_key=short_key)

    repr_str_short = repr(config_short)
    assert short_key not in repr_str_short
    assert "***" in repr_str_short

    # Test with None key
    config_none = LLMConfig(provider="local", api_key=None)

    repr_str_none = repr(config_none)
    assert "api_key='***'" in repr_str_none
