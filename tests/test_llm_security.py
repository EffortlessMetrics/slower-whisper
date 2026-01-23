from transcription.historian.llm_client import LLMConfig

def test_llm_config_repr_security():
    """Test that LLMConfig.__repr__ masks the API key."""
    # Test with long key
    config = LLMConfig(
        provider="openai",
        model="gpt-4o",
        api_key="sk-this-is-a-very-secret-key-that-should-be-masked",
        temperature=0.7
    )
    repr_str = repr(config)
    assert "sk-this-is-a-very-secret-key" not in repr_str
    assert "'sk-...'" in repr_str or "'sk-***'" in repr_str or "sk-" in repr_str

    # Test with short key
    config_short = LLMConfig(
        provider="anthropic",
        api_key="12345"
    )
    repr_str_short = repr(config_short)
    assert "12345" not in repr_str_short
    assert "***" in repr_str_short

    # Test with no key
    config_none = LLMConfig(
        provider="local"
    )
    assert "api_key=None" in repr(config_none)
