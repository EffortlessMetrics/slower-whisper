# Sentinel Journal

This journal records CRITICAL security learnings, vulnerabilities found, and prevention strategies.

## 2026-02-18 - API Key Leakage in Dataclasses
**Vulnerability:** `LLMConfig` dataclass stored API keys in plaintext, and the default `__repr__` method exposed them. Any logging of the config object would leak the key.
**Learning:** Python `dataclasses` automatically generate `__repr__` methods that include all fields. For classes containing secrets, this is insecure by default.
**Prevention:** Always implement a custom `__repr__` method for dataclasses or classes that hold secrets, ensuring sensitive fields are redacted or masked.
