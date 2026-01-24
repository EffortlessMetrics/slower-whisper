## 2025-05-15 - Dataclass Secret Leakage
**Vulnerability:** `LLMConfig` dataclass included `api_key` in its default `__repr__`, causing secrets to be logged in plain text during debugging or error reporting.
**Learning:** Python dataclasses automatically generate a `__repr__` method that includes all fields. This is unsafe for classes holding secrets.
**Prevention:** Always implement a custom `__repr__` method for configuration dataclasses that hold sensitive information, explicitly masking secrets (e.g., `api_key="sk-...***"`).
