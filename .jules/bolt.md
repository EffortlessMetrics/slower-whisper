## 2024-03-24 - Fast-path literal string check before regex
**Learning:** For regex-based keyword annotators (like `transcription.semantic.KeywordSemanticAnnotator`), performance can be significantly improved by pre-computing lowercase keywords in `__post_init__` and utilizing fast-path string inclusion checks (`kw_lower in text_lower`).
**Action:** Use fast-path literal string checks `kw_lower in text_lower` when optimizing regex searches, ensuring original casing is retained for downstream reporting by storing a tuple like `(kw, kw_lower, pattern)`.
