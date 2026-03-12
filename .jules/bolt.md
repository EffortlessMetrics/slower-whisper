## 2024-05-17 - [Optimizing Keyword Annotators]
**Learning:** For regex-based keyword annotators (like `transcription.semantic.KeywordSemanticAnnotator`), performance can be significantly improved by pre-computing lowercase keywords in `__post_init__` and utilizing fast-path string inclusion checks (`kw_lower in text_lower`) before expensive regex matching.
**Action:** Always consider adding a fast-path literal string check before expensive regex matching, ensuring the literal kw_lower is strictly required by the word boundary regex pattern.
