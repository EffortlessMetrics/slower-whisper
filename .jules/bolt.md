## 2024-05-24 - Fast-path literal string check in regex annotators
**Learning:** For regex-based keyword annotators (like `transcription.semantic.KeywordSemanticAnnotator`), performance can be significantly improved by pre-computing lowercase keywords in `__post_init__` and utilizing fast-path string inclusion checks (`kw_lower in text_lower`) before running `pattern.search()`. The literal string is strictly required by the regex pattern `\b{kw}\b`.
**Action:** Always pre-compute lowercases and use fast-path literal string checks before regex searches where applicable.
