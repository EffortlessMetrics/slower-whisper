## 2024-05-24 - Optimizing keyword semantic annotation performance

**Learning:** For regex-based keyword annotators like `transcription.semantic.KeywordSemanticAnnotator`, executing full regex searches for words that do not even appear in the transcript text introduces an unnecessary performance bottleneck.

**Action:** Performance can be significantly improved by pre-computing lowercase keywords in `__post_init__` and utilizing fast-path string inclusion checks (`if kw_lower in text_lower`) before falling back to regex evaluations. When applying this optimization, always ensure that original keyword casing is retained for downstream reporting by storing a tuple like `(kw, kw_lower, pattern)`.
