## YYYY-MM-DD - [Title]
**Learning:** [Insight]
**Action:** [How to apply next time]

## 2024-05-24 - [KeywordSemanticAnnotator regex loop performance bottleneck]
**Learning:** For regex-based keyword annotators (like `transcription.semantic.KeywordSemanticAnnotator`), running full regex searches `pattern.search(text_lower)` for every predefined keyword over many segments in a tight loop acts as a significant performance bottleneck because regex execution is much slower than a simple substring search.
**Action:** Pre-compute lowercase keywords in `__post_init__` and utilize fast-path string inclusion checks (`kw_lower in text_lower`) before performing the regex search. Ensure the literal string is strictly required by the regex pattern to avoid skipping valid matches.
