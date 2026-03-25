
## 2025-03-25 - Keyword Matching Fast-Path Check
**Learning:** For rule-based semantic annotation using regular expressions (`KeywordSemanticAnnotator`), evaluating exact word-boundary regex searches (`\bkeyword\b`) on large amounts of text is surprisingly slow and creates a bottleneck.
**Action:** When evaluating literal string constraints within a regex, pre-compute the lowercase requirement and use Python's fast string inclusion check (`kw_lower in text_lower`) as a fast-path condition *before* executing the regex search. Make sure to document this safety guarantee.
