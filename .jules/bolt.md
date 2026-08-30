## 2024-05-24 - Fast-path string inclusion before regex evaluation
**Learning:** Regex evaluation can be slow, especially in tight loops like `KeywordSemanticAnnotator` over transcript segments. Doing a simple `keyword in text_lower` check prior to regex execution dramatically skips unnecessary regex evaluations and speeds up processing.
**Action:** Use string subset checks (`in`) as a fast-path filter before evaluating `re.Pattern.search` for exact word boundary matches.
