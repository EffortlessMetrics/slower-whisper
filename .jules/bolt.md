## 2024-03-26 - Fast-path literal checks for Semantic Annotator
**Learning:** Optimizing regex keyword searches with literal string inclusion fast-paths (`kw_lower in text_lower`) can significantly reduce annotation time in rule-based semantic taggers, especially when processing long transcripts.
**Action:** Always consider fast-path string inclusion checks before executing expensive regex searches when the literal string presence is a prerequisite for the regex bounds check to succeed.
