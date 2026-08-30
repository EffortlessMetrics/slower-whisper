## 2024-05-24 - [Fast path checking for regex keyword matching]
**Learning:** For regex-based keyword annotators like `KeywordSemanticAnnotator`, executing compiled regex directly across every segment's text for every keyword is slow. Most of the time, the regex won't match.
**Action:** Adding a simple substring inclusion check (`keyword in text_lower`) as a fast-path condition prior to expensive regex evaluation (`pattern.search(text_lower)`) significantly improves performance by bypassing the regex engine almost entirely when the substring isn't present.
