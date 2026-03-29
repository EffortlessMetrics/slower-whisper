## 2026-03-29 - Fast-path literal check for regex keyword annotators
**Learning:** Regex searches on thousands of segments for multiple keywords can be a major bottleneck. Pre-computing lowercase literals and using string inclusion checks before invoking regex search yields massive speedups on sparse matches.
**Action:** Use `kw_lower in text_lower` fast paths to avoid regex overhead when searching for simple exact keyword matches.
