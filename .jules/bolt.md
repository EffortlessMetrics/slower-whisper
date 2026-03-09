## 2024-03-09 - [Fast-Path Inclusion Checks Before Regex]
**Learning:** Regex word-boundary searches on many segments are a hidden CPU tax. Even simple `\bword\b` patterns evaluate much slower than built-in Python string `in` checks.
**Action:** When scanning transcripts for static keywords using regex (for word boundaries, etc.), always pre-compute lowercase keywords and perform a fast `kw_lower in text_lower` check before invoking the regex engine.
