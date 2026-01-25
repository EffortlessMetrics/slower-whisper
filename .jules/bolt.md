## 2024-05-22 - [Optimizing Diarization Speaker Assignment]
**Learning:** `assign_speakers` was O(N*M) due to re-scanning all speaker turns for each segment. Since both segments and turns are sorted by time, we can maintain a "window" of relevant turns, skipping those that have ended before the current segment starts. This makes the complexity roughly O(N + M).
**Action:** When matching sorted intervals, always look for sliding window optimizations to avoid quadratic complexity.
