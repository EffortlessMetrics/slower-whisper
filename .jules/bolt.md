## 2026-01-26 - [Quadratic Diarization Alignment]
**Learning:** `assign_speakers` and `assign_speakers_to_words` iterate all speaker turns for every segment, leading to O(N*M) complexity. This becomes a bottleneck with long transcripts (many segments/turns).
**Action:** Implement sliding window optimization to prune "expired" turns from the search space, achieving O(N+M) for chronological data.
