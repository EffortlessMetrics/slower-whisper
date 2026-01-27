# Bolt's Journal

## 2024-05-22 - O(N*M) Diarization Complexity
**Learning:** The `assign_speakers` function in `transcription/diarization.py` had O(N*M) complexity because it iterated through all speaker turns for every segment. This became a bottleneck for long audio files with many segments and turns.
**Action:** Implemented a sliding window approach (O(N+M)) by maintaining a `turn_idx` that advances as segments progress, leveraging the fact that both segments and turns are sorted by time.
