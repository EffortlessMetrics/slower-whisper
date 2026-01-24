## 2024-05-23 - Speaker Diarization Assignment Optimization
**Learning:** Nested loops over time-series data (e.g., matching speaker turns to segments) can perform poorly ($O(N \cdot M)$) even with early breaks if the outer loop resets the search every time.
**Action:** Use a sliding window (or pointer) approach for sorted time-series data to achieve near-linear time ($O(N + M)$). Ensure to handle unsorted edge cases by resetting the pointer.
