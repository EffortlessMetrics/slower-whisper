## 2024-04-14 - Initialize Bolt Journal
**Learning:** Initializing journal to track critical learnings.
**Action:** Keep track of performance optimization lessons here.
## 2024-04-14 - Vectorized Audio VAD Processing
**Learning:** Python loops over audio frame slices with numpy operations per slice are significantly slower than reshaping the audio array and using vectorized operations.
**Action:** Always favor vectorized numpy processing over looping when computing statistics like frame energy on chunked buffers.
