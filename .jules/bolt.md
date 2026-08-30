## 2025-02-12 - Vectorizing audio frame iteration in streaming ASR
**Learning:** Python `for` loops iterating over numpy array slices are a significant bottleneck in audio processing logic (like VAD).
**Action:** Replace `for` loops with vectorized `numpy` operations. Reshape the array into `(num_frames, frame_size)` and compute across axes directly.
