# Bolt's Performance Journal

This file contains critical learnings, architectural bottlenecks, and surprising edge cases related to performance in this codebase.

## 2024-05-24 - Vectorizing Sequential Audio Chunk Operations
**Learning:** In Python, pure loops computing energy or RMS for sequential audio frames (like VAD) are a significant bottleneck on large arrays.
**Action:** Always vectorize sequential frame operations by reshaping the 1D audio array into a 2D array of `(num_frames, frame_size)` and applying axis-wise NumPy operations (e.g., `np.mean(frames**2, axis=1)`).
