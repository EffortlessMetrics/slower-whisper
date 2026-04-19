## 2025-05-18 - Vectorized Energy Calculation
**Learning:** Sequential frame processing in audio operations (like `_detect_speech_frames`) introduces high overhead when creating slices and calculating energy repeatedly in Python loops.
**Action:** Always vectorize sequential chunk/frame operations using NumPy native functions (`reshape`, and axis-wise aggregations like `np.mean(..., axis=1)`) instead of pure Python `for` loops.
