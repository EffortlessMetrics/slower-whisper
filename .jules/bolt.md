## 2025-04-11 - Vectorize audio processing loops
**Learning:** Python `for` loops iterating over array slices are slow for audio framing tasks.
**Action:** Use NumPy vectorization by reshaping the 1D audio array to 2D and computing aggregations like `np.mean(..., axis=1)`.
