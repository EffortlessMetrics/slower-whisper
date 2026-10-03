## 2026-06-11 - Optimize RMS energy calculation
**Learning:** Using `np.mean(array**2)` to calculate RMS energy is slow and allocates a temporary array. For 1D arrays, `np.dot(array, array) / len(array)` is much faster (leverages BLAS). For 2D batched frames (like in audio processing loops), `np.einsum` with `np.lib.stride_tricks.sliding_window_view` provides a massive speedup by removing python loops and vectorizing the operation.
**Action:** Use `np.dot` for 1D RMS energy and `np.einsum` with `sliding_window_view` for windowed frame RMS calculations.
