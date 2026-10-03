## 2024-05-18 - Optimized RMS Energy Calculation

**Learning:** Using `np.vdot(array, array) / array.size` is roughly 4x faster than `np.mean(array**2)` for 1D float arrays because it avoids allocating an intermediate array for `array**2`. For multi-dimensional frames, `np.einsum('ij,ij->i', frames, frames) / frames.shape[1]` provides massive speedups. Similarly, for linear regression, centering variables and using `np.vdot` avoids allocating large intermediate arrays for multiplication compared to `np.sum((x - x_mean) * (y - y_mean))`.

**Action:** Whenever `np.mean(x**2)` or sum of squares is needed, prefer `vdot` for 1D arrays or `einsum` for multi-dimensional arrays to avoid allocating temporary memory.
