## 2024-05-24 - Numpy Performance Optimizations
**Learning:** `np.mean(array**2)` involves a slow intermediate array allocation.
**Action:** Use `np.vdot(array, array) / array.size` to calculate RMS instead, which is ~4x faster and avoids the intermediate array overhead. Similarly, use `np.vdot(x_centered, y_centered)` instead of `np.sum((x - x_mean) * (y - y_mean))` for computing covariances and regression slopes.
