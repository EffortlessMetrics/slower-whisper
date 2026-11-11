## 2026-06-16 - Vectorized Linear Regression
**Learning:** `np.vdot` on centered arrays is ~2x faster than using `np.sum((x - x_mean) * (y - y_mean))` because it avoids the allocation of intermediate arrays.
**Action:** Use `np.vdot` to efficiently calculate sums of squares and cross-products in performance-critical numerical operations.
