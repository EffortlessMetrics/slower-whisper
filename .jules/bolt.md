## 2024-06-27 - Fast Numpy Aggregations
**Learning:** `np.mean(array**2)` and `np.sum((x - x_mean) * (y - y_mean))` are common but allocate temporary arrays and run slower than optimized dot products.
**Action:** Use `np.vdot(array, array) / array.size` for RMS and `np.vdot` on pre-centered arrays for cross-products to avoid temporary arrays and speed up computation by 2-4x.
