## 2024-05-13 - [Percentile Optimization]
**Learning:** Calculating multiple percentiles on the same NumPy array using a list `np.percentile(array, [p1, p2])` computes them concurrently in a single pass and is significantly faster than multiple separate `np.percentile` calls.
**Action:** Consolidate multiple `np.percentile` calls on the same array into a single call passing a list of percentiles.
