
## 2024-05-18 - Batching NumPy Percentile Calculations
**Learning:** Calculating multiple percentiles sequentially on the same NumPy array using separate `np.percentile` calls incurs repeated O(n) traversal overhead.
**Action:** Always pass percentiles as a list to a single `np.percentile` call (e.g., `np.percentile(array, [10, 90])`) to compute them concurrently in a single pass, which is significantly faster.
