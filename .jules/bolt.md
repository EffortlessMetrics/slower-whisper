
## 2024-04-28 - NumPy Percentile Array Optimization
**Learning:** Calling `np.percentile` sequentially multiple times on the same array causes redundant O(n log n) partial sorting and array passes. However, passing a list of percentiles (e.g., `np.percentile(arr, [10, 90])`) computes them concurrently in a single pass, which is significantly faster.
**Action:** When calculating multiple percentiles over the same NumPy array, always consolidate the calculation into a single `np.percentile` call passing a list of desired percentiles.
