# Bolt's Journal

## 2024-04-23 - [Performance] Vectorize multiple np.percentile calculations
**Learning:** Computing multiple percentiles on the same NumPy array using separate `np.percentile` calls requires multiple passes over the array. Passing a list of percentiles to a single `np.percentile` call computes them concurrently in a single pass, which is significantly faster.
**Action:** When calculating multiple percentiles on the same array, always pass the percentiles as a list to a single `np.percentile` call (e.g., `np.percentile(array, [10, 90])`) instead of making separate calls.
