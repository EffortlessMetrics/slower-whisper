## 2024-05-24 - Batching np.percentile calls
**Learning:** Sequential `np.percentile` calls on the same array are inefficient as they traverse and sort the array multiple times.
**Action:** When calculating multiple percentiles for the same NumPy array, pass the percentiles as a list to a single `np.percentile` call (e.g., `np.percentile(energy, [10, 90])`) to compute them concurrently in a single pass.
