## 2024-04-27 - [NumPy Percentile Optimization]
**Learning:** Calculating multiple percentiles on the same NumPy array using a single `np.percentile(array, [p1, p2])` call is significantly faster (~30% faster in tests) than making separate calls, as it computes them concurrently in a single pass.
**Action:** When calculating multiple percentiles on the same NumPy array, pass the percentiles as a list to a single `np.percentile` call.
