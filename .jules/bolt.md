## 2024-04-22 - Optimizing Multiple Percentile Calculations
**Learning:** Calculating multiple percentiles sequentially on the same numpy array causes the underlying array to be iterated multiple times.
**Action:** Use a list of percentiles in a single `np.percentile(array, [p1, p2])` call to consolidate the array traversals into a single pass.
