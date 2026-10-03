## 2024-05-10 - Combine multiple np.percentile calls
**Learning:** To optimize multiple `np.percentile` calculations on the same array in NumPy, combine them into a single call passing a list of percentiles (e.g., `np.percentile(array, [10, 90])`) to significantly reduce Python overhead and internal sorting passes.
**Action:** Always check if multiple `np.percentile` calls on the same array can be combined into a single call.
