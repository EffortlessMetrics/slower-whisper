## 2024-05-02 - Bolt Initialization
**Learning:** Initialized Bolt journal.
**Action:** Ready to record critical performance learnings.

## 2024-05-02 - Batched numpy percentile calculation
**Learning:** Calculating multiple percentiles on the same NumPy array using a list (e.g. `np.percentile(arr, [10, 90])`) computes them concurrently in a single pass, which is ~10x faster than calling `np.percentile` sequentially.
**Action:** Always batch percentile requests into a single list argument when computing multiple percentiles on the same data.
