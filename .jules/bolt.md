## 2024-06-10 - Initial
**Learning:** Initialized Bolt journal.
**Action:** Ready to optimize.

## 2026-06-10 - NumPy optimizations for array metrics
**Learning:** Calculating RMS energy with `np.dot(arr, arr) / len(arr)` is ~4x faster than `np.mean(arr**2)` by leveraging BLAS and avoiding temporary allocations. Grouping multiple percentile calculations into a single `np.percentile(arr, [low, high])` call also yields ~2x speedup by reducing sorting passes.
**Action:** Look for mean-of-squares patterns to replace with dot products and combine multiple percentile extractions.
