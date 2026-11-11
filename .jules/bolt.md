## 2026-06-09 - Optimize NumPy Calculations

**Learning:** Grouping multiple `np.percentile` calls into a single call with a list of percentiles reduces Python overhead and internal sorting passes. Also, replacing `np.mean(array**2)` with `np.dot(array, array) / len(array)` leverages optimized BLAS routines and prevents allocating temporary arrays, yielding a ~5x speedup for RMS energy computation.

**Action:** Always combine `np.percentile` calls when calculating multiple percentiles on the same array. For RMS energy calculations on 1D arrays, use `np.dot(array, array) / len(array)` instead of squaring and computing the mean.
