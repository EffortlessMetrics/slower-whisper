## 2026-06-07 - Optimize RMS energy calculations using np.dot
**Learning:** Using `np.mean(array**2)` for 1D arrays calculates squares first allocating a new temporary array. Using `np.dot(array, array) / len(array)` is an order of magnitude faster as it uses optimized BLAS routines and avoids allocations.
**Action:** Whenever computing mean squared values or RMS energy for 1D arrays in Python/numpy, use `np.dot(array, array) / len(array)` instead of `np.mean(array**2)`.
