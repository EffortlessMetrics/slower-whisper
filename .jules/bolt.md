## 2026-06-23 - Optimize RMS energy and linear regression
**Learning:** Using `np.vdot(array, array) / array.size` instead of `np.mean(array**2)` avoids temporary array allocations and yields a massive performance speedup when computing sums of squares and cross-products in NumPy.
**Action:** Apply this NumPy performance pattern for RMS and variance computations across all data pipeline processing.
