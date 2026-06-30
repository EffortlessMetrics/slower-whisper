# Bolt's Journal
## 2024-05-23 - Fast RMS calculation in Python
**Learning:** `np.vdot(array, array) / array.size` is around 14x faster for calculating RMS energy compared to `np.mean(array**2)`.
**Action:** Replace `np.mean(array**2)` with `np.vdot(array, array) / array.size` for all RMS energy calculations across the codebase.
