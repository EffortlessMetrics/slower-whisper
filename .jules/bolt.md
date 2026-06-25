## 2026-06-25 - Optimizing RMS calculation
**Learning:** `np.mean(arr**2)` is an anti-pattern for RMS calculation in high-performance paths (like streaming audio chunks). It creates temporary arrays for `arr**2` which slows down execution significantly. `np.vdot(arr, arr) / arr.size` is ~10x faster because it avoids allocating temporary arrays and computes the dot product in compiled C code.
**Action:** Always use `np.vdot(arr, arr) / arr.size` for variance/energy calculations instead of squaring the array and taking the mean.
