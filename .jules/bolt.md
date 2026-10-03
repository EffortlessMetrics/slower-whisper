## 2024-05-14 - RMS Calculation
**Learning:** `np.mean(array**2)` involves intermediate array allocations for `array**2` and handles multidimensional shapes in ways that may not be optimal.
**Action:** Use `np.vdot(array, array) / array.size` instead for better performance, avoiding intermediate allocations.
