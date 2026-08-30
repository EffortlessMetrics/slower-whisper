## 2024-05-18 - Optimize Audio Energy Calculation
**Learning:** To optimize sequential frame-based calculations on NumPy arrays (e.g., evaluating RMS energy across audio frames), avoid using Python `for` loops over array slices.
**Action:** Truncate the array to an exact multiple of the frame size, reshape it using `.reshape(-1, frame_size)`, and apply a vectorized aggregate function like `np.mean(..., axis=1)` to bypass Python loop overhead entirely.
