## 2026-06-05 - Vectorized RMS calculation
**Learning:** For highly optimized, vectorized calculation of RMS energy across 2D NumPy arrays (e.g., reshaped audio frames), use `np.sqrt(np.einsum('ij,ij->i', frames, frames) / frame_samples)` to efficiently perform the row-wise dot product and eliminate Python loops entirely.
**Action:** When calculating energies or aggregating multiple frames of array data, truncate to multiples of the chunk size, reshape, and use vectorized ops or `np.einsum` instead of explicit loops.
