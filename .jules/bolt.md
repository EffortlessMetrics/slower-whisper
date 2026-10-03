## 2024-05-14 - Fast RMS Energy Calculation

**Learning:** Calculating RMS energy in Python by doing `np.mean(array**2)` is computationally expensive and causes unnecessary array allocations, which impacts real-time performance.
**Action:** Use `np.vdot(array, array) / array.size` instead. It does not create temporary arrays, properly handles multi-dimensional arrays without shape regressions, and is ~4x faster. For computing framed energy, `np.einsum('ij,ij->i', frames, frames) / frame_length` is extremely fast.
