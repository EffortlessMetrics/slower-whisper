## 2024-04-16 - Vectorize VAD Processing
**Learning:** Pure Python loops inside frequent frame-by-frame VAD checks are extremely slow. By truncating the array to valid frame sizes, reshaping it, and applying np.sqrt(np.mean(..., axis=1)), processing performance saw a ~30x speedup.
**Action:** Always attempt to vectorize array frame iterations into reshape() and axis-wise numpy operations before falling back to pure Python for loops.
