## 2024-06-26 - Optimized RMS Energy Calculation
**Learning:** `np.mean(audio**2)` in NumPy creates a temporary array for the squared values, which can cause significant memory pressure and slowdowns in hot paths like streaming audio processing.
**Action:** Always use `np.vdot(audio, audio) / audio.size` instead for calculating RMS energy, as it avoids the temporary array allocation and handles multidimensional arrays safely.
