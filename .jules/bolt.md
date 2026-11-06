## 2023-10-25 - Vectorized RMS Energy Calculation
**Learning:** When processing raw binary audio data (PCM 16-bit) into frames, avoid unpacking bytes into Python tuples with `struct.unpack` and iterating over them with Python loops.
**Action:** Instead, directly convert the buffer to a NumPy array using `np.frombuffer(buffer, dtype="<h")` and explicitly cast to `np.float32` before calculating energy (`np.sqrt(np.mean(frames**2, axis=1))`) to avoid integer overflow and leverage vectorized processing.
