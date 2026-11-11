## 2026-06-08 - Fast PCM Audio Frame Energy Calculation
**Learning:** Processing raw binary audio data (PCM 16-bit) into frames by unpacking bytes into Python tuples with `struct.unpack` and iterating over them with Python loops is extremely slow.
**Action:** Directly convert the buffer to a NumPy array using `np.frombuffer(buffer, dtype="<h")`, cast to float32, and apply vectorized operations like `np.sqrt(np.einsum('ij,ij->i', frames, frames) / frame_samples)` to achieve up to 67x speedups.
