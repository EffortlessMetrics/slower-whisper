# Bolt's Journal
## 2025-04-04 - [Vectorize NumPy Loops in Streaming ASR]
**Learning:** Iterating over NumPy array slices in a Python `for` loop to calculate frame energies creates significant overhead for streaming audio chunks.
**Action:** Always use NumPy vectorization (`np.reshape` and `np.mean(..., axis=1)`) when applying calculations to sequential blocks of data instead of Python loops.
