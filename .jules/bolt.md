## $(date +%Y-%m-%d) - [Optimize array iteration using NumPy vectorization]
**Learning:** Python loops over NumPy array slices are significantly slower than native vectorized operations. Vectorizing frame-wise audio processing (like energy calculation) provides massive speedups (~30x).
**Action:** Use NumPy vectorization (`reshape` and `axis` operations) instead of Python `for` loops when iterating over frames of arrays, ensuring strict typing compliance by explicitly casting `[bool(x) for x in boolean_array]`.
