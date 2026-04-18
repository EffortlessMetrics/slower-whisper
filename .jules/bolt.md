## 2024-04-17 - [Vectorize Audio Frame Processing]
**Learning:** Slicing arrays and calculating energy chunk-by-chunk in a Python for-loop is a massive bottleneck.
**Action:** When optimizing audio or array processing in Python, always vectorize sequential chunk/frame operations using NumPy native functions (like `reshape` and axis-wise operations such as `np.mean(..., axis=1)`) instead of slicing and processing individual chunks in pure Python `for` loops, converting back to lists only if strictly required downstream.
