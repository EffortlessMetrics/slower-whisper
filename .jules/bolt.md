## 2024-XX-XX - [Vectorize Streaming VAD Energy Calculations]
**Learning:** Sequential frame-by-frame RMS calculation in Python loops is extremely slow compared to native NumPy matrix operations.
**Action:** Always vectorize sequential chunk/frame operations using NumPy native functions (like reshape and axis-wise operations such as np.mean(..., axis=1)) instead of slicing and processing individual chunks in pure Python for loops. Make sure to cast int16 arrays to float32 before squaring to prevent integer overflow.
