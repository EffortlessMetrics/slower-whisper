## 2025-02-28 - Vectorize VAD energy frame processing
**Learning:** Sequential frame processing in Voice Activity Detection using pure Python `for` loops introduces significant and unnecessary overhead when handling large `numpy` audio arrays.
**Action:** Always vectorize chunk/frame operations using NumPy native functions (like `reshape` and axis-wise mathematical operations such as `np.mean(..., axis=1)`) instead of slicing and processing individual chunks in a loop, converting back to Python lists if strictly required downstream.
