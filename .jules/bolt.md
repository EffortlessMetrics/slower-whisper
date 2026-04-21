## 2024-04-21 - Vectorized VAD Energy Calculation
**Learning:** Sequential frame-wise energy calculations in pure Python loops create significant overhead in real-time streaming VAD pipelines.
**Action:** Always vectorize audio frame processing by truncating to exact frame multiples, reshaping to 2D arrays, and using axis-wise NumPy operations.
