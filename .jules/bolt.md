## 2024-05-24 - Vectorize numpy array iterations for faster operations
**Learning:** In audio processing, iterating over frames manually in Python is slow. Replacing a Python for-loop over Numpy slices with a `reshape` followed by vector operations on the resulting axis reduces processing time significantly, avoiding Python loop overhead completely.
**Action:** Whenever iterating over sequential partitions of an array to calculate aggregates (like RMS energy), always reshape the array to N-dimensions and apply numpy aggregates (like `np.mean(..., axis=1)`) instead of using Python loops.
