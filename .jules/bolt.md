## 2024-06-30 - O(n) sparse vector dot products
**Learning:** Computing cosine similarity between sparse vectors represented as dictionaries is inefficient when using `set(vec1.keys()) & set(vec2.keys())` due to set creation overhead.
**Action:** Always compute dot products or cosine similarity by iterating over the items of the smaller dictionary and performing key lookups in the larger dictionary to achieve O(n) time complexity and ~2x speedup.
