## 2024-06-30 - O(n) sparse vector dot products
**Learning:** Computing cosine similarity between sparse vectors represented as dictionaries is inefficient when using `set(vec1.keys()) & set(vec2.keys())` due to set creation overhead.
**Action:** Always compute dot products or cosine similarity by iterating over the items of the smaller dictionary and performing key lookups in the larger dictionary to achieve O(n) time complexity and ~2x speedup.
## 2024-06-30 - Docker build issues
**Learning:** `docker build` within the sandbox environment fails with 'overlayfs' mount errors due to docker-in-docker restrictions.
**Action:** Bypass the local smoke test assuming the Dockerfile syntax is correct. If `docker build` fails during `uv pip install -e .` with an `error: package directory '<name>' does not exist` (e.g., `slower_whisper`), ensure the corresponding Python package source directory is explicitly `COPY`'d into the container before the installation step, and also copied into any runtime stages since editable installs require the source code at runtime.
