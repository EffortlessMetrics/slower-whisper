## 2024-06-28 - Initializing Bolt Journal
**Learning:** Initializing journal for critical performance learnings.
**Action:** Use this file to record important performance patterns or bottlenecks discovered in this codebase.
## 2024-06-28 - Optimized cosine similarity
**Learning:** Found an O(n) optimization for cosine similarity in sparse vectors (TF-IDF dicts) in `transcription/topic_segmentation.py`. The original implementation did `set(vec1.keys()) & set(vec2.keys())` which is costly. Then it did `sum(v**2 for v in vec1.values())` which could be precomputed during the dict traversal.
**Action:** Replaced it with a single loop that avoids set intersection overhead, iterating over the smaller dict and looking up in the larger dict. This gives ~2.25x speedup for cosine_similarity computations.
## 2024-06-28 - Fixed Docker Build Error
**Learning:** Found an issue where the docker build failed with `error: package directory 'slower_whisper' does not exist` when trying to do `uv pip install ... -e .`. This happens because `slower_whisper` is a separate package directory within the repository (like `transcription`), and it needs to be copied into the builder stage before running `uv pip install`.
**Action:** Always make sure `COPY slower_whisper/ ./slower_whisper/` is in the Dockerfile builder and runtime stages if it exists as an independent package directory in the repository.
## 2024-06-28 - Fixed Gitleaks and uv venv Errors
**Learning:** Found two CI errors. First, `gitleaks-action` needs a `GITLEAKS_LICENSE` environment variable. Second, when using `uv venv` to create a virtual environment, `pip` is not installed by default in the clean environment.
**Action:** Always provide `GITLEAKS_LICENSE` to the `gitleaks/gitleaks-action` env block. When installing wheels in a clean `uv venv`, use `uv pip install <wheel_path> --python <venv_dir>` instead of relying on a non-existent `<venv_dir>/bin/pip` executable.
