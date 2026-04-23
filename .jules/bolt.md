# Bolt's Journal

## 2024-04-23 - [Performance] Vectorize multiple np.percentile calculations
**Learning:** Computing multiple percentiles on the same NumPy array using separate `np.percentile` calls requires multiple passes over the array. Passing a list of percentiles to a single `np.percentile` call computes them concurrently in a single pass, which is significantly faster.
**Action:** When calculating multiple percentiles on the same array, always pass the percentiles as a list to a single `np.percentile` call (e.g., `np.percentile(array, [10, 90])`) instead of making separate calls.

## 2024-04-23 - [Build] MANIFEST.in requires explicit recursive inclusion for wheel builds
**Learning:** When building wheels or installing via `pip install -e .` using setuptools, if a package directory is not explicitly included in `MANIFEST.in` (e.g., via `recursive-include <package> *.py`), it may fail to be packaged with an error like `error: package directory '<package>' does not exist`, even if it is listed in `pyproject.toml`.
**Action:** Always ensure that all source code directories are explicitly included in `MANIFEST.in` using `recursive-include <package> *.py` to guarantee they are packaged correctly across all build environments.

## 2024-04-23 - [Build] Packages require explicit inclusion in pyproject.toml [tool.setuptools]
**Learning:** When building wheels or installing via `pip install -e .` using setuptools with a flat layout, if a package or nested sub-package directory (like `transcription.store`) is missing from the `packages` list under `[tool.setuptools]` in `pyproject.toml`, the build process may fail or emit a warning, excluding the directory from the package distribution.
**Action:** Always verify that newly created sub-packages or missing nested directories are explicitly declared in the `[tool.setuptools] packages` list within `pyproject.toml` to guarantee they are packaged correctly.

## 2024-04-23 - [Build] Packages require explicit inclusion in pyproject.toml [tool.setuptools]
**Learning:** When building wheels or installing via `pip install -e .` using setuptools with a flat layout, if a package or nested sub-package directory (like `transcription.schemas`) is missing from the `packages` list under `[tool.setuptools]` in `pyproject.toml`, the build process may fail or emit a warning, excluding the directory from the package distribution.
**Action:** Always verify that newly created sub-packages or missing nested directories are explicitly declared in the `[tool.setuptools] packages` list within `pyproject.toml` to guarantee they are packaged correctly.
