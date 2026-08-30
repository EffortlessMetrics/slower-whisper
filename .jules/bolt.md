
## 2026-05-02 - Use uv pip install for faster virtual environment setup
**Learning:** `uv venv` does not install `pip` by default. Using `uv venv --seed` forces it to install pip, but then calling `pip install` uses standard pip, which is slow.
**Action:** Instead of seeding `pip`, always use `uv pip install --python <venv_path> <package>` to install directly into the virtual environment using `uv`'s significantly faster resolution and installation engine.
