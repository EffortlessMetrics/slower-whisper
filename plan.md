1. **Understand the Goal**: As "Palette" (a UX-focused agent), I need to find and implement *one micro-UX improvement* that makes the interface more intuitive, accessible, or pleasant to use. The change must be under 50 lines, not add new dependencies, and follow existing patterns.
2. **Identify Opportunities**:
   - `input()` calls in Python CLI scripts can crash with ugly stack traces when users hit `Ctrl+C` (`KeyboardInterrupt`). This is a common, frustrating UX issue in command-line tools.
   - I saw that `input()` is used in `transcription/cli.py` (lines 802, 875) and `transcription/speaker_identity.py` (line 1566) for confirmation prompts.
   - The memory guidelines even explicitly mention: "CLI commands with interactive confirmation prompts (e.g., using `input()`) should be wrapped in `try...except KeyboardInterrupt` blocks that catch `Ctrl+C`, print `\nAborted.`, and cleanly return code `0` to prevent unhandled stack traces and improve terminal UX." This confirms it's a known issue/pattern in this codebase.
3. **Select the Enhancement**: Wrap `input()` calls with a `try...except KeyboardInterrupt` block in `transcription/cli.py` (cache clear and samples copy commands) and `transcription/speaker_identity.py` (delete speaker command) to handle `Ctrl+C` gracefully.
4. **Implement**:
   - In `transcription/cli.py`:
     - Around line 802 (cache clear confirmation).
     - Around line 875 (samples copy overwrite confirmation).
   - In `transcription/speaker_identity.py`:
     - Around line 1566 (speaker delete confirmation).
   - Add unit tests simulating this behavior if possible, or verify manually. Wait, memory says: "To write unit tests simulating a KeyboardInterrupt (Ctrl+C) during an interactive CLI prompt, use unittest.mock.patch on sys.stdin.isatty (returning True) and builtins.input (with side_effect=KeyboardInterrupt)."
5. **Verify**: Run `uv run pytest tests/test_samples_ux.py` (and add a test for KeyboardInterrupt in it if it makes sense). Run `uv run ruff check --fix .` and `uv run ruff format .`.
6. **Pre-commit**: Run `pre_commit_instructions` and follow steps.
7. **Submit**: Create PR with a title like "🎨 Palette: Gracefully handle Ctrl+C in CLI prompts".
