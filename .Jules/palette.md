## 2024-05-22 - [Lightweight CLI Colors]
**Learning:** Avoiding heavy dependencies like `rich` for CLI tools requires implementing robust TTY detection (e.g. `sys.stdout.isatty()` and `NO_COLOR`) to support pipes/CI.
**Action:** Reuse `transcription.color_utils` for future CLI enhancements to maintain consistency without bloat.
