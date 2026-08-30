## 2026-01-25 - Destructive Action Confirmation
**Learning:** Destructive actions (like clearing cache) require distinct visual warnings. Plain text warnings are often missed.
**Action:** Use `Colors.red()` for critical warnings like "This cannot be undone." inside interactive prompts to enforce attention.

## 2026-01-26 - Atomic Pre-flight Checks
**Learning:** For bulk file operations (like copying samples), users prefer a "check-then-act" model where all conflicts are reported upfront, rather than failing on the first conflict.
**Action:** Implement pre-flight checks that gather *all* conflicts and raise a custom error (like `SampleExistsError`) containing the full list, allowing the CLI to present a complete summary before asking for confirmation.

## 2024-05-18 - Graceful Exit on Interrupt
**Learning:** When building interactive CLIs in Python, raw `input()` calls leave the app vulnerable to ugly stack traces if the user instinctively hits `Ctrl+C` to abort. Catching `KeyboardInterrupt` and exiting with 0 respects the user's intent to cleanly cancel.
**Action:** Always wrap `input()` confirmation prompts in a `try/except KeyboardInterrupt` block that cleanly returns 0 and prints "Aborted".
