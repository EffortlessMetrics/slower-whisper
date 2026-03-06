## 2026-01-25 - Destructive Action Confirmation
**Learning:** Destructive actions (like clearing cache) require distinct visual warnings. Plain text warnings are often missed.
**Action:** Use `Colors.red()` for critical warnings like "This cannot be undone." inside interactive prompts to enforce attention.

## 2026-01-26 - Atomic Pre-flight Checks
**Learning:** For bulk file operations (like copying samples), users prefer a "check-then-act" model where all conflicts are reported upfront, rather than failing on the first conflict.
**Action:** Implement pre-flight checks that gather *all* conflicts and raise a custom error (like `SampleExistsError`) containing the full list, allowing the CLI to present a complete summary before asking for confirmation.

## 2026-01-26 - Destructive Action Confirmation and Graceful Interrupt Handling
**Learning:** Destructive actions (like deleting a speaker, clearing cache) require distinct visual warnings. Plain text warnings are often missed. Interactive prompts (like `input()`) should be wrapped in `try...except KeyboardInterrupt` blocks to cleanly handle user cancellation (Ctrl+C) instead of showing a stack trace.
**Action:** Use `Colors.red()` for critical warnings like "This cannot be undone." inside interactive prompts to enforce attention. Wrap `input()` prompts in `try...except KeyboardInterrupt` blocks that print `\nAborted.` and return cleanly.
