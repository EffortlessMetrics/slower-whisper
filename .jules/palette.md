## 2026-01-25 - Destructive Action Confirmation
**Learning:** Destructive actions (like clearing cache) require distinct visual warnings. Plain text warnings are often missed.
**Action:** Use `Colors.red()` for critical warnings like "This cannot be undone." inside interactive prompts to enforce attention.

## 2026-01-26 - Atomic Pre-flight Checks
**Learning:** For bulk file operations (like copying samples), users prefer a "check-then-act" model where all conflicts are reported upfront, rather than failing on the first conflict.
**Action:** Implement pre-flight checks that gather *all* conflicts and raise a custom error (like `SampleExistsError`) containing the full list, allowing the CLI to present a complete summary before asking for confirmation.

## 2026-03-08 - KeyboardInterrupt Handling in Interactive Prompts
**Learning:** Unhandled `KeyboardInterrupt` (Ctrl+C) in interactive prompts causes ungraceful stack traces, poor terminal UX, and incorrect process exit statuses.
**Action:** Wrap all interactive `input()` calls in `try...except KeyboardInterrupt` blocks that print a clear `\nAborted.` message and return exit code `130` to correctly signal process interruption to the shell. Ensure destructive actions reinforce the warning using `Colors.red()`.
