## 2026-01-25 - Destructive Action Confirmation
**Learning:** Destructive actions (like clearing cache) require distinct visual warnings. Plain text warnings are often missed.
**Action:** Use `Colors.red()` for critical warnings like "This cannot be undone." inside interactive prompts to enforce attention.

## 2026-01-26 - Atomic Pre-flight Checks
**Learning:** For bulk file operations (like copying samples), users prefer a "check-then-act" model where all conflicts are reported upfront, rather than failing on the first conflict.
**Action:** Implement pre-flight checks that gather *all* conflicts and raise a custom error (like `SampleExistsError`) containing the full list, allowing the CLI to present a complete summary before asking for confirmation.

## 2026-01-27 - Interruptible Interactive Prompts
**Learning:** Destructive actions with interactive prompts should be wrappable by `try...except KeyboardInterrupt` blocks to cleanly catch `Ctrl+C` inputs, printing `\nAborted.` and returning an exit code of `130` instead of leaking unhandled stack traces and improving terminal UX.
**Action:** Wrap all `input()` calls for CLI commands involving destructive actions inside a `try...except KeyboardInterrupt` to accurately signal a process interrupt to the shell and improve terminal UX.
