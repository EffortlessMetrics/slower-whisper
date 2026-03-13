## 2026-01-25 - Destructive Action Confirmation
**Learning:** Destructive actions (like clearing cache) require distinct visual warnings. Plain text warnings are often missed.
**Action:** Use `Colors.red()` for critical warnings like "This cannot be undone." inside interactive prompts to enforce attention.

## 2026-01-26 - Atomic Pre-flight Checks
**Learning:** For bulk file operations (like copying samples), users prefer a "check-then-act" model where all conflicts are reported upfront, rather than failing on the first conflict.
**Action:** Implement pre-flight checks that gather *all* conflicts and raise a custom error (like `SampleExistsError`) containing the full list, allowing the CLI to present a complete summary before asking for confirmation.

## 2025-02-13 - [CLI Interaction Polish]
**Learning:** Destructive CLI prompts using `input()` can cause ugly Python stack traces when interrupted with `Ctrl+C` (`KeyboardInterrupt`), which degrades the terminal user experience and can appear as a crash rather than a deliberate user abort.
**Action:** Always wrap interactive CLI prompts (`input()`) in a `try...except KeyboardInterrupt` block. When caught, print a clear `\nAborted.` message and return an exit code of `0` to signal a clean, intentional exit.
