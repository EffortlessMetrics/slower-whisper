## 2026-01-25 - Destructive Action Confirmation
**Learning:** Destructive actions (like clearing cache) require distinct visual warnings. Plain text warnings are often missed.
**Action:** Use `Colors.red()` for critical warnings like "This cannot be undone." inside interactive prompts to enforce attention.

## 2026-01-26 - Atomic Pre-flight Checks
**Learning:** For bulk file operations (like copying samples), users prefer a "check-then-act" model where all conflicts are reported upfront, rather than failing on the first conflict.
**Action:** Implement pre-flight checks that gather *all* conflicts and raise a custom error (like `SampleExistsError`) containing the full list, allowing the CLI to present a complete summary before asking for confirmation.
## 2025-03-04 - Handle KeyboardInterrupt Gracefully in CLI Prompts
**Learning:** Raw stack traces from `KeyboardInterrupt` (Ctrl+C) during interactive CLI prompts represent a surprisingly jarring and poor user experience, as it exposes users to the underlying program structure instead of providing a controlled exit.
**Action:** Always wrap interactive CLI prompts (e.g., `input()`) in `try...except KeyboardInterrupt` blocks that catch the interruption, print a clear exit message (like `\nAborted.`), and return a clean `0` status code to prevent the exception from propagating up and crashing the program. This should be adopted as a standard practice for all interactive terminal programs.
