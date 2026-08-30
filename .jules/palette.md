## 2026-01-25 - Destructive Action Confirmation
**Learning:** Destructive actions (like clearing cache) require distinct visual warnings. Plain text warnings are often missed.
**Action:** Use `Colors.red()` for critical warnings like "This cannot be undone." inside interactive prompts to enforce attention.

## 2026-01-26 - Atomic Pre-flight Checks
**Learning:** For bulk file operations (like copying samples), users prefer a "check-then-act" model where all conflicts are reported upfront, rather than failing on the first conflict.
**Action:** Implement pre-flight checks that gather *all* conflicts and raise a custom error (like `SampleExistsError`) containing the full list, allowing the CLI to present a complete summary before asking for confirmation.

## 2026-03-12 - Graceful Ctrl+C Handling in CLI Prompts
**Learning:** Terminal applications crashing with ugly stack traces when the user hits Ctrl+C (KeyboardInterrupt) during an interactive prompt creates a frightening and unprofessional UX.
**Action:** Always wrap interactive confirmation prompts (like `input()`) in `try...except KeyboardInterrupt` blocks that catch `Ctrl+C`, print a clean `\nAborted.`, and exit gracefully.
