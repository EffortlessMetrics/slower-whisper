## 2026-01-25 - Destructive Action Confirmation
**Learning:** Destructive actions (like clearing cache) require distinct visual warnings. Plain text warnings are often missed.
**Action:** Use `Colors.red()` for critical warnings like "This cannot be undone." inside interactive prompts to enforce attention.

## 2026-01-26 - Atomic Pre-flight Checks
**Learning:** For bulk file operations (like copying samples), users prefer a "check-then-act" model where all conflicts are reported upfront, rather than failing on the first conflict.
**Action:** Implement pre-flight checks that gather *all* conflicts and raise a custom error (like `SampleExistsError`) containing the full list, allowing the CLI to present a complete summary before asking for confirmation.
## 2024-05-24 - Interactive Prompts Graceful Degradation
**Learning:** Interactive CLI prompts for destructive actions should include a distinct visual warning and handle Ctrl+C gracefully by catching KeyboardInterrupt, returning 130, and printing Aborted to prevent stack traces.
**Action:** Wrap `input()` calls in `try...except KeyboardInterrupt` and use `Colors.red()` for destructive warnings.
