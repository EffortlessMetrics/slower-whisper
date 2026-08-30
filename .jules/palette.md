## 2026-01-25 - Destructive Action Confirmation
**Learning:** Destructive actions (like clearing cache) require distinct visual warnings. Plain text warnings are often missed.
**Action:** Use `Colors.red()` for critical warnings like "This cannot be undone." inside interactive prompts to enforce attention.

## 2026-01-26 - Atomic Pre-flight Checks
**Learning:** For bulk file operations (like copying samples), users prefer a "check-then-act" model where all conflicts are reported upfront, rather than failing on the first conflict.
**Action:** Implement pre-flight checks that gather *all* conflicts and raise a custom error (like `SampleExistsError`) containing the full list, allowing the CLI to present a complete summary before asking for confirmation.

## 2026-06-10 - Validation Error Coloring
**Learning:** Validation failures in the CLI are much easier to parse when they use color-coding (e.g., `Colors.red` for failures and `Colors.green` for successes), helping users quickly distinguish between successful checks and actionable errors.
**Action:** Use ANSI color utilities like `Colors.red()` and `Colors.green()` to format validation results, emphasizing failures and confirming successes clearly.
