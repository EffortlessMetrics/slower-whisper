## 2026-01-25 - Destructive Action Confirmation
**Learning:** Destructive actions (like clearing cache) require distinct visual warnings. Plain text warnings are often missed.
**Action:** Use `Colors.red()` for critical warnings like "This cannot be undone." inside interactive prompts to enforce attention.

## 2026-01-26 - Atomic Pre-flight Checks
**Learning:** For bulk file operations (like copying samples), users prefer a "check-then-act" model where all conflicts are reported upfront, rather than failing on the first conflict.
**Action:** Implement pre-flight checks that gather *all* conflicts and raise a custom error (like `SampleExistsError`) containing the full list, allowing the CLI to present a complete summary before asking for confirmation.

## 2026-02-12 - Batch Validation Feedback
**Learning:** When validating multiple items (like transcripts), users need granular, per-item status reports rather than a single "Validation failed" message. This allows them to quickly identify which files need attention.
**Action:** Iterate through items and print "[PASS]" or "[FAIL]" for each, grouping errors under the failing item, and provide a summary count at the end.
