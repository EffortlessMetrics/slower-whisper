## 2026-01-25 - Destructive Action Confirmation
**Learning:** Destructive actions (like clearing cache) require distinct visual warnings. Plain text warnings are often missed.
**Action:** Use `Colors.red()` for critical warnings like "This cannot be undone." inside interactive prompts to enforce attention.

## 2026-01-26 - Atomic Pre-flight Checks
**Learning:** For bulk file operations (like copying samples), users prefer a "check-then-act" model where all conflicts are reported upfront, rather than failing on the first conflict.
**Action:** Implement pre-flight checks that gather *all* conflicts and raise a custom error (like `SampleExistsError`) containing the full list, allowing the CLI to present a complete summary before asking for confirmation.

## 2026-02-05 - Inconsistent Warning Formats for Destructive CLI Actions
**Learning:** Destructive actions across different CLI commands (e.g., `cache --clear` vs `speaker-identity --delete`) can lack consistent visual warning weight. The `speaker-identity` command was missing the explicit red "This cannot be undone." warning used elsewhere.
**Action:** Always use `Colors.red("This cannot be undone.")` inside interactive confirmation prompts for irreversible operations to maintain a consistent UX pattern for danger warnings.
