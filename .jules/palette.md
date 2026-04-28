## 2026-01-25 - Destructive Action Confirmation
**Learning:** Destructive actions (like clearing cache) require distinct visual warnings. Plain text warnings are often missed.
**Action:** Use `Colors.red()` for critical warnings like "This cannot be undone." inside interactive prompts to enforce attention.

## 2026-01-26 - Atomic Pre-flight Checks
**Learning:** For bulk file operations (like copying samples), users prefer a "check-then-act" model where all conflicts are reported upfront, rather than failing on the first conflict.
**Action:** Implement pre-flight checks that gather *all* conflicts and raise a custom error (like `SampleExistsError`) containing the full list, allowing the CLI to present a complete summary before asking for confirmation.

## 2026-02-15 - Visualizing Destructive Choices
**Learning:** In CLI confirmation prompts, visually coloring options (like red 'y' vs green 'N') effectively reinforces safe versus destructive choices for users.
**Action:** Explicitly color prompt choices (e.g. `[{Colors.red('y')}/{Colors.green('N')}]`) in interactive confirmation prompts for destructive operations.
