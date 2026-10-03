## 2026-01-25 - Destructive Action Confirmation
**Learning:** Destructive actions (like clearing cache) require distinct visual warnings. Plain text warnings are often missed.
**Action:** Use `Colors.red()` for critical warnings like "This cannot be undone." inside interactive prompts to enforce attention.

## 2026-01-26 - Atomic Pre-flight Checks
**Learning:** For bulk file operations (like copying samples), users prefer a "check-then-act" model where all conflicts are reported upfront, rather than failing on the first conflict.
**Action:** Implement pre-flight checks that gather *all* conflicts and raise a custom error (like `SampleExistsError`) containing the full list, allowing the CLI to present a complete summary before asking for confirmation.

## 2026-02-15 - Explicit Action Confirmation Colors
**Learning:** For CLI applications, a simple but effective micro-UX improvement is to explicitly color the options in confirmation prompts (e.g., `[{Colors.red('y')}/{Colors.green('N')}]`) to visually reinforce safe versus destructive choices.
**Action:** Always color prompt options for destructive commands, combining with a clear warning.
