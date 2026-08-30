## 2026-01-25 - Destructive Action Confirmation
**Learning:** Destructive actions (like clearing cache) require distinct visual warnings. Plain text warnings are often missed.
**Action:** Use `Colors.red()` for critical warnings like "This cannot be undone." inside interactive prompts to enforce attention.

## 2026-01-26 - Atomic Pre-flight Checks
**Learning:** For bulk file operations (like copying samples), users prefer a "check-then-act" model where all conflicts are reported upfront, rather than failing on the first conflict.
**Action:** Implement pre-flight checks that gather *all* conflicts and raise a custom error (like `SampleExistsError`) containing the full list, allowing the CLI to present a complete summary before asking for confirmation.

## 2026-01-27 - Prompt Option Coloring
**Learning:** For destructive or significant actions in CLI prompts, users benefit from having the options themselves colored (e.g., red for 'y' to delete, green for 'N' to abort) to visually reinforce the consequences of their choice.
**Action:** Use `Colors.red('y')` and `Colors.green('N')` inside `[{y}/{N}]` prompts for destructive actions.
