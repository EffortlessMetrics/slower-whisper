## 2026-01-25 - Destructive Action Confirmation
**Learning:** Destructive actions (like clearing cache) require distinct visual warnings. Plain text warnings are often missed.
**Action:** Use `Colors.red()` for critical warnings like "This cannot be undone." inside interactive prompts to enforce attention.

## 2026-01-26 - Atomic Pre-flight Checks
**Learning:** For bulk file operations (like copying samples), users prefer a "check-then-act" model where all conflicts are reported upfront, rather than failing on the first conflict.
**Action:** Implement pre-flight checks that gather *all* conflicts and raise a custom error (like `SampleExistsError`) containing the full list, allowing the CLI to present a complete summary before asking for confirmation.

## 2026-02-12 - Handling User Interrupts
**Learning:** Raw stack traces from `KeyboardInterrupt` or `EOFError` in interactive CLIs are poor UX and can panic users.
**Action:** Always wrap `input()` calls in `try...except (KeyboardInterrupt, EOFError)` and exit cleanly (e.g., `print("\nAborted.")`) for interactive CLI prompts.

## 2026-02-12 - Screen Reader Landmarks
**Learning:** Missing `<main>` landmarks and `lang` attributes make raw HTML logs (like transcript exports) harder to navigate for screen readers.
**Action:** Add `<html lang="...">` and wrap the core content in `<main>` when generating HTML files, even simple ones.
