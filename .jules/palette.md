## 2026-01-25 - Destructive Action Confirmation
**Learning:** Destructive actions (like clearing cache) require distinct visual warnings. Plain text warnings are often missed.
**Action:** Use `Colors.red()` for critical warnings like "This cannot be undone." inside interactive prompts to enforce attention.

## 2026-01-26 - Atomic Pre-flight Checks
**Learning:** For bulk file operations (like copying samples), users prefer a "check-then-act" model where all conflicts are reported upfront, rather than failing on the first conflict.
**Action:** Implement pre-flight checks that gather *all* conflicts and raise a custom error (like `SampleExistsError`) containing the full list, allowing the CLI to present a complete summary before asking for confirmation.

## 2026-01-27 - Viewport Meta Tag for Exported Artifacts
**Learning:** Exported standalone HTML files (like transcripts) often omit the viewport meta tag, causing mobile browsers to render them at desktop width and scale them down, resulting in unreadable text.
**Action:** Always verify that standalone HTML templates include `<meta name="viewport" content="width=device-width, initial-scale=1.0" />` to ensure proper mobile readability and scaling.
