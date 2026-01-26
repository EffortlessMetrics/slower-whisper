## 2026-01-26 - Colored Warnings for Destructive Actions
**Learning:** Users often scan confirmation prompts quickly. Standard text often fails to convey the severity of destructive actions like deleting caches or data. Using red text for key warning phrases ("This cannot be undone") significantly improves visibility and reduces accidental data loss.
**Action:** Always wrap irreversible warning text in `Colors.red(...)` within confirmation prompts.
