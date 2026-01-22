## 2026-01-22 - Micro-UX: The "What's Next" Pattern

**Learning:** CLI tools often complete a long-running process (like transcription) and exit silently or with just stats, leaving the user to remember the next command. Adding a "Next Steps" section pointing to the logical next stage in the pipeline (e.g., `enrich` after `transcribe`) significantly reduces cognitive load and reinforces the intended workflow.

**Action:** Whenever building multi-stage CLI tools, always include a "Next steps:" suggestion in the success summary of each stage.
