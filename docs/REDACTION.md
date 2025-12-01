# Redaction (best-effort)

Cloud APIs ship PII redaction; this repo stays lightweight but provides a starter script.

## Example script

```bash
python examples/redaction/redact_transcript.py transcripts/call.json --output transcripts/call.redacted.json
```

What it does:
- Masks emails (`[EMAIL]`), phone-like numbers (`[PHONE]`), and credit-card-like sequences (`[CARD]`) with regexes.
- Redacts both segments and turn text; leaves metadata untouched.

## Caveats

- This is **not** a compliance feature. Regexes will miss edge cases and may over-redact numbers in normal text.
- If you need stronger guarantees, add domain-specific regexes or a second pass with an LLM redactor before storing/serving the data.

## Production checklist

If you use this in production, you **must**:

- Review and tune regexes for your jurisdiction and data formats.
- Apply redaction **before** exporting to CSV/HTML or sharing transcripts.
- Validate the redacted file with `slower-whisper validate` to catch schema breaks.
