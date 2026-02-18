# Privacy Notice

**slower-whisper** is designed as a local-first audio processing tool. This document describes when and how data may leave your machine.

---

## Local-First Defaults

By default, all processing runs entirely on your hardware:

- **ASR (speech-to-text)** — CTranslate2/faster-whisper models run locally
- **Speaker diarization** — pyannote.audio models run locally
- **Audio enrichment** — prosody, emotion, and acoustic features extracted locally
- **Post-processing** — topic segmentation, turn-taking analysis, safety filters all local
- **Semantic analysis (local adapter)** — keyword-based triage with no network calls

No audio data, transcripts, or user content is sent to external services in the default configuration.

---

## When Data Leaves Your Machine

Data is transmitted externally only when you explicitly enable these features:

| Feature | Destination | What is sent |
|---------|-------------|--------------|
| **Cloud semantic adapters** (`openai`, `anthropic`) | OpenAI API / Anthropic API | Transcript text for LLM analysis |
| **Webhook delivery** | User-configured endpoint | Transcript JSON payloads |
| **Model downloads** | HuggingFace Hub, PyPI | HTTP requests to download model weights (no user data sent) |
| **API service mode** | Network clients | Transcripts served over HTTP/WebSocket to connecting clients |

### Model Downloads

On first use, models are downloaded from HuggingFace Hub and cached locally at `~/.cache/huggingface/`. Subsequent runs use the local cache. No user audio or transcript data is included in download requests.

---

## Data Retention

slower-whisper does not manage persistent storage of transcripts or audio. Output files are written to paths you specify. You control:

- Where transcripts are saved
- How long they are retained
- Who has access to output directories

For production deployments, implement retention policies appropriate to your jurisdiction and use case.

---

## PII Handling

slower-whisper includes tools for PII detection and redaction:

- **`slower-whisper privacy`** CLI — scan and redact PII patterns in transcripts
- **Safety post-processor** — configurable filters in the post-processing pipeline
- **Redaction guide** — see [docs/REDACTION.md](REDACTION.md)

These tools assist with compliance but do not guarantee complete PII removal. Review output for your specific requirements.

---

## Consent and Data Controller Responsibilities

If you deploy slower-whisper to process audio from other people:

- **You are the data controller.** slower-whisper is a processing tool, not a service.
- Obtain appropriate consent for audio recording and transcription.
- Comply with applicable privacy regulations (GDPR, CCPA, etc.).
- Implement access controls and audit logging appropriate to your deployment.

---

## CI Security Scanning

The project's CI pipeline includes automated scanning:

- **gitleaks** — scans for accidentally committed secrets
- **detect-secrets** — pre-commit hook for secret detection
- **PII pattern scan** — checks docs/examples for email-like patterns
- **pip-audit** — dependency vulnerability scanning
- **bandit** — static security analysis

---

## Questions

For security concerns, see [SECURITY.md](../SECURITY.md).
For general questions, open a GitHub discussion.
