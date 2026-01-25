# Sentinel's Journal

## 2024-05-22 - Prevent Memory Exhaustion DoS in File Uploads
**Vulnerability:** Endpoints `transcribe_audio` and `enrich_audio` were reading entire uploaded files into memory (`await audio.read()`) before validation, allowing a Denial of Service (DoS) attack via large file uploads.
**Learning:** FastAPI's `UploadFile` (via Starlette) buffers large files to disk, but calling `.read()` loads the entire content into RAM. Size validation must happen *during* the read process, not after.
**Prevention:** Use a chunked streaming approach to write the file to disk, checking the cumulative size against the limit after each chunk. Always ensure file handles are closed before attempting to delete partial files (crucial for Windows compatibility).
