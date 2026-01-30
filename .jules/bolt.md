## 2024-05-23 - Audio Extraction File I/O Overhead
**Learning:** `AudioSegmentExtractor` was repeatedly opening and closing `soundfile.SoundFile` handles for every segment extraction, causing significant I/O overhead in batch processing loops like `enrich_transcript_audio`.
**Action:** Implemented context manager protocol for `AudioSegmentExtractor` to allow reusing the file handle across multiple extractions. Always check for repeated I/O in tight loops and prefer keeping handles open when safe.
