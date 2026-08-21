"""Build-time provenance embedded in the installed package.

Release/build automation may replace these values while constructing an
artifact. Runtime code must not infer slower-whisper provenance from the
caller's current working directory or repository.
"""

SOURCE_COMMIT: str | None = None
BUILD_ID: str | None = None
