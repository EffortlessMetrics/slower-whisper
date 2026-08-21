"""Package-local build identity.

Official artifact builds replace these values through
``scripts/write_build_info.py`` before constructing the wheel and sdist.
Source checkouts and unlabelled local builds deliberately remain unknown.
"""

BUILD_INFO_VERSION = 1
SOURCE_COMMIT: str | None = None
BUILD_ID: str | None = None
