"""Safe caller-facing source identity for in-memory and uploaded audio."""

from __future__ import annotations

from pathlib import Path, PurePosixPath

from .audio_io import sanitize_filename

_MAX_SOURCE_NAME_CHARS = 255
_MAX_SOURCE_SUFFIX_CHARS = 32


def safe_source_name(
    filename: str | None,
    *,
    fallback_suffix: str = "",
) -> str:
    """Return a bounded basename without trusting caller path components.

    This value is public transcript identity, not a filesystem destination and
    not artifact provenance. POSIX and Windows separators are collapsed before
    sanitization; control characters and pathological names fall back to
    ``audio`` plus the validated fallback suffix.
    """
    _, safe_fallback_suffix = sanitize_filename(
        "audio",
        fallback_suffix,
        default="audio",
    )
    safe_fallback_suffix = safe_fallback_suffix[:_MAX_SOURCE_SUFFIX_CHARS]
    fallback = f"audio{safe_fallback_suffix}"
    if not filename:
        return fallback

    normalized = filename.replace("\\", "/")
    basename = PurePosixPath(normalized).name
    basename = "".join(character for character in basename if character.isprintable())
    basename = basename.strip()
    if basename in {"", ".", ".."}:
        return fallback

    path = Path(basename)
    safe_stem, safe_suffix = sanitize_filename(
        path.stem,
        path.suffix,
        default="audio",
    )
    if not safe_suffix:
        safe_suffix = safe_fallback_suffix
    safe_suffix = safe_suffix[:_MAX_SOURCE_SUFFIX_CHARS]

    stem_limit = max(1, _MAX_SOURCE_NAME_CHARS - len(safe_suffix))
    return f"{safe_stem[:stem_limit]}{safe_suffix}"
