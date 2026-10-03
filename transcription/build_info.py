"""Validated access to package-local build identity.

Runtime code never shells out to git and never reads identity from the caller's
working directory or environment. Unknown build fields remain unknown.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from . import _build_info

_SOURCE_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{7,64}$")
_BUILD_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


@dataclass(frozen=True, slots=True)
class PackageBuildInfo:
    """Trusted identity embedded in the installed package."""

    source_commit: str | None
    build_id: str | None

    def to_dict(self) -> dict[str, str]:
        result: dict[str, str] = {}
        if self.source_commit is not None:
            result["source_commit"] = self.source_commit
        if self.build_id is not None:
            result["build_id"] = self.build_id
        return result


def _validated_optional_string(
    value: object,
    *,
    pattern: re.Pattern[str],
) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    candidate = value.strip()
    if not candidate or pattern.fullmatch(candidate) is None:
        return None
    return candidate


def get_package_build_info() -> PackageBuildInfo:
    """Return validated identity from the package-local generated module."""
    return PackageBuildInfo(
        source_commit=_validated_optional_string(
            getattr(_build_info, "SOURCE_COMMIT", None),
            pattern=_SOURCE_COMMIT_PATTERN,
        ),
        build_id=_validated_optional_string(
            getattr(_build_info, "BUILD_ID", None),
            pattern=_BUILD_ID_PATTERN,
        ),
    )


def get_source_commit() -> str | None:
    """Return the trusted package source commit, or ``None`` when unknown."""
    return get_package_build_info().source_commit


def get_build_id() -> str | None:
    """Return the trusted package build identifier, or ``None`` when unknown."""
    return get_package_build_info().build_id
