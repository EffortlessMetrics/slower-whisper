"""Compatibility finder for the deprecated ``transcription`` namespace.

This module installs a :class:`importlib.abc.MetaPathFinder` that redirects
``from transcription.xxx import Y`` to ``from slower_whisper.pipeline.xxx import Y``
while emitting a :class:`DeprecationWarning` so library consumers know to update
their imports.

The finder is installed once via :func:`install` and is idempotent.
"""

from __future__ import annotations

import importlib
import sys
import warnings
from importlib.abc import MetaPathFinder
from importlib.machinery import ModuleSpec
from types import ModuleType
from typing import Any

_INSTALLED = False
_PREFIX = "transcription."
_TARGET = "slower_whisper.pipeline."


class _CompatLoader:
    """Loader that redirects to the real module under slower_whisper.pipeline."""

    def __init__(self, real_name: str, compat_name: str) -> None:
        self._real_name = real_name
        self._compat_name = compat_name

    def create_module(self, spec: ModuleSpec) -> ModuleType | None:  # noqa: ARG002
        return None  # use default semantics

    def exec_module(self, module: ModuleType) -> None:
        warnings.warn(
            f"Importing from '{self._compat_name}' is deprecated. "
            f"Use '{self._real_name}' instead. "
            "The 'transcription' namespace will be removed in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )
        real = importlib.import_module(self._real_name)

        # Copy all attributes from real module to compat module
        module.__dict__.update(real.__dict__)
        module.__spec__ = module.__spec__  # keep compat spec
        module.__name__ = self._compat_name
        module.__loader__ = self  # type: ignore[assignment]

        # Also register in sys.modules so subsequent imports are fast
        sys.modules[self._compat_name] = module


class _CompatFinder(MetaPathFinder):
    """MetaPathFinder that intercepts ``transcription.*`` imports."""

    def find_module(self, fullname: str, path: Any = None) -> Any:  # noqa: ARG002
        # Legacy protocol — delegate to find_spec
        return None

    def find_spec(
        self,
        fullname: str,
        path: Any = None,  # noqa: ARG002
        target: Any = None,  # noqa: ARG002
    ) -> ModuleSpec | None:
        if not fullname.startswith(_PREFIX):
            return None

        # Already resolved — skip to avoid recursion
        if fullname in sys.modules:
            return None

        suffix = fullname[len(_PREFIX) :]
        real_name = _TARGET + suffix

        return ModuleSpec(
            fullname,
            _CompatLoader(real_name, fullname),  # type: ignore[arg-type]
            origin=f"compat-redirect:{real_name}",
        )


def install() -> None:
    """Install the compatibility finder into ``sys.meta_path`` (idempotent)."""
    global _INSTALLED  # noqa: PLW0603
    if _INSTALLED:
        return
    sys.meta_path.insert(0, _CompatFinder())
    _INSTALLED = True
