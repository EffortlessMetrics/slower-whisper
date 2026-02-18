"""Deprecated: use ``slower_whisper.pipeline`` instead.

This shim package keeps ``from transcription import ...`` and
``from transcription.xxx import Y`` working during the deprecation window.
All real code now lives under :mod:`slower_whisper.pipeline`.

A :class:`DeprecationWarning` is emitted on every redirected import so
consumers know to update their code.
"""

from __future__ import annotations

# Install the MetaPathFinder *first* so submodule imports are redirected
# before any star-import triggers them.
from transcription._compat_finder import install as _install_compat

_install_compat()

import warnings as _warnings  # noqa: E402

_warnings.warn(
    "Importing from 'transcription' is deprecated. "
    "Use 'slower_whisper.pipeline' instead. "
    "The 'transcription' namespace will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export everything from the real package so ``from transcription import X``
# continues to work.
from slower_whisper.pipeline import *  # noqa: F401, F403, E402
from slower_whisper.pipeline import __all__ as _pipeline_all  # noqa: E402
from slower_whisper.pipeline import __version__  # noqa: E402, F401

__all__ = _pipeline_all
