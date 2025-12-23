"""Tests that package imports don't eagerly load heavy optional dependencies.

These tests ensure that base installs (without full/enrichment extras) can
import the transcription package and use the CLI without ImportError.

The lazy import pattern (PEP 562) is used for streaming_enrich and
streaming_semantic modules which depend on soundfile and other heavy deps.
"""

from __future__ import annotations

import sys


def test_transcription_import_does_not_eager_import_streaming_enrich() -> None:
    """Importing transcription should not eagerly import streaming_enrich.

    This ensures base Docker images can run `slower-whisper --version`
    without needing soundfile and other enrichment dependencies.
    """
    # Clear any cached imports
    sys.modules.pop("transcription.streaming_enrich", None)
    sys.modules.pop("transcription.audio_enrichment", None)
    sys.modules.pop("transcription.audio_utils", None)

    # Force reimport of transcription package
    if "transcription" in sys.modules:
        # If already imported, we need to check it didn't import streaming_enrich
        # The module should NOT be in sys.modules after a fresh import
        pass

    import transcription  # noqa: F401

    # These modules should NOT have been imported just from `import transcription`
    assert "transcription.streaming_enrich" not in sys.modules, (
        "streaming_enrich was eagerly imported; this breaks base Docker images"
    )
    assert "transcription.audio_enrichment" not in sys.modules, (
        "audio_enrichment was eagerly imported; this breaks base Docker images"
    )


def test_transcription_import_does_not_eager_import_streaming_semantic() -> None:
    """Importing transcription should not eagerly import streaming_semantic."""
    sys.modules.pop("transcription.streaming_semantic", None)

    import transcription  # noqa: F401

    assert "transcription.streaming_semantic" not in sys.modules, (
        "streaming_semantic was eagerly imported"
    )


def test_lazy_imports_appear_in_dir() -> None:
    """Lazy imports should appear in dir(transcription) for discoverability."""
    import transcription

    exports = dir(transcription)

    # These should be discoverable even though they're lazy
    assert "StreamingEnrichmentConfig" in exports
    assert "StreamingEnrichmentSession" in exports
    assert "LiveSemanticsConfig" in exports
    assert "LiveSemanticSession" in exports
    assert "SemanticUpdatePayload" in exports
    assert "run_pipeline" in exports


def test_lazy_imports_not_in_all() -> None:
    """Lazy optional imports should NOT be in __all__.

    This is critical: `from transcription import *` must succeed on base installs.
    If lazy imports are in __all__, Python will try to resolve them, causing
    ImportError when optional deps (soundfile, etc.) aren't installed.
    """
    import transcription

    # These must NOT be in __all__ to prevent base install breakage
    assert "StreamingEnrichmentConfig" not in transcription.__all__, (
        "StreamingEnrichmentConfig in __all__ breaks base installs"
    )
    assert "StreamingEnrichmentSession" not in transcription.__all__, (
        "StreamingEnrichmentSession in __all__ breaks base installs"
    )
    assert "LiveSemanticsConfig" not in transcription.__all__, (
        "LiveSemanticsConfig in __all__ breaks base installs"
    )
    assert "LiveSemanticSession" not in transcription.__all__, (
        "LiveSemanticSession in __all__ breaks base installs"
    )
    assert "SemanticUpdatePayload" not in transcription.__all__, (
        "SemanticUpdatePayload in __all__ breaks base installs"
    )
    assert "run_pipeline" not in transcription.__all__, (
        "run_pipeline in __all__ (lazy for circular import avoidance)"
    )
