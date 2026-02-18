"""Tests for the ``transcription`` → ``slower_whisper.pipeline`` deprecation shim.

These tests verify that old-style imports continue to work while emitting
:class:`DeprecationWarning`, and that the shim redirects correctly to the
real modules under ``slower_whisper.pipeline``.
"""

from __future__ import annotations

import importlib
import sys
import warnings

import pytest  # noqa: F401


class TestTopLevelShim:
    """Test ``from transcription import X`` via the shim __init__.py."""

    def test_import_transcription_config(self):
        """Importing TranscriptionConfig via transcription shim works and warns."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", DeprecationWarning)
            # Force re-import to trigger warning
            if "transcription" in sys.modules:
                del sys.modules["transcription"]
            mod = importlib.import_module("transcription")
            TranscriptionConfig = mod.TranscriptionConfig  # noqa: N806

        assert TranscriptionConfig is not None
        deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(deprecation_warnings) >= 1
        assert "deprecated" in str(deprecation_warnings[0].message).lower()

    def test_version_matches(self):
        """transcription shim __version__ matches slower_whisper.__version__."""
        import slower_whisper
        import slower_whisper.pipeline

        # Re-import transcription to get shim
        if "transcription" in sys.modules:
            del sys.modules["transcription"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            mod = importlib.import_module("transcription")

        assert mod.__version__ == slower_whisper.__version__
        assert mod.__version__ == slower_whisper.pipeline.__version__


class TestSubmoduleShim:
    """Test ``from transcription.xxx import Y`` via the MetaPathFinder."""

    def test_import_models_segment(self):
        """Importing Segment via transcription.models shim works and warns."""
        # Clear cached module to force fresh import through finder
        sys.modules.pop("transcription.models", None)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", DeprecationWarning)
            mod = importlib.import_module("transcription.models")

        # The module should have Segment
        assert hasattr(mod, "Segment")
        assert hasattr(mod, "Transcript")

        # Should have emitted a deprecation warning
        deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(deprecation_warnings) >= 1
        assert "slower_whisper.pipeline.models" in str(deprecation_warnings[0].message)

    def test_import_models_transcript(self):
        """Importing Transcript via transcription.models shim works and warns."""
        sys.modules.pop("transcription.models", None)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", DeprecationWarning)
            mod = importlib.import_module("transcription.models")
            Transcript = mod.Transcript  # noqa: N806

        from slower_whisper.pipeline.models import Transcript as RealTranscript

        assert Transcript is RealTranscript
        deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(deprecation_warnings) >= 1

    def test_deep_submodule_historian(self):
        """Importing via transcription.historian shim works."""
        sys.modules.pop("transcription.historian", None)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", DeprecationWarning)
            mod = importlib.import_module("transcription.historian")

        assert hasattr(mod, "gather_pr_data")
        assert hasattr(mod, "FactBundle")
        deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(deprecation_warnings) >= 1


class TestDirectPipelineImport:
    """Test that direct imports from slower_whisper.pipeline work without warnings."""

    def test_no_warning_on_direct_import(self):
        """Direct import from slower_whisper.pipeline.models emits no DeprecationWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", DeprecationWarning)
            from slower_whisper.pipeline.models import Segment  # noqa: F401

        deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(deprecation_warnings) == 0

    def test_pipeline_reexports_from_top_level(self):
        """Top-level slower_whisper re-exports pipeline functions."""
        from slower_whisper import transcribe_file

        assert callable(transcribe_file)

    def test_compat_types_at_top_level(self):
        """Top-level slower_whisper exports faster-whisper compat types."""
        from slower_whisper import Segment, WhisperModel, Word

        assert WhisperModel is not None
        assert Segment is not None
        assert Word is not None
