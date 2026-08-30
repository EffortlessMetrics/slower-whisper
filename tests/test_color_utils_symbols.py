"""Tests for the Symbols class in color_utils."""

from unittest.mock import patch

from transcription.color_utils import Colors, Symbols


def test_symbols_unicode_when_color_enabled():
    """Symbols should return Unicode characters when color is enabled."""
    with patch.object(Colors, "_should_use_color", return_value=True):
        assert "✔" in Symbols.check()
        assert "✖" in Symbols.cross()
        assert "⚠" in Symbols.warn()
        assert "ℹ" in Symbols.info()
        assert "➜" in Symbols.arrow()
        assert "•" in Symbols.dot()


def test_symbols_ascii_when_color_disabled():
    """Symbols should return ASCII fallbacks when color is disabled."""
    with patch.object(Colors, "_should_use_color", return_value=False):
        assert Symbols.check() == "[v]"
        assert Symbols.cross() == "[x]"
        assert Symbols.warn() == "[!]"
        assert Symbols.info() == "[i]"
        assert Symbols.arrow() == "->"
        assert Symbols.dot() == "*"


def test_symbols_override_true():
    """Symbols should respect override=True."""
    # Even if global is False
    with patch.object(Colors, "_should_use_color", return_value=False):
        assert "✔" in Symbols.check(use_color=True)


def test_symbols_override_false():
    """Symbols should respect override=False."""
    # Even if global is True
    with patch.object(Colors, "_should_use_color", return_value=True):
        assert Symbols.check(use_color=False) == "[v]"
