"""Tests for the Symbols class in color_utils.py."""

from transcription.color_utils import Colors, Symbols


def test_symbols_with_color(monkeypatch):
    """Test symbols return colored unicode when color is enabled."""
    monkeypatch.setenv("FORCE_COLOR", "1")
    monkeypatch.delenv("NO_COLOR", raising=False)

    assert Colors.should_use_color() is True
    assert Symbols.check(use_color=True) == Colors.green(Symbols.CHECK)
    assert Symbols.cross(use_color=True) == Colors.red(Symbols.CROSS)
    assert Symbols.warn(use_color=True) == Colors.yellow(Symbols.WARN)
    assert Symbols.skip(use_color=True) == Colors.dim(Symbols.SKIP)
    assert Symbols.arrow(use_color=True) == Colors.cyan(Symbols.ARROW)


def test_symbols_without_color(monkeypatch):
    """Test symbols return ASCII fallbacks when color is disabled."""
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.delenv("FORCE_COLOR", raising=False)

    assert Colors.should_use_color() is False
    assert Symbols.check(use_color=True) == "[v]"
    assert Symbols.cross(use_color=True) == "[x]"
    assert Symbols.warn(use_color=True) == "[!]"
    assert Symbols.skip(use_color=True) == "[-]"
    assert Symbols.arrow(use_color=True) == "->"


def test_symbols_explicit_no_color(monkeypatch):
    """Test symbols return ASCII fallbacks when use_color=False is passed."""
    monkeypatch.setenv("FORCE_COLOR", "1")  # Color is enabled in environment

    assert Colors.should_use_color() is True
    assert Symbols.check(use_color=False) == "[v]"
    assert Symbols.cross(use_color=False) == "[x]"
    assert Symbols.warn(use_color=False) == "[!]"
    assert Symbols.skip(use_color=False) == "[-]"
    assert Symbols.arrow(use_color=False) == "->"
