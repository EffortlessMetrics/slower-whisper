"""
Tests for the Symbols class in transcription.color_utils.
"""

from transcription.color_utils import Symbols


class TestSymbols:
    """Test the Symbols class."""

    def test_symbols_are_present(self) -> None:
        """Verify that all expected symbols are present and are strings."""
        assert isinstance(Symbols.CHECK, str)
        assert isinstance(Symbols.CROSS, str)
        assert isinstance(Symbols.INFO, str)
        assert isinstance(Symbols.WARN, str)
        assert isinstance(Symbols.ARROW, str)
        assert isinstance(Symbols.DOT, str)

    def test_symbols_values(self) -> None:
        """Verify the values of the symbols."""
        assert Symbols.CHECK == "✔"
        assert Symbols.CROSS == "✘"
        assert Symbols.INFO == "ℹ"
        assert Symbols.WARN == "⚠"
        assert Symbols.ARROW == "➜"
        assert Symbols.DOT == "•"
