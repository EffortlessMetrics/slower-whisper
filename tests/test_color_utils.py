from slower_whisper.pipeline.color_utils import Colors


def test_colors_enabled_forced(monkeypatch):
    """Test that colors are enabled when FORCE_COLOR is set."""
    monkeypatch.setenv("FORCE_COLOR", "1")
    monkeypatch.delenv("NO_COLOR", raising=False)

    assert Colors._should_use_color() is True
    assert Colors.red("test") == "\033[31mtest\033[0m"


def test_colors_disabled_no_color(monkeypatch):
    """Test that colors are disabled when NO_COLOR is set."""
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.delenv("FORCE_COLOR", raising=False)

    assert Colors._should_use_color() is False
    assert Colors.red("test") == "test"


def test_colors_disabled_dumb_term(monkeypatch):
    """Test that colors are disabled when TERM is dumb."""
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    monkeypatch.setenv("TERM", "dumb")

    assert Colors._should_use_color() is False
    assert Colors.red("test") == "test"


def test_colors_methods(monkeypatch):
    """Test all color methods produce correct output."""
    monkeypatch.setenv("FORCE_COLOR", "1")
    monkeypatch.delenv("NO_COLOR", raising=False)

    assert Colors.green("test") == "\033[32mtest\033[0m"
    assert Colors.yellow("test") == "\033[33mtest\033[0m"
    assert Colors.blue("test") == "\033[34mtest\033[0m"
    assert Colors.cyan("test") == "\033[36mtest\033[0m"
    assert Colors.magenta("test") == "\033[35mtest\033[0m"
    assert Colors.bold("test") == "\033[1mtest\033[0m"
    assert Colors.dim("test") == "\033[2mtest\033[0m"
