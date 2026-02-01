"""
UX tests for the CLI.
"""

from transcription.color_utils import Colors, Symbols
from transcription.telemetry import CheckStatus, DoctorCheck, DoctorReport, format_doctor_report


def test_doctor_report_symbols(monkeypatch):
    """Test that doctor report uses symbols when color is enabled."""
    monkeypatch.setenv("FORCE_COLOR", "1")
    report = DoctorReport(
        checks=[
            DoctorCheck("Test Check", CheckStatus.PASS, "All good"),
            DoctorCheck("Warning Check", CheckStatus.WARN, "Be careful"),
            DoctorCheck("Failed Check", CheckStatus.FAIL, "Something broke"),
            DoctorCheck("Skipped Check", CheckStatus.SKIP, "Not applicable"),
        ]
    )

    # Test with color enabled
    output = format_doctor_report(report, use_color=True)

    assert Symbols.CHECK in output
    assert Symbols.WARNING in output
    assert Symbols.CROSS in output
    assert Symbols.DOT in output

    # Verify colors are also present
    assert Colors.GREEN in output
    assert Colors.YELLOW in output
    assert Colors.RED in output


def test_doctor_report_no_symbols_without_color():
    """Test that doctor report falls back to text labels when color is disabled."""
    report = DoctorReport(
        checks=[
            DoctorCheck("Test Check", CheckStatus.PASS, "All good"),
        ]
    )

    # Test with color disabled
    output = format_doctor_report(report, use_color=False)

    assert Symbols.CHECK not in output
    assert "[PASS]" in output
