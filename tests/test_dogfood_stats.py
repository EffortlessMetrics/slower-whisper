import json
from pathlib import Path

import pytest

from transcription.dogfood_utils import compute_diarization_stats, print_diarization_stats


def synthetic_stats(tmp_path: Path, status: str | None, speakers: list[str | None]):
    path = tmp_path / "synthetic.json"
    path.write_text(
        json.dumps(
            {
                "file": "synthetic.wav",
                "meta": {"diarization": {"status": status}} if status else {},
                "segments": [
                    {"id": i, "start": i, "end": i + 1, "text": "Synthetic.", "speaker": speaker}
                    for i, speaker in enumerate(speakers)
                ],
            }
        )
    )
    return compute_diarization_stats(path)


@pytest.mark.parametrize(
    ("status", "speakers"),
    [("skipped", [None]), ("disabled", [None]), ("error", ["A", "B"]), (None, [None])],
)
def test_unavailable_diarization(tmp_path, capsys, status, speakers):
    stats = synthetic_stats(tmp_path, status, speakers)
    assert stats["diarization_status"] == status
    assert stats["quality_checks"] == {
        "speaker_count_reasonable": None,
        "good_labeling_coverage": None,
        "good_turn_alternation": None,
    }
    print_diarization_stats(stats)
    output = capsys.readouterr().out
    assert f"Diarization status: {status or 'unreported'}" in output
    assert "N/A Good turn alternation" in output
    assert "✓" not in output and "✗" not in output


@pytest.mark.parametrize(
    ("status", "speakers", "coverage", "alternation"),
    [
        ("ok", ["A", "B"], True, True),
        (None, ["A", "A"], True, False),
        ("success", ["A", None], False, None),
        ("ok", [None], False, None),
    ],
)
def test_assess_diarization_with_evidence(tmp_path, status, speakers, coverage, alternation):
    stats = synthetic_stats(tmp_path, status, speakers)
    assert stats["quality_checks"] == {
        "speaker_count_reasonable": any(speakers),
        "good_labeling_coverage": coverage,
        "good_turn_alternation": alternation,
    }
