"""Transcribe one recording and print its speaker assignments and turns."""

import argparse
import sys
from pathlib import Path

from transcription import TranscriptionConfig, transcribe_file
from transcription.speaker_id import get_speaker_id


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path, help="Audio file to transcribe")
    parser.add_argument(
        "--root", required=True, type=Path, help="Dedicated output project directory"
    )
    args = parser.parse_args()
    if not args.audio.is_file():
        parser.error(f"Audio file does not exist: {args.audio}")

    config = TranscriptionConfig(
        model="base",
        device="cpu",
        compute_type="int8",
        enable_diarization=True,
        diarization_device="cpu",
    )
    transcript = transcribe_file(audio_path=args.audio, root=args.root, config=config)
    diarization = (transcript.meta or {}).get("diarization", {})
    if diarization.get("status") != "ok":
        print(
            f"Diarization {diarization.get('status', 'unknown')}: "
            f"{diarization.get('error') or diarization.get('message') or 'No details'}",
            file=sys.stderr,
        )
        return 1

    print("Segments (recording-local speaker IDs)")
    for segment in transcript.segments:
        speaker = get_speaker_id(segment.speaker) or "unassigned"
        print(f"{segment.start:.2f}–{segment.end:.2f}s  {speaker}  {segment.text.strip()}")

    print("Speaker turns")
    for turn in transcript.turns or []:
        print(f"{turn['start']:.2f}–{turn['end']:.2f}s  {turn['speaker_id']}  {turn['text']}")
    if not transcript.turns:
        print("No labeled turns.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
