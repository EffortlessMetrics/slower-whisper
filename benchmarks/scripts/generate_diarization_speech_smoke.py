#!/usr/bin/env python3
"""Generate deterministic speech-based diarization smoke fixtures.

This script creates small two-speaker WAV clips using ffmpeg's built-in
flite voices plus matching RTTM annotations and manifest JSONL rows for:
    benchmarks/data/diarization/
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Utterance:
    speaker_id: str
    voice: str
    text: str


@dataclass(frozen=True)
class SampleSpec:
    sample_id: str
    notes: str
    utterances: tuple[Utterance, ...]


SAMPLES: tuple[SampleSpec, ...] = (
    SampleSpec(
        sample_id="meeting_dual_voice",
        notes="Alternating 2-speaker meeting recap with short pauses.",
        utterances=(
            Utterance("SPEAKER_A", "slt", "Hello team this is Alice with the sprint recap"),
            Utterance("SPEAKER_B", "awb", "Thanks Alice this is Bob and deployment stayed green"),
            Utterance("SPEAKER_A", "slt", "Great please share blockers before tomorrow noon"),
            Utterance("SPEAKER_B", "awb", "Will do I will post updates right after this call"),
        ),
    ),
    SampleSpec(
        sample_id="support_handoff_dual_voice",
        notes="Alternating support handoff with distinct speaker timbre.",
        utterances=(
            Utterance("SPEAKER_A", "slt", "Hi this is Taylor handing over ticket four two one"),
            Utterance("SPEAKER_B", "awb", "Got it this is Morgan I will own the follow up"),
            Utterance("SPEAKER_A", "slt", "Customer asked for written steps and call back"),
            Utterance("SPEAKER_B", "awb", "Understood I will send details and confirm by five"),
        ),
    ),
    SampleSpec(
        sample_id="planning_sync_dual_voice",
        notes="Alternating planning sync with concise turn boundaries.",
        utterances=(
            Utterance("SPEAKER_A", "slt", "For planning we need two reviewers by Wednesday"),
            Utterance("SPEAKER_B", "awb", "I can cover backend and Priya can review frontend"),
            Utterance("SPEAKER_A", "slt", "Perfect then we freeze scope after final approval"),
            Utterance("SPEAKER_B", "awb", "Agreed I will update the board and notify stakeholders"),
        ),
    ),
)


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def _duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as reader:
        frames = reader.getnframes()
        rate = reader.getframerate()
    return float(frames) / float(rate)


def _generate_silence(path: Path, duration_s: float) -> None:
    _run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=r=16000:cl=mono",
            "-t",
            f"{duration_s}",
            str(path),
        ]
    )


def _generate_utterance(path: Path, voice: str, text: str) -> None:
    _run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"flite=text='{text}':voice={voice}",
            "-ar",
            "16000",
            "-ac",
            "1",
            str(path),
        ]
    )


def _concat_audio(parts: list[Path], out_path: Path, tmp_dir: Path) -> None:
    concat_file = tmp_dir / "concat.txt"
    lines = [f"file '{p.name}'\n" for p in parts]
    concat_file.write_text("".join(lines), encoding="utf-8")
    _run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_file),
            "-c",
            "copy",
            str(out_path),
        ]
    )


def _write_rttm(path: Path, sample_id: str, segments: list[tuple[float, float, str]]) -> None:
    lines = []
    for start, duration, speaker_id in segments:
        lines.append(
            f"SPEAKER {sample_id} 1 {start:.3f} {duration:.3f} <NA> <NA> {speaker_id} <NA> <NA>\n"
        )
    path.write_text("".join(lines), encoding="utf-8")


def build_sample(spec: SampleSpec, out_dir: Path, pause_s: float) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix=f"{spec.sample_id}_") as tmp:
        tmp_dir = Path(tmp)
        silence_path = tmp_dir / "silence.wav"
        _generate_silence(silence_path, pause_s)
        pause_duration = _duration_seconds(silence_path)

        audio_parts: list[Path] = []
        turns: list[tuple[float, float, str]] = []
        cursor = 0.0
        for idx, utt in enumerate(spec.utterances):
            utt_path = tmp_dir / f"utt_{idx:02d}.wav"
            _generate_utterance(utt_path, utt.voice, utt.text)
            dur = _duration_seconds(utt_path)

            audio_parts.append(utt_path)
            turns.append((cursor, dur, utt.speaker_id))
            cursor += dur

            if idx < len(spec.utterances) - 1:
                audio_parts.append(silence_path)
                cursor += pause_duration

        out_wav = out_dir / f"{spec.sample_id}.wav"
        out_rttm = out_dir / f"{spec.sample_id}.rttm"
        _concat_audio(audio_parts, out_wav, tmp_dir)
        _write_rttm(out_rttm, spec.sample_id, turns)

        return {
            "id": spec.sample_id,
            "audio_path": f"{spec.sample_id}.wav",
            "reference_rttm": f"{spec.sample_id}.rttm",
            "expected_speaker_count": 2,
            "notes": spec.notes,
            "duration_s": round(_duration_seconds(out_wav), 3),
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="benchmarks/data/diarization",
        help="Directory for wav/rttm/manifest outputs.",
    )
    parser.add_argument(
        "--pause-s",
        type=float,
        default=0.25,
        help="Pause duration inserted between utterances.",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [build_sample(spec, out_dir, pause_s=args.pause_s) for spec in SAMPLES]
    manifest_path = out_dir / "manifest.jsonl"
    manifest_path.write_text(
        "".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows),
        encoding="utf-8",
    )

    print(f"Wrote {len(rows)} samples to {out_dir}")
    print(f"Wrote manifest: {manifest_path}")
    for row in rows:
        print(f"  - {row['id']}: {row['duration_s']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
