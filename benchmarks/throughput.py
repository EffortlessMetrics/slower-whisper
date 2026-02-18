#!/usr/bin/env python3
"""Minimal throughput probe for slower-whisper."""

from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path

from slower_whisper.pipeline import (
    EnrichmentConfig,
    TranscriptionConfig,
    enrich_transcript,
    transcribe_file,
)
from slower_whisper.pipeline.audio_utils import AudioSegmentExtractor


def _pick_audio(explicit: str | Path | None) -> Path:
    if explicit:
        chosen = Path(explicit)
        if chosen.exists():
            return chosen
        raise SystemExit(f"Audio file not found: {chosen}")

    candidates = [
        Path("raw_audio/test_sample.wav"),
        Path("benchmarks/test_audio/test_audio_30s.wav"),
        Path("benchmarks/test_audio/test_audio_10s.wav"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise SystemExit(
        "No sample audio found. Provide --audio pointing to a WAV/MP3/FLAC file to benchmark."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Measure slower-whisper throughput on a sample audio file.",
    )
    parser.add_argument("--audio", type=Path, help="Path to audio file to benchmark.")
    parser.add_argument("--model", default="base", help="Whisper model name (default: base).")
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device to run ASR on (default: cpu).",
    )
    parser.add_argument(
        "--compute-type",
        default=None,
        help="Optional compute_type override for faster-whisper (e.g., float16, int8_float16).",
    )
    parser.add_argument(
        "--skip-enrich",
        action="store_true",
        help="Skip Stage 2 enrichment (only measure ASR).",
    )
    parser.add_argument(
        "--enable-emotion",
        action="store_true",
        help="Enable emotion models during enrichment (off by default for speed).",
    )
    parser.add_argument(
        "--enable-semantic-annotator",
        action="store_true",
        help="Run the keyword semantic annotator during enrichment.",
    )
    parser.add_argument(
        "--enrich-device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device to run enrichment on (default: cpu).",
    )
    args = parser.parse_args(argv)

    audio_path = _pick_audio(args.audio)
    print(f"🔊 Using audio: {audio_path}")

    with tempfile.TemporaryDirectory(prefix="sw-throughput-") as tmpdir:
        tmp_root = Path(tmpdir)
        tx_cfg = TranscriptionConfig(
            model=args.model,
            device=args.device,
            compute_type=args.compute_type,
            skip_existing_json=False,
        )

        t0 = time.perf_counter()
        transcript = transcribe_file(audio_path, root=tmp_root, config=tx_cfg)
        transcribe_time = time.perf_counter() - t0

        normalized = tmp_root / "input_audio" / f"{audio_path.stem}.wav"
        duration = AudioSegmentExtractor(normalized).get_duration()

        enrich_time: float | None = None
        if not args.skip_enrich:
            enrich_cfg = EnrichmentConfig(
                skip_existing=False,
                enable_prosody=True,
                enable_emotion=args.enable_emotion,
                enable_categorical_emotion=False,
                enable_turn_metadata=True,
                enable_speaker_stats=True,
                enable_semantic_annotator=args.enable_semantic_annotator,
                device=args.enrich_device,
            )
            t1 = time.perf_counter()
            try:
                enrich_transcript(transcript, normalized, enrich_cfg)
                enrich_time = time.perf_counter() - t1
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] Enrichment failed or missing deps: {exc}")

        total_time = transcribe_time + (enrich_time or 0.0)
        rtf_asr = duration / transcribe_time if transcribe_time > 0 else 0.0
        rtf_total = duration / total_time if total_time > 0 else 0.0

        print("\nThroughput summary")
        print("------------------")
        print(f"Audio duration:      {duration:6.2f} s")
        print(f"ASR wall time:       {transcribe_time:6.2f} s  ({rtf_asr:4.2f}x realtime)")
        if enrich_time is not None:
            print(f"Enrich wall time:    {enrich_time:6.2f} s")
            print(f"ASR+enrich wall:     {total_time:6.2f} s  ({rtf_total:4.2f}x realtime)")
        else:
            print("Enrich wall time:    skipped")
            print(f"ASR wall (only):     {transcribe_time:6.2f} s  ({rtf_asr:4.2f}x realtime)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
