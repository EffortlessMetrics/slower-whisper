import argparse
from pathlib import Path

from .config import AppConfig, AsrConfig, Paths
from .pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Local transcription pipeline (ffmpeg + faster-whisper)"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(),
        help="Root directory containing raw_audio/ etc. (default: current directory)",
    )
    parser.add_argument(
        "--model",
        default="large-v3",
        help="Whisper model name (default: large-v3)",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device for inference (cuda or cpu, default: cuda)",
    )
    parser.add_argument(
        "--compute-type",
        default="float16",
        help="Compute type for faster-whisper (e.g. float16, int8_float16).",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Force language code (e.g. en, es) instead of auto-detect.",
    )
    parser.add_argument(
        "--task",
        default="transcribe",
        choices=["transcribe", "translate"],
        help="Task to perform: transcribe (default) or translate to English.",
    )
    parser.add_argument(
        "--vad-min-silence-ms",
        type=int,
        default=500,
        help="Minimum silence duration in ms to split segments (default: 500).",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=5,
        help="Beam size for decoding (default: 5).",
    )
    parser.add_argument(
        "--skip-existing-json",
        action="store_true",
        help="If set, skip transcription for files that already have a JSON output.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    paths = Paths(root=args.root)
    asr_cfg = AsrConfig(
        model_name=args.model,
        device=args.device,
        compute_type=args.compute_type,
        vad_min_silence_ms=args.vad_min_silence_ms,
        beam_size=args.beam_size,
        language=args.language,
        task=args.task,
    )
    cfg = AppConfig(paths=paths, asr=asr_cfg, skip_existing_json=args.skip_existing_json)

    run_pipeline(cfg)


if __name__ == "__main__":
    main()
