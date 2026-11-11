"""Read-only previews for the batch CLI, without model or device initialization."""

from __future__ import annotations

import math
import subprocess
import wave
from pathlib import Path

from ..audio_io import index_raw_sources
from ..config import EnrichmentConfig, Paths, TranscriptionConfig
from ..exceptions import EnrichmentError
from ..writers import load_transcript_from_json


def _audio_duration(path: Path) -> float | None:
    """Read metadata only; unavailable or invalid durations remain unknown."""
    try:
        with wave.open(str(path), "rb") as audio:
            return audio.getnframes() / audio.getframerate()
    except (OSError, EOFError, wave.Error, ZeroDivisionError, RuntimeError):
        pass

    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                # Media containers/playlists must not cause network requests.
                "-protocol_whitelist",
                "file,pipe",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path.resolve()),
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode != 0:
            return None
        duration = float(result.stdout.strip())
        return duration if math.isfinite(duration) and duration >= 0 else None
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return None


def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "duration unknown"
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    return f"{minutes}m {secs:02d}s" if minutes else f"{secs}s"


def _print_summary(durations: list[float | None], skipped: int, failed: int) -> int:
    print(f"\nWould process: {len(durations)} file(s)")
    print(f"Would skip: {skipped} file(s)")
    print(f"Would fail: {failed} file(s)")
    known_duration = sum(duration for duration in durations if duration is not None)
    print(f"Total estimated audio duration: {_format_duration(known_duration)}")
    unknown = durations.count(None)
    if unknown:
        print(f"Duration unknown for {unknown} file(s); total includes known durations only.")
    print("Audio duration is not an estimate of processing time.")
    print("No output files were written. Run without --dry-run to process.")
    return 1 if failed else 0


def preview_transcription(paths: Paths, cfg: TranscriptionConfig) -> int:
    """Preview normalization and subsequent work assuming normalization succeeds."""
    raw_sources = index_raw_sources(paths)
    # Map each projected normalized path to the file whose audio it will contain.
    sources = {wav: wav for wav in sorted(paths.norm_dir.glob("*.wav")) if wav.is_file()}
    normalize: dict[Path, Path] = {}
    for src in raw_sources.values():
        dst = paths.norm_dir / f"{src.stem}.wav"
        try:
            up_to_date = dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime
        except OSError:
            up_to_date = False
        if up_to_date:
            sources[dst] = dst
        else:
            normalize[dst] = src
            sources[dst] = src

    print("=== Dry run: Transcription ===")
    print(f"Project root: {paths.root}")
    print("Preview assumes normalization succeeds; models and devices are not initialized.")
    durations: list[float | None] = []
    skipped = failed = 0
    for wav, source in sorted(sources.items()):
        actions = []
        if wav in normalize:
            actions.append(f"normalize {normalize[wav].name}")
        json_path = paths.json_dir / f"{wav.stem}.json"
        if cfg.skip_existing_json and json_path.exists():
            if cfg.enable_chunking or cfg.enable_diarization:
                try:
                    transcript = load_transcript_from_json(json_path)
                    if cfg.enable_chunking:
                        actions.append("update chunks")
                    if cfg.enable_diarization:
                        diarization = (transcript.meta or {}).get("diarization", {})
                        if diarization.get("status") not in {"success", "ok"}:
                            actions.append("diarize existing transcript")
                except Exception as exc:
                    if cfg.enable_diarization:
                        print(f"  {wav.name}: would fail to read {json_path.name}: {exc}")
                        failed += 1
                        continue
                    # The pipeline logs chunk-only failures, then skips ASR.
                    print(f"  {wav.name}: warning: could not update chunks: {exc}")
            if not actions:
                print(f"  {wav.name}: skip (existing JSON)")
                skipped += 1
                continue
            actions.append("skip transcription (existing JSON)")
        else:
            actions.append("transcribe")
            if cfg.enable_diarization:
                actions.append("diarize")
            if cfg.enable_chunking:
                actions.append("build chunks")
        duration = _audio_duration(source)
        durations.append(duration)
        print(f"  {wav.name}: {', '.join(actions)} ({_format_duration(duration)})")
    return _print_summary(durations, skipped, failed)


def preview_enrichment(paths: Paths, cfg: EnrichmentConfig) -> int:
    """Apply the CLI's audio-presence and all-segments-enriched skip rules."""
    if not paths.json_dir.exists():
        raise EnrichmentError(
            f"JSON directory does not exist: {paths.json_dir}. "
            "Run transcription first using transcribe command."
        )
    json_files = sorted(paths.json_dir.glob("*.json"))
    if not json_files:
        raise EnrichmentError(
            f"No JSON transcript files found in {paths.json_dir}. "
            "Run transcription first to generate transcript files."
        )

    print("=== Dry run: Enrichment ===")
    print(f"Project root: {paths.root}")
    print("Models and devices are not initialized.")
    durations: list[float | None] = []
    skipped = failed = 0
    for json_path in json_files:
        wav = paths.norm_dir / f"{json_path.stem}.wav"
        if not wav.exists():
            print(f"  {json_path.name}: would fail (audio file not found: {wav})")
            failed += 1
            continue
        try:
            transcript = load_transcript_from_json(json_path)
        except Exception as exc:
            print(f"  {json_path.name}: would fail to read transcript: {exc}")
            failed += 1
            continue
        if (
            cfg.skip_existing
            and transcript.segments
            and all(segment.audio_state is not None for segment in transcript.segments)
        ):
            print(f"  {json_path.name}: skip (already enriched)")
            skipped += 1
            continue
        duration = _audio_duration(wav)
        durations.append(duration)
        print(
            f"  {json_path.name}: enrich ({len(transcript.segments)} segments, "
            f"{_format_duration(duration)})"
        )
    return _print_summary(durations, skipped, failed)
