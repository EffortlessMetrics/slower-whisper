import time
import wave
from pathlib import Path

from . import __version__ as PIPELINE_VERSION
from . import audio_io, writers
from .asr_engine import TranscriptionEngine
from .config import AppConfig, TranscriptionConfig
from .meta_utils import build_generation_metadata
from .models import Transcript


def _get_duration_seconds(path: Path) -> float:
    """
    Return the duration of a WAV file in seconds.

    Assumes path points to a valid WAV file.
    """
    try:
        with wave.open(str(path), "rb") as w:
            frames = w.getnframes()
            rate = w.getframerate()
        return frames / float(rate) if rate else 0.0
    except Exception as e:
        print(f"[warn] Could not read duration for {path.name}: {e}")
        return 0.0


def _build_meta(
    cfg: AppConfig, transcript: Transcript, audio_path: Path, duration_sec: float
) -> dict:
    """
    Build a metadata dictionary describing this transcript generation run.
    """
    return build_generation_metadata(
        transcript,
        duration_sec=duration_sec,
        model_name=cfg.asr.model_name,
        config_device=cfg.asr.device,
        config_compute_type=cfg.asr.compute_type,
        beam_size=cfg.asr.beam_size,
        vad_min_silence_ms=cfg.asr.vad_min_silence_ms,
        language_hint=cfg.asr.language,
        task=cfg.asr.task,
        pipeline_version=PIPELINE_VERSION,
        root=cfg.paths.root,
    )


def run_pipeline(cfg: AppConfig, diarization_config: TranscriptionConfig | None = None) -> None:
    """
    Orchestrate the full pipeline:
    1) Ensure directories.
    2) Normalize raw audio to 16 kHz mono WAV.
    3) Transcribe normalized audio with Whisper.
    4) (v1.1) Optionally run diarization if diarization_config.enable_diarization=True.
    5) Write JSON, TXT, and SRT outputs per file.

    If cfg.skip_existing_json is True, files that already have a JSON
    output will be skipped at the transcription step.

    Args:
        cfg: AppConfig (internal pipeline config).
        diarization_config: Optional TranscriptionConfig with diarization settings.
                           If provided and enable_diarization=True, diarization
                           will run on each transcript before writing outputs.
    """
    paths = cfg.paths

    audio_io.ensure_dirs(paths)
    audio_io.normalize_all(paths)

    norm_files = sorted(paths.norm_dir.glob("*.wav"))
    if not norm_files:
        print("No .wav files found in input_audio/. Nothing to transcribe.")
        return

    engine = TranscriptionEngine(cfg.asr)

    print("\n=== Step 3: Transcribing normalized audio ===")
    total = len(norm_files)
    processed = skipped = failed = 0
    diarized_only = 0
    total_audio = 0.0
    total_time = 0.0

    for idx, wav in enumerate(norm_files, start=1):
        print(f"[{idx}/{total}] {wav.name}")
        stem = Path(wav.name).stem
        json_path = paths.json_dir / f"{stem}.json"
        txt_path = paths.transcripts_dir / f"{stem}.txt"
        srt_path = paths.transcripts_dir / f"{stem}.srt"

        if cfg.skip_existing_json and json_path.exists():
            if diarization_config and diarization_config.enable_diarization:
                # Upgrade existing transcript with diarization without re-transcribing
                try:
                    transcript = writers.load_transcript_from_json(json_path)
                except Exception as exc:
                    print(f"[error-diarization] Failed to load {json_path.name}: {exc}")
                    failed += 1
                    continue

                diar_meta = (transcript.meta or {}).get("diarization", {})
                if diar_meta.get("status") == "success":
                    print(
                        f"[skip-transcribe] {wav.name} because {json_path.name} already exists "
                        "(diarization present)"
                    )
                    skipped += 1
                    continue

                print(f"[diarize-existing] {wav.name} (reusing existing transcript)")
                try:
                    from .api import _maybe_run_diarization

                    transcript = _maybe_run_diarization(
                        transcript=transcript,
                        wav_path=wav,
                        config=diarization_config,
                    )
                except Exception as exc:
                    print(f"[error-diarization] Failed to run diarization for {wav.name}: {exc}")
                    failed += 1
                    continue

                writers.write_json(transcript, json_path)
                writers.write_txt(transcript, txt_path)
                writers.write_srt(transcript, srt_path)

                diarized_only += 1
                print(f"  → [diarization-only] {json_path}")
                print(f"  → [diarization-only] {txt_path}")
                print(f"  → [diarization-only] {srt_path}")
            else:
                print(f"[skip-transcribe] {wav.name} because {json_path.name} already exists")
                skipped += 1
            continue

        duration = _get_duration_seconds(wav)
        total_audio += duration

        start = time.time()
        try:
            transcript = engine.transcribe_file(wav)
        except Exception as e:
            print(f"[error-transcribe] Failed to transcribe {wav.name}: {e}")
            failed += 1
            continue
        elapsed = time.time() - start
        total_time += elapsed

        rtf = elapsed / duration if duration > 0 else 0.0
        print(f"  [stats] audio={duration / 60:.1f} min, wall={elapsed:.1f}s, RTF={rtf:.2f}x")

        # Attach metadata to transcript before writing JSON.
        transcript.meta = _build_meta(cfg, transcript, wav, duration)

        # v1.1: Run diarization if enabled (skeleton for now)
        if diarization_config and diarization_config.enable_diarization:
            from .api import _maybe_run_diarization

            transcript = _maybe_run_diarization(
                transcript=transcript,
                wav_path=wav,
                config=diarization_config,
            )

        writers.write_json(transcript, json_path)
        writers.write_txt(transcript, txt_path)
        writers.write_srt(transcript, srt_path)

        print(f"  → JSON: {json_path}")
        print(f"  → TXT:  {txt_path}")
        print(f"  → SRT:  {srt_path}")

        processed += 1

    print("\n=== Summary ===")
    print(
        f"  transcribed={processed}, diarized_only={diarized_only}, skipped={skipped}, "
        f"failed={failed}, total={total}"
    )
    if total_audio > 0 and total_time > 0:
        overall_rtf = total_time / total_audio
        print(
            f"  audio={total_audio / 60:.1f} min, wall={total_time / 60:.1f} min, RTF={overall_rtf:.2f}x"
        )
    print("All done.")
