# Diarization Data Flow - Line-by-Line Trace

This document shows the exact line-by-line execution path when diarization is enabled.

---

## Execution Trace

### Step 1: User Input
```
$ slower-whisper transcribe --enable-diarization --min-speakers 2 --max-speakers 4
```

### Step 2: CLI Argument Parsing
**File:** `/home/steven/code/Python/slower-whisper/transcription/cli.py`

**Lines 24-189:** `build_parser()`
- Define argument parser with subcommands
- Line 35-113: Define "transcribe" subcommand with all arguments

**Lines 97-101:** `--enable-diarization` flag definition
```python
p_trans.add_argument(
    "--enable-diarization",
    action=argparse.BooleanOptionalAction,
    default=None,
    help="Enable speaker diarization (v1.1 experimental, default: False).",
)
```

**Lines 103-107:** `--min-speakers` flag definition
```python
p_trans.add_argument(
    "--min-speakers",
    type=int,
    default=None,
    help="Minimum number of speakers expected (diarization hint, optional).",
)
```

**Lines 109-113:** `--max-speakers` flag definition
```python
p_trans.add_argument(
    "--max-speakers",
    type=int,
    default=None,
    help="Maximum number of speakers expected (diarization hint, optional).",
)
```

**Lines 555-604:** `main()` function entry point
```python
def main(argv: Sequence[str] | None = None) -> int:
    try:
        parser = build_parser()
        args = parser.parse_args(argv)  # ← PARSED HERE

        if args.command == "transcribe":
            # Check for experimental diarization flag (v1.1 experimental)
            if getattr(args, "enable_diarization", False):
                print(
                    "\n[INFO] Speaker diarization is EXPERIMENTAL in v1.1.\n"
                    "Requires: uv sync --extra diarization\n"
                    "Requires: HF_TOKEN environment variable (huggingface.co/settings/tokens)\n"
                    "See docs/SPEAKER_DIARIZATION.md for details.\n",
                    file=sys.stderr,
                )

            cfg = _config_from_transcribe_args(args)  # ← CONFIG BUILDING STARTS HERE
            transcripts = transcribe_directory(args.root, config=cfg)
            print(f"\n[done] Transcribed {len(transcripts)} files")
```

**Result:**
- `args.enable_diarization = True`
- `args.min_speakers = 2`
- `args.max_speakers = 4`

---

### Step 3: Config Precedence Chain
**File:** `/home/steven/code/Python/slower-whisper/transcription/cli.py`

**Lines 360-410:** `_config_from_transcribe_args(args)`

```python
def _config_from_transcribe_args(args: argparse.Namespace) -> TranscriptionConfig:
    # Step 1: Start with defaults
    config = TranscriptionConfig()  # ← Line 377
    # Result: enable_diarization=False (default), min_speakers=None, max_speakers=None

    # Step 2: Override with environment variables
    env_config = TranscriptionConfig.from_env()  # ← Line 380
    config = _merge_configs(config, env_config)  # ← Line 381
    # Result: If SLOWER_WHISPER_ENABLE_DIARIZATION env var set, override

    # Step 3: Override with config file if provided
    if args.config is not None:  # ← Line 384
        file_config = TranscriptionConfig.from_file(args.config)
        config = _merge_configs(config, file_config)
    # Result: If --config file provided, merge its values

    # Step 4: Override with explicit CLI flags (only if not None)
    config = TranscriptionConfig(  # ← Line 389
        model=args.model if args.model is not None else config.model,
        device=args.device if args.device is not None else config.device,
        compute_type=args.compute_type if args.compute_type is not None else config.compute_type,
        language=args.language if args.language is not None else config.language,
        task=args.task if args.task is not None else config.task,
        vad_min_silence_ms=args.vad_min_silence_ms
            if args.vad_min_silence_ms is not None
            else config.vad_min_silence_ms,
        beam_size=args.beam_size if args.beam_size is not None else config.beam_size,
        skip_existing_json=args.skip_existing_json
            if args.skip_existing_json is not None
            else config.skip_existing_json,
        enable_diarization=args.enable_diarization  # ← Line 402
            if args.enable_diarization is not None  # TRUE (args.enable_diarization = True)
            else config.enable_diarization,
        diarization_device=config.diarization_device,  # ← Line 405 (not overridden by CLI)
        min_speakers=args.min_speakers if args.min_speakers is not None else config.min_speakers,  # ← Line 406 (TRUE, args.min_speakers = 2)
        max_speakers=args.max_speakers if args.max_speakers is not None else config.max_speakers,  # ← Line 407 (TRUE, args.max_speakers = 4)
    )

    return config  # ← Line 410
```

**Result:** TranscriptionConfig instance with:
- `enable_diarization = True` ✅
- `diarization_device = "auto"` (default)
- `min_speakers = 2` ✅
- `max_speakers = 4` ✅
- `overlap_threshold = 0.3` (default)

---

### Step 4: Config Class Definition
**File:** `/home/steven/code/Python/slower-whisper/transcription/config.py`

**Lines 78-107:** `TranscriptionConfig` dataclass
```python
@dataclass(slots=True)
class TranscriptionConfig:
    """High-level transcription configuration for public API and CLI."""

    # Whisper / faster-whisper settings
    model: str = "large-v3"
    device: str = "cuda"
    compute_type: str = "float16"
    language: str | None = None
    task: WhisperTask = "transcribe"

    # Behavior
    skip_existing_json: bool = True

    # Advanced options
    vad_min_silence_ms: int = 500
    beam_size: int = 5

    # v1.1+ diarization (L2) — opt-in
    enable_diarization: bool = False  # ← Line 103
    diarization_device: str = "auto"  # Line 104 ("cuda" | "cpu" | "auto")
    min_speakers: int | None = None  # Line 105
    max_speakers: int | None = None  # Line 106
    overlap_threshold: float = 0.3  # Line 107 (exposed via --overlap-threshold)
```

**Result:** Config object created with diarization fields populated.

---

### Step 5: API Layer - transcribe_directory()
**File:** `/home/steven/code/Python/slower-whisper/transcription/api.py`

**Lines 151-223:** `transcribe_directory(root, config)`
```python
def transcribe_directory(
    root: str | Path,
    config: TranscriptionConfig,  # ← RECEIVES FULL CONFIG HERE
) -> list[Transcript]:
    """Transcribe all audio files under a project root."""
    from .config import AppConfig
    from .pipeline import run_pipeline

    root = Path(root)

    # Convert public config to internal AppConfig
    paths = Paths(root=root)  # Line 182
    asr_cfg = AsrConfig(  # Line 183
        model_name=config.model,
        device=config.device,
        compute_type=config.compute_type,
        vad_min_silence_ms=config.vad_min_silence_ms,
        beam_size=config.beam_size,
        language=config.language,
        task=config.task,
    )
    app_cfg = AppConfig(  # Line 192
        paths=paths,
        asr=asr_cfg,
        skip_existing_json=config.skip_existing_json,
    )

    # Run the pipeline (modifies files on disk)
    # Pass the config for diarization if enabled
    run_pipeline(app_cfg, diarization_config=config if config.enable_diarization else None)  # ← Line 200
    # RESULT: If config.enable_diarization is True:
    #         run_pipeline(app_cfg, diarization_config=config)
    #         If config.enable_diarization is False:
    #         run_pipeline(app_cfg, diarization_config=None)

    # Load and return all transcripts
    json_dir = paths.json_dir  # Line 203
    json_files = sorted(json_dir.glob("*.json"))  # Line 204

    if not json_files:
        raise TranscriptionError(...)

    transcripts = []
    for json_path in json_files:  # Line 213
        try:
            transcript = load_transcript_from_json(json_path)
            transcripts.append(transcript)
        except Exception as e:
            raise TranscriptionError(...) from e

    return transcripts  # Line 223
```

**Result:** `diarization_config=config` passed to pipeline (since config.enable_diarization is True).

---

### Step 6: Pipeline Layer - run_pipeline()
**File:** `/home/steven/code/Python/slower-whisper/transcription/pipeline.py`

**Lines 51-145:** `run_pipeline(cfg, diarization_config)`
```python
def run_pipeline(cfg: AppConfig, diarization_config: TranscriptionConfig | None = None) -> None:
    """Orchestrate the full pipeline."""

    paths = cfg.paths  # Line 69

    audio_io.ensure_dirs(paths)  # Line 71
    audio_io.normalize_all(paths)  # Line 72

    norm_files = sorted(paths.norm_dir.glob("*.wav"))  # Line 74
    if not norm_files:
        print("No .wav files found in input_audio/. Nothing to transcribe.")
        return

    engine = TranscriptionEngine(cfg.asr)  # Line 79

    print("\n=== Step 3: Transcribing normalized audio ===")
    total = len(norm_files)
    processed = skipped = failed = 0
    total_audio = 0.0
    total_time = 0.0

    for idx, wav in enumerate(norm_files, start=1):  # Line 87
        print(f"[{idx}/{total}] {wav.name}")
        stem = Path(wav.name).stem  # Line 89
        json_path = paths.json_dir / f"{stem}.json"  # Line 90
        txt_path = paths.transcripts_dir / f"{stem}.txt"  # Line 91
        srt_path = paths.transcripts_dir / f"{stem}.srt"  # Line 92

        if cfg.skip_existing_json and json_path.exists():  # Line 94
            print(f"[skip-transcribe] {wav.name} because {json_path.name} already exists")
            skipped += 1
            continue

        duration = _get_duration_seconds(wav)  # Line 99
        total_audio += duration  # Line 100

        start = time.time()  # Line 102
        try:
            transcript = engine.transcribe_file(wav)  # Line 104 ← ASR PRODUCES TRANSCRIPT
        except Exception as e:
            print(f"[error-transcribe] Failed to transcribe {wav.name}: {e}")
            failed += 1
            continue
        elapsed = time.time() - start  # Line 109
        total_time += elapsed  # Line 110

        rtf = elapsed / duration if duration > 0 else 0.0
        print(f"  [stats] audio={duration / 60:.1f} min, wall={elapsed:.1f}s, RTF={rtf:.2f}x")

        # Attach metadata to transcript before writing JSON.
        transcript.meta = _build_meta(cfg, transcript, wav, duration)  # Line 116

        # v1.1: Run diarization if enabled (skeleton for now)
        if diarization_config and diarization_config.enable_diarization:  # ← Line 119 CONDITIONAL
            # CONDITION EVALUATES TO TRUE:
            # - diarization_config = config (passed from api.py:200)
            # - diarization_config.enable_diarization = True

            from .api import _maybe_run_diarization  # Line 120

            transcript = _maybe_run_diarization(  # Line 122 ← DIARIZATION CALLED HERE
                transcript=transcript,  # Transcript with segments from ASR
                wav_path=wav,  # Normalized audio file
                config=diarization_config,  # TranscriptionConfig with enable_diarization=True
            )
            # ← TRANSCRIPT RETURNED WITH DIARIZATION DATA (or unchanged if error)

        writers.write_json(transcript, json_path)  # Line 128
        writers.write_txt(transcript, txt_path)  # Line 129
        writers.write_srt(transcript, srt_path)  # Line 130

        print(f"  → JSON: {json_path}")  # Line 132
        print(f"  → TXT:  {txt_path}")  # Line 133
        print(f"  → SRT:  {srt_path}")  # Line 134

        processed += 1  # Line 136

    print("\n=== Summary ===")  # Line 138
    print(f"  processed={processed}, skipped={skipped}, failed={failed}, total={total}")
    if total_audio > 0 and total_time > 0:
        overall_rtf = total_time / total_audio
        print(
            f"  audio={total_audio / 60:.1f} min, wall={total_time / 60:.1f} min, RTF={overall_rtf:.2f}x"
        )
    print("All done.")  # Line 145
```

**Result:** Conditional diarization block executes (lines 119-126), calls `_maybe_run_diarization()`.

---

### Step 7: Diarization Orchestrator - _maybe_run_diarization()
**File:** `/home/steven/code/Python/slower-whisper/transcription/api.py`

**Lines 39-148:** `_maybe_run_diarization(transcript, wav_path, config)`
```python
def _maybe_run_diarization(
    transcript: Transcript,
    wav_path: Path,
    config: TranscriptionConfig,
) -> Transcript:
    """Run diarization and turn building if enabled."""

    if not config.enable_diarization:  # Line 66
        # NOT EXECUTED (config.enable_diarization = True)
        return transcript

    # v1.1: Real diarization implementation with pyannote.audio
    try:  # Line 70
        from .diarization import Diarizer, assign_speakers  # Line 71
        from .turns import build_turns  # Line 72

        diarizer = Diarizer(  # Line 74 ← INSTANTIATE DIARIZER
            device=config.diarization_device,  # = "auto"
            min_speakers=config.min_speakers,  # = 2
            max_speakers=config.max_speakers,  # = 4
        )

        speaker_turns = diarizer.run(wav_path)  # Line 79 ← RUN DIARIZATION
        # RETURNS: list[SpeakerTurn] with speaker_id, start, end

        if len(speaker_turns) == 0:  # Line 81
            logger.warning("Diarization produced no speaker turns for %s", wav_path.name)

        # Check for suspiciously high speaker counts
        unique_speakers = len({t.speaker_id for t in speaker_turns})  # Line 85
        if unique_speakers > 10:  # Line 86
            logger.warning(
                "Diarization found %d speakers for %s; this may indicate "
                "noisy audio or misconfiguration.",
                unique_speakers,
                wav_path.name,
            )

        # Assign speakers to segments based on overlap
        transcript = assign_speakers(  # Line 95 ← ASSIGN SPEAKERS
            transcript,
            speaker_turns,
            overlap_threshold=config.overlap_threshold,  # = 0.3
        )
        # RESULT:
        # - transcript.segments[i].speaker populated with {"id": "spk_0", "confidence": 0.95}
        # - transcript.speakers list built with aggregate stats

        # Build turn structure from speaker-labeled segments
        transcript = build_turns(transcript)  # Line 102 ← BUILD TURNS
        # RESULT: transcript.turns populated with turn groupings

        # Record success in metadata
        if transcript.meta is None:  # Line 105
            transcript.meta = {}
        diar_meta = transcript.meta.setdefault("diarization", {})  # Line 107
        diar_meta.update(  # Line 108
            {
                "status": "success",  # Line 110
                "requested": True,  # Line 111
                "backend": "pyannote.audio",  # Line 112
                "num_speakers": len(transcript.speakers) if transcript.speakers else 0,  # Line 113
            }
        )

        return transcript  # Line 117 ← RETURN UPDATED TRANSCRIPT

    except Exception as exc:  # Line 119 ← ERROR HANDLING
        logger.warning(
            "Diarization failed for %s: %s. Proceeding without speakers/turns.",
            wav_path.name,
            exc,
        )
        if transcript.meta is None:  # Line 125
            transcript.meta = {}
        diar_meta = transcript.meta.setdefault("diarization", {})  # Line 127

        # Categorize error for better debugging
        error_msg = str(exc)  # Line 130
        error_type = "unknown"  # Line 131

        if "HF_TOKEN" in error_msg or "use_auth_token" in error_msg:  # Line 133
            error_type = "auth"
        elif "pyannote.audio" in error_msg or "ImportError" in str(type(exc)):  # Line 135
            error_type = "missing_dependency"
        elif "not found" in error_msg.lower() or isinstance(exc, FileNotFoundError):  # Line 137
            error_type = "file_not_found"

        diar_meta.update(  # Line 140
            {
                "status": "failed",  # Line 142
                "requested": True,  # Line 143
                "error": error_msg,  # Line 144
                "error_type": error_type,  # Line 145
            }
        )
        return transcript  # Line 148 ← RETURN UNCHANGED (GRACEFUL DEGRADATION)
```

**Result:**
- SUCCESS: Transcript returned with speakers[], turns[], segment.speaker populated, meta.diarization.status="success"
- FAILURE: Transcript returned unchanged, meta.diarization.status="failed" with error details

---

### Step 8: Diarizer Class - Instantiation
**File:** `/home/steven/code/Python/slower-whisper/transcription/diarization.py`

**Lines 106-115:** `Diarizer.__init__()`
```python
class Diarizer:
    """Speaker diarization engine using pyannote.audio."""

    def __init__(
        self,
        device: str = "auto",  # ← Receives "auto" from config.diarization_device
        min_speakers: int | None = None,  # ← Receives 2 from config.min_speakers
        max_speakers: int | None = None,  # ← Receives 4 from config.max_speakers
    ):
        self.device = device  # Line 112 → "auto"
        self.min_speakers = min_speakers  # Line 113 → 2
        self.max_speakers = max_speakers  # Line 114 → 4
        self._pipeline = None  # Line 115 (lazy loading)
```

**Result:** Diarizer instance created with config parameters stored.

---

### Step 9: Diarizer.run() - Execute Diarization
**File:** `/home/steven/code/Python/slower-whisper/transcription/diarization.py`

**Lines 166-211:** `Diarizer.run(audio_path)`
```python
def run(self, audio_path: Path | str) -> list[SpeakerTurn]:
    """Run speaker diarization on audio file."""

    audio_path = Path(audio_path)  # Line 181
    if not audio_path.exists():  # Line 182
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Load pipeline (lazy initialization)
    pipeline = self._ensure_pipeline()  # Line 186
    # CALLS: Lines 117-164 (_ensure_pipeline)
    # - Imports pyannote.audio.Pipeline
    # - Loads "pyannote/speaker-diarization-3.1" model
    # - Moves to device (auto/cuda/cpu)
    # RESULT: pipeline object ready to use

    # Run diarization
    # pyannote returns an Annotation object with segments
    diarization_result = pipeline(  # Line 190
        str(audio_path),
        min_speakers=self.min_speakers,  # = 2
        max_speakers=self.max_speakers,  # = 4
    )
    # RESULT: Annotation with speaker segments

    # Convert pyannote output to list[SpeakerTurn]
    turns: list[SpeakerTurn] = []  # Line 197
    for segment, _, label in diarization_result.itertracks(yield_label=True):  # Line 198
        turns.append(  # Line 199
            SpeakerTurn(  # Line 200
                start=float(segment.start),  # Line 201
                end=float(segment.end),  # Line 202
                speaker_id=str(label),  # Line 203 (e.g., "SPEAKER_00")
                confidence=None,  # Line 204
            )
        )

    # Sort by start time (pyannote usually returns sorted, but be explicit)
    turns.sort(key=lambda t: t.start)  # Line 209

    return turns  # Line 211 ← RETURN SPEAKER TURNS
    # RESULT: [
    #   SpeakerTurn(start=0.0, end=2.5, speaker_id="SPEAKER_00", confidence=None),
    #   SpeakerTurn(start=2.5, end=5.0, speaker_id="SPEAKER_01", confidence=None),
    #   ...
    # ]
```

**Result:** List of SpeakerTurn objects with speaker IDs and timestamps.

---

### Step 10: assign_speakers() - Map Speakers to Segments
**File:** `/home/steven/code/Python/slower-whisper/transcription/diarization.py`

**Lines 248-355:** `assign_speakers(transcript, speaker_turns, overlap_threshold)`
```python
def assign_speakers(
    transcript: Transcript,
    speaker_turns: list[SpeakerTurn],
    overlap_threshold: float = 0.3,  # ← Receives 0.3 from config.overlap_threshold
) -> Transcript:
    """Assign speaker labels to ASR segments based on diarization output."""

    # Normalize speaker IDs: backend IDs → spk_N
    speaker_map: dict[str, int] = {}  # Line 300

    # Per-speaker aggregates for building speakers[] array
    speaker_stats: dict[str, dict[str, Any]] = {}  # Line 303

    # Assign speakers to each segment
    for segment in transcript.segments:  # Line 306
        seg_start = segment.start  # Line 307
        seg_end = segment.end  # Line 308
        seg_duration = seg_end - seg_start  # Line 309

        if seg_duration <= 0:  # Line 311
            # Zero or negative duration → skip assignment
            segment.speaker = None  # Line 313
            continue

        # Find best speaker by max overlap
        best_speaker_id: str | None = None  # Line 317
        max_overlap_duration = 0.0  # Line 318

        for turn in speaker_turns:  # Line 320
            overlap_duration = _compute_overlap(seg_start, seg_end, turn.start, turn.end)  # Line 321

            if overlap_duration > max_overlap_duration:  # Line 323
                max_overlap_duration = overlap_duration  # Line 324
                best_speaker_id = turn.speaker_id  # Line 325
            elif overlap_duration == max_overlap_duration and overlap_duration > 0:  # Line 326
                # Equal overlap: choose first alphabetically (deterministic)
                if best_speaker_id is None or turn.speaker_id < best_speaker_id:  # Line 328
                    best_speaker_id = turn.speaker_id  # Line 329

        # Compute confidence = overlap_ratio
        overlap_ratio = max_overlap_duration / seg_duration if seg_duration > 0 else 0.0  # Line 332

        # Assign if above threshold
        if overlap_ratio >= overlap_threshold and best_speaker_id is not None:  # Line 335
            # overlap_ratio >= 0.3 AND speaker found
            normalized_id = _normalize_speaker_id(best_speaker_id, speaker_map)  # Line 336
            # Line 214-227: Maps "SPEAKER_00" → "spk_0", "SPEAKER_01" → "spk_1", etc.

            segment.speaker = {"id": normalized_id, "confidence": overlap_ratio}  # Line 337 ← ASSIGN SPEAKER

            # Update speaker stats
            if normalized_id not in speaker_stats:  # Line 340
                speaker_stats[normalized_id] = {  # Line 341
                    "id": normalized_id,  # Line 342
                    "label": None,  # Line 343
                    "total_speech_time": 0.0,  # Line 344
                    "num_segments": 0,  # Line 345
                }
            speaker_stats[normalized_id]["total_speech_time"] += seg_duration  # Line 347
            speaker_stats[normalized_id]["num_segments"] += 1  # Line 348
        else:
            segment.speaker = None  # Line 350 (overlap too low or no speaker)

    # Build speakers[] array
    transcript.speakers = sorted(speaker_stats.values(), key=lambda s: s["id"])  # Line 353
    # RESULT: [
    #   {"id": "spk_0", "label": None, "total_speech_time": 10.5, "num_segments": 5},
    #   {"id": "spk_1", "label": None, "total_speech_time": 8.2, "num_segments": 3},
    # ]

    return transcript  # Line 355 ← RETURN WITH SPEAKERS POPULATED
```

**Result:**
- Each segment.speaker populated with {"id": "spk_X", "confidence": Y}
- transcript.speakers array built with aggregate stats

---

### Step 11: build_turns() - Group Segments by Speaker
**File:** `/home/steven/code/Python/slower-whisper/transcription/turns.py`

**Lines 62-138:** `build_turns(transcript, pause_threshold)`
```python
def build_turns(
    transcript: Transcript,
    pause_threshold: float | None = None,  # ← Line 64 (None by default)
) -> Transcript:
    """Build conversational turns from speaker-attributed segments."""

    # Filter segments with known speakers
    speaker_segments = [seg for seg in transcript.segments if seg.speaker is not None]  # Line 108
    # RESULT: List of segments with speaker assignments (from assign_speakers)

    if not speaker_segments:  # Line 110
        # No speakers assigned → no turns
        transcript.turns = []  # Line 112
        return transcript

    turns: list[dict[str, Any]] = []  # Line 115
    current_turn_segments: list[Any] = []  # Line 116
    current_speaker_id: str | None = None  # Line 117

    for segment in speaker_segments:  # Line 119
        speaker_id = segment.speaker["id"]  # Line 120 (e.g., "spk_0")

        if speaker_id != current_speaker_id:  # Line 122 (speaker changed)
            # Speaker change → finalize current turn and start new
            if current_turn_segments:  # Line 124
                turns.append(_finalize_turn(len(turns), current_turn_segments, current_speaker_id))  # Line 125
                # CALLS: Lines 141-165 (_finalize_turn)
                # Builds dict with:
                # - "id": "turn_0"
                # - "speaker_id": "spk_0"
                # - "start": segment[0].start
                # - "end": segment[-1].end
                # - "segment_ids": [seg.id for seg in segments]
                # - "text": " ".join(segment texts)
            current_turn_segments = [segment]  # Line 126
            current_speaker_id = speaker_id  # Line 127
        else:
            # Same speaker → add to current turn
            # TODO(v1.2): Check pause_threshold here if enabled
            current_turn_segments.append(segment)  # Line 131

    # Finalize last turn
    if current_turn_segments:  # Line 134
        turns.append(_finalize_turn(len(turns), current_turn_segments, current_speaker_id))  # Line 135

    transcript.turns = turns  # Line 137 ← ASSIGN TURNS
    # RESULT: [
    #   {
    #     "id": "turn_0",
    #     "speaker_id": "spk_0",
    #     "start": 0.0,
    #     "end": 2.5,
    #     "segment_ids": [0, 1],
    #     "text": "Hello world how are you"
    #   },
    #   {
    #     "id": "turn_1",
    #     "speaker_id": "spk_1",
    #     "start": 2.5,
    #     "end": 4.0,
    #     "segment_ids": [2],
    #     "text": "Thanks for joining"
    #   },
    # ]

    return transcript  # Line 138
```

**Result:** transcript.turns populated with turn groupings.

---

### Step 12: Metadata Recording
**File:** `/home/steven/code/Python/slower-whisper/transcription/api.py`

**Lines 105-115:** Record Success
```python
# Record success in metadata
if transcript.meta is None:  # Line 105
    transcript.meta = {}
diar_meta = transcript.meta.setdefault("diarization", {})  # Line 107
diar_meta.update(  # Line 108
    {
        "status": "success",  # Line 110 ← Status recorded
        "requested": True,  # Line 111 ← Flag recorded
        "backend": "pyannote.audio",  # Line 112 ← Implementation recorded
        "num_speakers": len(transcript.speakers) if transcript.speakers else 0,  # Line 113 ← Count recorded
    }
)

return transcript  # Line 117
```

**Result:** `transcript.meta["diarization"]` populated with success metadata.

---

### Step 13: JSON Serialization
**File:** `/home/steven/code/Python/slower-whisper/transcription/writers.py`

**Lines 7-43:** `write_json(transcript, out_path)`
```python
def write_json(transcript: Transcript, out_path: Path) -> None:
    """Write transcript to JSON with a stable schema for downstream processing."""

    data = {  # Line 15
        "schema_version": SCHEMA_VERSION,  # Line 16 (= 2)
        "file": transcript.file_name,  # Line 17
        "language": transcript.language,  # Line 18
        "meta": transcript.meta or {},  # Line 19 ← Contains diarization metadata
        "segments": [  # Line 20
            {
                "id": s.id,  # Line 22
                "start": s.start,  # Line 23
                "end": s.end,  # Line 24
                "text": s.text,  # Line 25
                "speaker": s.speaker,  # Line 26 ← v1.1 SPEAKER FIELD
                "tone": s.tone,  # Line 27
                "audio_state": s.audio_state,  # Line 28
            }
            for s in transcript.segments  # Line 30
        ],
    }

    # Add v1.1+ fields if present
    if transcript.speakers is not None:  # Line 35
        data["speakers"] = transcript.speakers  # Line 36 ← v1.1 SPEAKERS ARRAY
    if transcript.turns is not None:  # Line 37
        data["turns"] = transcript.turns  # Line 38 ← v1.1 TURNS ARRAY

    out_path.write_text(  # Line 40
        json.dumps(data, ensure_ascii=False, indent=2),  # Line 41
        encoding="utf-8",  # Line 42
    )  # Line 43
```

**Result:** JSON file written with all diarization data (speakers, turns, segment.speaker, meta.diarization).

---

### Step 14: Final Output File
**File:** `whisper_json/audio.json` (created at line 40)

```json
{
  "schema_version": 2,
  "file": "audio.wav",
  "language": "en",
  "meta": {
    "generated_at": "2025-11-15T10:30:00+00:00",
    "model_name": "large-v3",
    "device": "cuda",
    "diarization": {
      "status": "success",
      "requested": true,
      "backend": "pyannote.audio",
      "num_speakers": 2
    }
  },
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 2.47,
      "text": "Hello everyone welcome to the meeting",
      "speaker": {
        "id": "spk_0",
        "confidence": 0.95
      },
      "tone": null,
      "audio_state": null
    },
    {
      "id": 1,
      "start": 2.47,
      "end": 4.15,
      "text": "Thanks for joining us",
      "speaker": {
        "id": "spk_1",
        "confidence": 0.87
      },
      "tone": null,
      "audio_state": null
    }
  ],
  "speakers": [
    {
      "id": "spk_0",
      "label": null,
      "total_speech_time": 10.45,
      "num_segments": 4
    },
    {
      "id": "spk_1",
      "label": null,
      "total_speech_time": 8.2,
      "num_segments": 3
    }
  ],
  "turns": [
    {
      "id": "turn_0",
      "speaker_id": "spk_0",
      "start": 0.0,
      "end": 2.47,
      "segment_ids": [0],
      "text": "Hello everyone welcome to the meeting"
    },
    {
      "id": "turn_1",
      "speaker_id": "spk_1",
      "start": 2.47,
      "end": 4.15,
      "segment_ids": [1],
      "text": "Thanks for joining us"
    }
  ]
}
```

---

## Summary

The complete execution trace shows:

1. **CLI flag parsed** → args.enable_diarization = True
2. **Config built** → TranscriptionConfig with enable_diarization=True, min_speakers=2, max_speakers=4
3. **Config passed to API** → transcribe_directory(root, config)
4. **Config passed to pipeline** → run_pipeline(app_cfg, diarization_config=config)
5. **Pipeline checks flag** → if diarization_config.enable_diarization (TRUE)
6. **Diarizer created** → Diarizer(device="auto", min_speakers=2, max_speakers=4)
7. **Diarizer.run()** → Returns list[SpeakerTurn] with speaker IDs
8. **assign_speakers()** → Populates segment.speaker and transcript.speakers
9. **build_turns()** → Populates transcript.turns
10. **Metadata recorded** → meta.diarization.status="success"
11. **JSON serialized** → All diarization fields included
12. **File written** → whisper_json/audio.json with complete diarization data

All 14 steps are connected and verified. The data flow is complete.
