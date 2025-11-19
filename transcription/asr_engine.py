from collections.abc import Iterable
from pathlib import Path

from faster_whisper import WhisperModel

from .cache import CachePaths
from .config import AsrConfig
from .models import Segment, Transcript


class TranscriptionEngine:
    """
    Thin wrapper around faster-whisper that returns Transcript objects.
    """

    def __init__(self, cfg: AsrConfig) -> None:
        self.cfg = cfg
        print(
            f"=== Step 2: Loading Whisper model "
            f"{cfg.model_name} on {cfg.device} ({cfg.compute_type}) ==="
        )
        try:
            # Use centralized cache for Whisper model downloads
            paths = CachePaths.from_env().ensure_dirs()
            self.model = WhisperModel(
                cfg.model_name,
                device=cfg.device,
                compute_type=cfg.compute_type,
                download_root=str(paths.whisper_root),
            )
        except Exception as e:
            msg = (
                "Failed to initialize the Whisper model.\n"
                f"  model_name={cfg.model_name}, device={cfg.device}, compute_type={cfg.compute_type}\n"
                "If you requested 'cuda', ensure that:\n"
                "  - An NVIDIA GPU is present\n"
                "  - Drivers and CUDA runtime are installed\n"
                "You can also try re-running with '--device cpu' for a CPU-only run.\n"
            )
            raise RuntimeError(msg) from e

    def transcribe_file(self, audio_path: Path) -> Transcript:
        """
        Transcribe a single audio file into a Transcript.
        """
        print(f"\n[transcribe] {audio_path.name}")
        segments, info = self.model.transcribe(
            str(audio_path),
            beam_size=self.cfg.beam_size,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": self.cfg.vad_min_silence_ms},
            language=self.cfg.language,
            task=self.cfg.task,
        )

        seg_objs: list[Segment] = []
        for idx, seg in enumerate(segments):
            seg_objs.append(
                Segment(
                    id=idx,
                    start=seg.start,
                    end=seg.end,
                    text=seg.text.strip(),
                )
            )

        return Transcript(
            file_name=audio_path.name,
            language=info.language,
            segments=seg_objs,
        )

    def transcribe_many(self, audio_files: Iterable[Path]) -> Iterable[Transcript]:
        """
        Convenience generator to transcribe multiple files.
        """
        for path in audio_files:
            yield self.transcribe_file(path)
