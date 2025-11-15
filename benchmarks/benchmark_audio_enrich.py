#!/usr/bin/env python3
"""
Audio Enrichment Performance Benchmarking Script.

This script measures the performance of the audio enrichment pipeline across
different configurations, audio durations, and hardware setups (CPU/GPU).

Usage:
    python benchmark_audio_enrich.py --all
    python benchmark_audio_enrich.py --duration 60 --config full
    python benchmark_audio_enrich.py --duration 300 --config prosody --device cpu
    python benchmark_audio_enrich.py --quick  # Quick test with small durations
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import psutil
import torch

from benchmarks.test_audio_generator import generate_test_audio_file
from transcription.audio_enrichment import enrich_segment_audio
from transcription.models import Segment

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""

    duration_seconds: float
    config_mode: str  # 'prosody', 'emotion', 'full', 'categorical'
    device: str  # 'cpu' or 'cuda'
    description: str


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""

    config: BenchmarkConfig
    execution_time_seconds: float
    peak_memory_mb: float
    avg_memory_mb: float
    segments_processed: int
    segments_per_second: float
    time_per_segment_ms: float
    prosody_time_ms: float | None
    emotion_time_ms: float | None
    success_count: int
    failure_count: int
    timestamp: str
    system_info: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary for serialization."""
        result = asdict(self)
        result["config"] = asdict(self.config)
        return result


class MemoryMonitor:
    """Monitor memory usage during execution."""

    def __init__(self):
        self.process = psutil.Process()
        self.baseline_mb = self._get_memory_mb()
        self.peak_mb = self.baseline_mb
        self.samples = []

    def _get_memory_mb(self) -> float:
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / 1024 / 1024

    def sample(self):
        """Take a memory sample."""
        current_mb = self._get_memory_mb()
        self.samples.append(current_mb)
        self.peak_mb = max(self.peak_mb, current_mb)

    def get_peak_delta_mb(self) -> float:
        """Get peak memory increase from baseline."""
        return self.peak_mb - self.baseline_mb

    def get_avg_delta_mb(self) -> float:
        """Get average memory increase from baseline."""
        if not self.samples:
            return 0.0
        return np.mean(self.samples) - self.baseline_mb


class AudioEnrichmentBenchmark:
    """Main benchmarking class for audio enrichment."""

    # Test durations in seconds
    DURATIONS = {
        "quick": [10, 30],
        "standard": [10, 60, 300, 1800],
        "full": [10, 60, 300, 1800, 3600],
    }

    # Configuration modes
    CONFIGS = {
        "prosody": {
            "enable_prosody": True,
            "enable_emotion": False,
            "enable_categorical_emotion": False,
            "description": "Prosody only (pitch, energy, rate, pauses)",
        },
        "emotion": {
            "enable_prosody": False,
            "enable_emotion": True,
            "enable_categorical_emotion": False,
            "description": "Emotion only (dimensional: valence, arousal, dominance)",
        },
        "full": {
            "enable_prosody": True,
            "enable_emotion": True,
            "enable_categorical_emotion": False,
            "description": "Prosody + dimensional emotion",
        },
        "categorical": {
            "enable_prosody": True,
            "enable_emotion": True,
            "enable_categorical_emotion": True,
            "description": "Full feature set (prosody + dimensional + categorical emotion)",
        },
    }

    def __init__(self, output_dir: Path = None):
        """Initialize benchmark suite."""
        self.output_dir = output_dir or Path(__file__).parent / "results"
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.test_audio_dir = Path(__file__).parent / "test_audio"
        self.test_audio_dir.mkdir(exist_ok=True, parents=True)

        logger.info(f"Benchmark output directory: {self.output_dir}")
        logger.info(f"Test audio directory: {self.test_audio_dir}")

    def get_system_info(self) -> dict[str, Any]:
        """Collect system information."""
        cuda_device = None
        if torch.cuda.is_available():
            try:
                cuda_device = torch.cuda.get_device_name(0)
            except Exception:
                cuda_device = "CUDA available but device name unknown"

        return {
            "cpu_count": psutil.cpu_count(logical=False),
            "cpu_count_logical": psutil.cpu_count(logical=True),
            "total_ram_gb": psutil.virtual_memory().total / 1024 / 1024 / 1024,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": cuda_device,
            "python_version": sys.version.split()[0],
            "torch_version": torch.__version__,
        }

    def prepare_test_audio(self, duration_seconds: float) -> Path:
        """
        Generate or retrieve test audio file for given duration.

        Args:
            duration_seconds: Duration of test audio to generate

        Returns:
            Path to test audio file
        """
        audio_path = self.test_audio_dir / f"test_audio_{int(duration_seconds)}s.wav"

        if not audio_path.exists():
            logger.info(f"Generating test audio: {duration_seconds}s")
            generate_test_audio_file(
                output_path=audio_path,
                duration_seconds=duration_seconds,
                sample_rate=16000,
                add_speech_characteristics=True,
            )
        else:
            logger.info(f"Using existing test audio: {audio_path}")

        return audio_path

    def create_test_segments(
        self, duration_seconds: float, segment_duration: float = 5.0
    ) -> list[Segment]:
        """
        Create test segments for the given audio duration.

        Args:
            duration_seconds: Total duration of audio
            segment_duration: Duration of each segment

        Returns:
            List of Segment objects
        """
        segments = []
        num_segments = int(duration_seconds / segment_duration)

        for i in range(num_segments):
            start = i * segment_duration
            end = min((i + 1) * segment_duration, duration_seconds)

            segments.append(
                Segment(
                    id=i,
                    start=start,
                    end=end,
                    text=f"Test segment {i + 1} with some sample text for syllable counting.",
                )
            )

        return segments

    def run_single_benchmark(
        self, duration_seconds: float, config_mode: str, device: str = "cpu"
    ) -> BenchmarkResult:
        """
        Run a single benchmark with specified configuration.

        Args:
            duration_seconds: Duration of test audio
            config_mode: Configuration mode ('prosody', 'emotion', 'full', 'categorical')
            device: Device to use ('cpu' or 'cuda')

        Returns:
            BenchmarkResult object
        """
        if config_mode not in self.CONFIGS:
            raise ValueError(f"Invalid config mode: {config_mode}")

        config = BenchmarkConfig(
            duration_seconds=duration_seconds,
            config_mode=config_mode,
            device=device,
            description=self.CONFIGS[config_mode]["description"],
        )

        logger.info(f"\n{'='*80}")
        logger.info(f"Running benchmark: {duration_seconds}s, {config_mode}, {device}")
        logger.info(f"{'='*80}")

        # Set device for torch models
        original_device = None
        if "CUDA_VISIBLE_DEVICES" in sys.modules:
            import os

            original_device = os.environ.get("CUDA_VISIBLE_DEVICES")

        if device == "cpu":
            import os

            os.environ["CUDA_VISIBLE_DEVICES"] = ""

        try:
            # Prepare test audio
            audio_path = self.prepare_test_audio(duration_seconds)

            # Create test segments (5-second segments)
            segments = self.create_test_segments(duration_seconds, segment_duration=5.0)
            logger.info(f"Created {len(segments)} test segments")

            # Get configuration parameters
            config_params = {
                "enable_prosody": self.CONFIGS[config_mode]["enable_prosody"],
                "enable_emotion": self.CONFIGS[config_mode]["enable_emotion"],
                "enable_categorical_emotion": self.CONFIGS[config_mode][
                    "enable_categorical_emotion"
                ],
            }

            # Initialize memory monitor
            memory_monitor = MemoryMonitor()

            # Run benchmark
            success_count = 0
            failure_count = 0
            prosody_times = []
            emotion_times = []

            start_time = time.time()

            for i, segment in enumerate(segments):
                memory_monitor.sample()

                try:
                    seg_start = time.time()

                    # Time prosody extraction separately
                    if config_params["enable_prosody"]:
                        prosody_start = time.time()

                    # Enrich segment
                    audio_state = enrich_segment_audio(
                        wav_path=audio_path, segment=segment, **config_params
                    )

                    if config_params["enable_prosody"]:
                        prosody_times.append((time.time() - prosody_start) * 1000)

                    # Check if emotion was extracted
                    if config_params["enable_emotion"] and audio_state.get("emotion"):
                        emotion_times.append((time.time() - seg_start) * 1000)

                    # Check for errors
                    errors = audio_state.get("extraction_status", {}).get("errors", [])
                    if errors:
                        failure_count += 1
                    else:
                        success_count += 1

                except Exception as e:
                    logger.error(f"Failed to process segment {i}: {e}")
                    failure_count += 1

                # Progress update every 10 segments
                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1}/{len(segments)} segments...")

            execution_time = time.time() - start_time

            # Calculate metrics
            segments_per_second = len(segments) / execution_time if execution_time > 0 else 0
            time_per_segment = (execution_time / len(segments)) * 1000 if segments else 0

            result = BenchmarkResult(
                config=config,
                execution_time_seconds=execution_time,
                peak_memory_mb=memory_monitor.get_peak_delta_mb(),
                avg_memory_mb=memory_monitor.get_avg_delta_mb(),
                segments_processed=len(segments),
                segments_per_second=segments_per_second,
                time_per_segment_ms=time_per_segment,
                prosody_time_ms=np.mean(prosody_times) if prosody_times else None,
                emotion_time_ms=np.mean(emotion_times) if emotion_times else None,
                success_count=success_count,
                failure_count=failure_count,
                timestamp=datetime.now().isoformat(),
                system_info=self.get_system_info(),
            )

            # Log results
            logger.info("\nResults:")
            logger.info(f"  Total time: {execution_time:.2f}s")
            logger.info(f"  Segments/sec: {segments_per_second:.2f}")
            logger.info(f"  Time/segment: {time_per_segment:.2f}ms")
            logger.info(f"  Peak memory: {result.peak_memory_mb:.2f}MB")
            logger.info(f"  Success: {success_count}/{len(segments)}")

            return result

        finally:
            # Restore original device setting
            if original_device is not None:
                import os

                os.environ["CUDA_VISIBLE_DEVICES"] = original_device

    def run_benchmark_suite(
        self,
        duration_preset: str = "standard",
        config_modes: list[str] = None,
        devices: list[str] = None,
    ) -> list[BenchmarkResult]:
        """
        Run a full benchmark suite.

        Args:
            duration_preset: Preset for durations ('quick', 'standard', 'full')
            config_modes: List of config modes to test (default: all)
            devices: List of devices to test (default: ['cpu'] or ['cpu', 'cuda'])

        Returns:
            List of BenchmarkResult objects
        """
        if duration_preset not in self.DURATIONS:
            raise ValueError(f"Invalid duration preset: {duration_preset}")

        durations = self.DURATIONS[duration_preset]

        if config_modes is None:
            config_modes = list(self.CONFIGS.keys())

        if devices is None:
            devices = ["cpu"]
            if torch.cuda.is_available():
                devices.append("cuda")

        results = []
        total_benchmarks = len(durations) * len(config_modes) * len(devices)
        current = 0

        logger.info(f"\n{'#'*80}")
        logger.info(f"Starting benchmark suite: {total_benchmarks} total benchmarks")
        logger.info(f"Durations: {durations}")
        logger.info(f"Configs: {config_modes}")
        logger.info(f"Devices: {devices}")
        logger.info(f"{'#'*80}\n")

        for duration in durations:
            for config_mode in config_modes:
                for device in devices:
                    current += 1
                    logger.info(f"\n[{current}/{total_benchmarks}] Starting benchmark...")

                    try:
                        result = self.run_single_benchmark(duration, config_mode, device)
                        results.append(result)
                    except Exception as e:
                        logger.error(f"Benchmark failed: {e}", exc_info=True)

        return results

    def save_results(self, results: list[BenchmarkResult], filename: str = None):
        """Save benchmark results to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_results_{timestamp}.json"

        output_path = self.output_dir / filename

        data = {
            "benchmark_info": {
                "timestamp": datetime.now().isoformat(),
                "total_runs": len(results),
                "system_info": self.get_system_info(),
            },
            "results": [r.to_dict() for r in results],
        }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"\nResults saved to: {output_path}")
        return output_path


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Benchmark audio enrichment pipeline performance")

    parser.add_argument("--duration", type=float, help="Single duration to test (in seconds)")

    parser.add_argument(
        "--preset",
        choices=["quick", "standard", "full"],
        default="standard",
        help="Duration preset to use (default: standard)",
    )

    parser.add_argument(
        "--config",
        choices=["prosody", "emotion", "full", "categorical"],
        help="Single config mode to test",
    )

    parser.add_argument(
        "--device", choices=["cpu", "cuda"], help="Device to use (default: both if CUDA available)"
    )

    parser.add_argument("--quick", action="store_true", help="Run quick benchmark (10s, 30s only)")

    parser.add_argument("--all", action="store_true", help="Run all configurations and durations")

    parser.add_argument("--output-dir", type=Path, help="Output directory for results")

    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create benchmark instance
    benchmark = AudioEnrichmentBenchmark(output_dir=args.output_dir)

    # Determine what to run
    if args.duration and args.config:
        # Single benchmark
        device = args.device or "cpu"
        result = benchmark.run_single_benchmark(args.duration, args.config, device)
        results = [result]
    else:
        # Benchmark suite
        preset = "quick" if args.quick else args.preset
        config_modes = [args.config] if args.config else None
        devices = [args.device] if args.device else None

        results = benchmark.run_benchmark_suite(
            duration_preset=preset, config_modes=config_modes, devices=devices
        )

    # Save results
    output_path = benchmark.save_results(results)

    # Generate reports
    from benchmarks.results_reporter import generate_reports

    generate_reports(output_path)

    logger.info("\nBenchmark complete!")


if __name__ == "__main__":
    main()
