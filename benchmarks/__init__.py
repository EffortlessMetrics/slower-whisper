"""
Audio enrichment benchmarking suite.

This package provides comprehensive performance benchmarking for the audio
enrichment pipeline, including:
- Performance measurement across different configurations
- CPU vs GPU comparison
- Memory profiling
- Scalability testing
- Automated report generation
"""

__version__ = "1.0.0"

from .benchmark_audio_enrich import AudioEnrichmentBenchmark, BenchmarkConfig, BenchmarkResult
from .results_reporter import BenchmarkReporter, generate_reports
from .test_audio_generator import generate_benchmark_test_suite, generate_test_audio_file

__all__ = [
    "AudioEnrichmentBenchmark",
    "BenchmarkConfig",
    "BenchmarkResult",
    "generate_test_audio_file",
    "generate_benchmark_test_suite",
    "BenchmarkReporter",
    "generate_reports",
]
