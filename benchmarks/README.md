# Audio Enrichment Benchmarking Suite

This directory contains a comprehensive benchmarking infrastructure for the audio enrichment pipeline. It measures performance across different configurations, audio durations, and hardware setups.

## Overview

The benchmarking suite provides:

- **Performance Metrics**: Execution time, throughput, memory usage
- **Multiple Configurations**: Test prosody-only, emotion-only, or full feature extraction
- **Hardware Comparison**: CPU vs GPU performance analysis
- **Scalability Testing**: Performance across different audio durations (10s to 30min)
- **Automated Reporting**: CSV and Markdown reports with analysis and recommendations

## Quick Start

### Run Quick Benchmark (10s, 30s audio)

```bash
cd benchmarks
python benchmark_audio_enrich.py --quick
```

### Run Standard Benchmark Suite

```bash
python benchmark_audio_enrich.py --all
```

### Run Specific Configuration

```bash
# Test prosody extraction only on 1-minute audio
python benchmark_audio_enrich.py --duration 60 --config prosody

# Test full pipeline with GPU
python benchmark_audio_enrich.py --duration 300 --config full --device cuda
```

## Components

### 1. `benchmark_audio_enrich.py`

Main benchmarking script that:
- Generates or uses test audio files
- Runs audio enrichment with different configurations
- Measures time and memory consumption
- Tracks success/failure rates
- Saves results to JSON

**Key Features:**
- Memory profiling with peak and average tracking
- Per-segment timing breakdown
- Automatic device selection (CPU/GPU)
- Graceful error handling

### 2. `test_audio_generator.py`

Synthetic audio generator that creates speech-like test files:
- Varying pitch contours
- Syllabic rhythm patterns
- Energy variations
- Realistic pauses
- Formant-like structure

**Advantages:**
- No need for real audio files
- Consistent test conditions
- Configurable characteristics
- Lightweight (no large test assets)

### 3. `results_reporter.py`

Report generation module that creates:
- **CSV Reports**: Raw data for analysis in Excel/Python
- **Markdown Reports**: Human-readable summaries with:
  - System information
  - Summary statistics
  - Configuration comparisons
  - Scaling analysis
  - Performance recommendations

## Benchmark Configurations

### Configuration Modes

| Mode | Prosody | Emotion (Dim) | Emotion (Cat) | Use Case |
|------|---------|---------------|---------------|----------|
| `prosody` | ✓ | ✗ | ✗ | Fast, CPU-friendly, basic features |
| `emotion` | ✗ | ✓ | ✗ | Emotion-only analysis |
| `full` | ✓ | ✓ | ✗ | Complete analysis without categorical |
| `categorical` | ✓ | ✓ | ✓ | Full feature set (slowest, most complete) |

### Duration Presets

| Preset | Durations | Total Time | Use Case |
|--------|-----------|------------|----------|
| `quick` | 10s, 30s | ~1-2 min | Quick sanity check |
| `standard` | 10s, 1m, 5m, 30m | ~10-20 min | Regular benchmarking |
| `full` | 10s, 1m, 5m, 30m, 1h | ~30-60 min | Comprehensive analysis |

## Usage Examples

### Basic Usage

```bash
# Quick test of all configurations
python benchmark_audio_enrich.py --quick

# Standard benchmark with all configs
python benchmark_audio_enrich.py --preset standard

# Full benchmark suite
python benchmark_audio_enrich.py --preset full
```

### Targeted Testing

```bash
# Test specific duration and config
python benchmark_audio_enrich.py --duration 300 --config prosody --device cpu

# Test GPU performance
python benchmark_audio_enrich.py --duration 60 --config full --device cuda

# Compare CPU vs GPU
python benchmark_audio_enrich.py --duration 120 --config emotion
# (automatically tests both CPU and GPU if available)
```

### Custom Output

```bash
# Specify output directory
python benchmark_audio_enrich.py --quick --output-dir ./my_results

# Enable verbose logging
python benchmark_audio_enrich.py --quick --verbose
```

## Understanding Results

### JSON Results File

Raw results are saved in `results/benchmark_results_YYYYMMDD_HHMMSS.json`:

```json
{
  "benchmark_info": {
    "timestamp": "2025-11-15T12:00:00",
    "total_runs": 12,
    "system_info": {
      "cpu_count": 8,
      "total_ram_gb": 32,
      "cuda_available": true,
      "cuda_device": "NVIDIA RTX 3080"
    }
  },
  "results": [
    {
      "config": {
        "duration_seconds": 10,
        "config_mode": "prosody",
        "device": "cpu"
      },
      "execution_time_seconds": 2.45,
      "peak_memory_mb": 145.3,
      "segments_per_second": 0.82,
      "time_per_segment_ms": 1225.0,
      "success_count": 2,
      "failure_count": 0
    }
  ]
}
```

### CSV Report

Contains all metrics in tabular format for easy analysis:
- `duration_seconds`: Audio duration
- `config_mode`: Configuration used
- `device`: CPU or GPU
- `execution_time_seconds`: Total processing time
- `peak_memory_mb`: Maximum memory usage
- `segments_per_second`: Throughput
- `time_per_segment_ms`: Average time per segment
- `success_rate`: Percentage of successful extractions

### Markdown Report

Comprehensive analysis including:

1. **System Information**: Hardware specs
2. **Summary Statistics**: Overall performance metrics
3. **Detailed Results**: Tables by configuration
4. **Performance Comparison**: Config and device comparisons
5. **Recommendations**: Optimal settings for your system

## Key Metrics Explained

### Execution Time
Total wall-clock time to process all segments. Lower is better.

### Segments per Second (Throughput)
Number of 5-second segments processed per second. Higher is better.
- **>1.0**: Real-time capable (can process faster than playback)
- **0.5-1.0**: Near real-time
- **<0.5**: Slower than real-time

### Time per Segment
Average processing time per segment in milliseconds. Lower is better.
- **<500ms**: Excellent (very fast)
- **500-1500ms**: Good (real-time capable)
- **1500-3000ms**: Acceptable (near real-time)
- **>3000ms**: Slow (optimization needed)

### Peak Memory
Maximum RAM used above baseline. Important for deployment planning.

### Success Rate
Percentage of segments successfully processed. Should be 100% for synthetic audio.

## Performance Tips

### CPU Optimization
1. Use `prosody` config for fastest CPU performance
2. Reduce segment count for long audio (increase segment duration)
3. Consider batch processing with multiprocessing

### GPU Optimization
1. Use `emotion` or `full` config to leverage GPU
2. Ensure CUDA is properly configured
3. GPU benefits increase with longer audio files
4. Categorical emotion models benefit most from GPU

### Memory Optimization
1. Process audio in smaller chunks
2. Use `prosody` config to minimize memory
3. Clear model caches between runs if memory-constrained

## Interpreting Results

### Good Performance Indicators
- Segments/sec > 1.0 (real-time capable)
- Success rate = 100%
- Linear scaling (time per segment constant across durations)
- Peak memory < 500MB for most configs

### Performance Issues
- Segments/sec < 0.5 (slower than real-time)
- Increasing time per segment with duration (poor scaling)
- High failure rates (>5%)
- Excessive memory usage (>1GB)

### CPU vs GPU Analysis
- Prosody-only: CPU and GPU should be similar
- Emotion models: GPU should be 2-5x faster
- If GPU is slower: Check CUDA setup or model initialization overhead

## Troubleshooting

### "CUDA out of memory"
- Use `--device cpu` to force CPU
- Reduce batch size in emotion models
- Close other GPU applications

### "Audio generation failed"
- Check disk space in `test_audio/` directory
- Verify write permissions
- Try smaller durations first

### "Import errors"
- Ensure all dependencies installed: `pip install -r requirements.txt`
- Check that you're in the correct directory
- Verify virtual environment activation

### Slow performance
- First run is always slower (model downloads)
- Check system resources (CPU/RAM usage)
- Use `--verbose` to identify bottlenecks
- Try `--quick` for faster testing

## Advanced Usage

### Generate Test Audio Manually

```bash
# Generate 5-minute test audio
python test_audio_generator.py --duration 300 --output test_5min.wav

# Generate simple sine wave for testing
python test_audio_generator.py --duration 60 --output sine.wav --simple
```

### Generate Reports from Existing Results

```bash
# Generate all reports
python results_reporter.py results/benchmark_results_20251115_120000.json

# CSV only
python results_reporter.py results/benchmark_results_20251115_120000.json --csv-only

# Markdown only
python results_reporter.py results/benchmark_results_20251115_120000.json --markdown-only
```

### Batch Processing

```bash
# Run multiple benchmarks with different configs
for config in prosody emotion full; do
    python benchmark_audio_enrich.py --duration 60 --config $config
done
```

## Baseline Performance

Typical performance on modern hardware (2023):

### Desktop (i7-10700K, 32GB RAM, RTX 3080)

| Config | Duration | Device | Time/Seg (ms) | Seg/s | Memory (MB) |
|--------|----------|--------|---------------|-------|-------------|
| prosody | 60s | CPU | 800 | 1.25 | 120 |
| emotion | 60s | GPU | 1200 | 0.83 | 450 |
| full | 60s | GPU | 1500 | 0.67 | 480 |
| categorical | 60s | GPU | 2000 | 0.50 | 650 |

### Laptop (i5-1135G7, 16GB RAM, no GPU)

| Config | Duration | Device | Time/Seg (ms) | Seg/s | Memory (MB) |
|--------|----------|--------|---------------|-------|-------------|
| prosody | 60s | CPU | 1200 | 0.83 | 110 |
| emotion | 60s | CPU | 3500 | 0.29 | 420 |
| full | 60s | CPU | 4000 | 0.25 | 450 |

*Note: These are approximate values for reference. Your results will vary.*

## Contributing

To add new benchmark configurations:

1. Add configuration to `CONFIGS` dict in `benchmark_audio_enrich.py`
2. Update this README with the new config details
3. Run benchmarks and update baseline performance table

## Files Generated

```
benchmarks/
├── test_audio/              # Generated test audio files
│   ├── test_audio_10s.wav
│   ├── test_audio_60s.wav
│   └── ...
├── results/                 # Benchmark results
│   ├── benchmark_results_YYYYMMDD_HHMMSS.json
│   ├── benchmark_results_YYYYMMDD_HHMMSS.csv
│   └── benchmark_results_YYYYMMDD_HHMMSS.md
└── ...
```

## License

Same as parent project (see root LICENSE file).

## Support

For issues or questions:
1. Check this README thoroughly
2. Review generated Markdown reports for recommendations
3. Open an issue with:
   - System specs
   - Command used
   - Error messages
   - Relevant log output
