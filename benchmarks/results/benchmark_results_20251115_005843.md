# Audio Enrichment Benchmark Results

**Generated:** 2025-11-15 00:58:43
**Total Benchmarks:** 1

## System Information

- **CPU:** 16 cores (32 logical)
- **RAM:** 196.6 GB
- **CUDA Available:** False
- **Python:** 3.12.3
- **PyTorch:** 2.9.1+cu128

## Summary Statistics

- **Total Processing Time:** 1.0 seconds
- **Total Segments Processed:** 2
- **Average Throughput:** 2.04 segments/second
- **Peak Memory Usage:** 138.8 MB
- **Average Memory Usage:** 69.4 MB
- **Overall Success Rate:** 100.0%

## Detailed Results by Configuration

### Prosody - CPU

| Duration | Exec Time (s) | Segments/s | Time/Segment (ms) | Memory (MB) | Success Rate |
|----------|---------------|------------|-------------------|-------------|--------------|
| 10s      |          0.98 |       2.04 |             490.7 |       138.8 |       100.0% |

## Performance Comparison

### Scaling Analysis


## Recommendations

### Optimal Configuration

For best performance, use:
- **Configuration:** prosody
- **Device:** cpu
- **Expected Performance:** 490.7 ms/segment

### Memory Requirements

- **Minimum RAM Recommended:** 278 MB (0.3 GB)
- **Peak Memory Observed:** 138.8 MB

### Configuration-Specific Notes

- **Prosody-only:** ~490.7 ms/segment (lightweight, CPU-friendly)
