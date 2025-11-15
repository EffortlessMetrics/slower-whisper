# Audio Enrichment Benchmark Results

**Generated:** 2025-11-15 00:59:47
**Total Benchmarks:** 8

## System Information

- **CPU:** 16 cores (32 logical)
- **RAM:** 196.6 GB
- **CUDA Available:** False
- **Python:** 3.12.3
- **PyTorch:** 2.9.1+cu128

## Summary Statistics

- **Total Processing Time:** 28.6 seconds
- **Total Segments Processed:** 32
- **Average Throughput:** 5.86 segments/second
- **Peak Memory Usage:** 1203.6 MB
- **Average Memory Usage:** 127.7 MB
- **Overall Success Rate:** 100.0%

## Detailed Results by Configuration

### Categorical - CPU

| Duration | Exec Time (s) | Segments/s | Time/Segment (ms) | Memory (MB) | Success Rate |
|----------|---------------|------------|-------------------|-------------|--------------|
| 10s      |          2.16 |       0.93 |            1080.3 |      1203.6 |       100.0% |
| 30s      |         15.74 |       0.38 |            2622.9 |         0.0 |       100.0% |

### Emotion - CPU

| Duration | Exec Time (s) | Segments/s | Time/Segment (ms) | Memory (MB) | Success Rate |
|----------|---------------|------------|-------------------|-------------|--------------|
| 10s      |          1.30 |       1.54 |             650.6 |       695.2 |       100.0% |
| 30s      |          2.03 |       2.96 |             337.8 |         0.1 |       100.0% |

### Full - CPU

| Duration | Exec Time (s) | Segments/s | Time/Segment (ms) | Memory (MB) | Success Rate |
|----------|---------------|------------|-------------------|-------------|--------------|
| 10s      |          0.83 |       2.41 |             414.7 |         7.7 |       100.0% |
| 30s      |          5.43 |       1.11 |             904.9 |         0.5 |       100.0% |

### Prosody - CPU

| Duration | Exec Time (s) | Segments/s | Time/Segment (ms) | Memory (MB) | Success Rate |
|----------|---------------|------------|-------------------|-------------|--------------|
| 10s      |          0.99 |       2.01 |             497.5 |       138.8 |       100.0% |
| 30s      |          0.17 |      35.55 |              28.1 |         0.2 |       100.0% |

## Performance Comparison

### Configuration Comparison (10s audio)

| Configuration | Device | Time/Segment (ms) | Memory (MB) |
|---------------|--------|-------------------|-------------|
| categorical   | cpu    |            1080.3 |      1203.6 |
| emotion       | cpu    |             650.6 |       695.2 |
| full          | cpu    |             414.7 |         7.7 |
| prosody       | cpu    |             497.5 |       138.8 |

### Scaling Analysis

- **categorical (cpu):** poor scaling (CV: 41.7%)
- **emotion (cpu):** poor scaling (CV: 31.7%)
- **full (cpu):** poor scaling (CV: 37.2%)
- **prosody (cpu):** poor scaling (CV: 89.3%)

## Recommendations

### Optimal Configuration

For best performance, use:
- **Configuration:** prosody
- **Device:** cpu
- **Expected Performance:** 28.1 ms/segment

### Memory Requirements

- **Minimum RAM Recommended:** 2407 MB (2.4 GB)
- **Peak Memory Observed:** 1203.6 MB

### Configuration-Specific Notes

- **Prosody-only:** ~262.8 ms/segment (lightweight, CPU-friendly)
- **With Emotion Models:** ~1001.9 ms/segment (benefits from GPU acceleration)
