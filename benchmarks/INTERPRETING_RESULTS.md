# Interpreting Benchmark Results

This guide helps you understand benchmark results and make informed decisions about configuration and optimization.

## Table of Contents
- [Understanding Metrics](#understanding-metrics)
- [Reading Reports](#reading-reports)
- [Performance Targets](#performance-targets)
- [Troubleshooting](#troubleshooting)
- [Optimization Guide](#optimization-guide)

## Understanding Metrics

### Execution Time
**What it is:** Total wall-clock time to process all segments
**Units:** Seconds
**Good value:** Lower is better
**Interpretation:**
- Includes model loading, audio extraction, feature computation, and overhead
- First run is always slower due to model downloads/initialization
- Subsequent runs reuse cached models

**Example:**
```
Execution Time: 2.16s for 2 segments (10s audio)
```
This means processing took 2.16 seconds total, including all overhead.

### Segments per Second (Throughput)
**What it is:** Number of segments processed per second of real time
**Units:** segments/second
**Good value:** Higher is better, >1.0 for real-time
**Interpretation:**
- **>1.0:** Real-time capable (can keep up with live audio)
- **0.5-1.0:** Near real-time (acceptable for most use cases)
- **<0.5:** Slower than real-time (better for batch processing)

**Example:**
```
Segments/sec: 2.41
```
This means you can process 2.41 five-second segments per second, so you're processing at ~2.4x real-time speed.

### Time per Segment
**What it is:** Average time to process one segment
**Units:** Milliseconds (ms)
**Good value:** Lower is better
**Interpretation:**
- Most directly comparable metric across runs
- Excludes segment count variations
- Includes per-segment overhead

**Performance Bands:**
- **<500ms:** Excellent (very fast)
- **500-1500ms:** Good (real-time capable)
- **1500-3000ms:** Acceptable (near real-time)
- **>3000ms:** Slow (needs optimization or hardware upgrade)

**Example:**
```
Time/segment: 414.7ms
```
Each segment takes ~415ms to process, which is excellent.

### Peak Memory
**What it is:** Maximum memory used above baseline during benchmark
**Units:** Megabytes (MB)
**Good value:** Lower is better, depends on use case
**Interpretation:**
- Measures RAM increase during processing
- First segment often shows highest memory (model loading)
- Important for deployment planning

**Memory Ranges:**
- **<200 MB:** Low (prosody-only)
- **200-800 MB:** Moderate (dimensional emotion)
- **800-1500 MB:** High (categorical emotion)
- **>1500 MB:** Very high (investigate)

**Example:**
```
Peak Memory: 695.2 MB
```
Processing required ~695 MB of additional RAM (for emotion model).

### Average Memory
**What it is:** Average memory used across all samples
**Units:** Megabytes (MB)
**Good value:** Should be close to peak for consistent workloads
**Interpretation:**
- If much lower than peak: initialization spike
- If similar to peak: consistent memory usage
- Negative values: memory freed during processing

### Success Rate
**What it is:** Percentage of segments processed without errors
**Units:** Percentage (%)
**Good value:** 100%
**Interpretation:**
- Should always be 100% for synthetic test audio
- <100% indicates:
  - Audio format issues
  - Model loading failures
  - Code bugs
  - Resource constraints

**Example:**
```
Success Rate: 100.0%
```
All segments processed successfully.

### Prosody Time / Emotion Time
**What it is:** Average time spent in specific feature extraction
**Units:** Milliseconds (ms)
**Good value:** Varies by configuration
**Interpretation:**
- Helps identify bottlenecks
- Prosody is typically faster (<500ms)
- Emotion is slower (500-2000ms) due to neural models
- Only reported when that feature is enabled

## Reading Reports

### CSV Report

Best for: Data analysis, spreadsheets, custom visualizations

**Structure:**
```csv
timestamp,duration_seconds,config_mode,device,execution_time_seconds,...
2025-11-15T00:59:19.338786,10,prosody,cpu,0.994971513748169,...
```

**How to use:**
1. Open in Excel/Google Sheets/pandas
2. Create pivot tables or charts
3. Compare multiple benchmark runs
4. Track performance over time

**Common Analyses:**
```python
import pandas as pd
df = pd.read_csv('benchmark_results.csv')

# Compare configurations
df.groupby('config_mode')['time_per_segment_ms'].mean()

# Check scaling
df.groupby('duration_seconds')['segments_per_second'].mean()

# Find optimal config
df.nsmallest(5, 'time_per_segment_ms')
```

### Markdown Report

Best for: Human reading, documentation, sharing results

**Sections:**

#### 1. System Information
Shows hardware specs. Use to verify:
- CPU count matches your system
- RAM is sufficient
- CUDA status is correct

#### 2. Summary Statistics
High-level overview. Look for:
- Overall success rate (should be 100%)
- Total processing time (compare across runs)
- Average throughput (higher is better)

#### 3. Detailed Results by Configuration
Tables showing performance by config. Use to:
- Compare configurations head-to-head
- Check how performance scales with duration
- Identify memory requirements per config

**What to look for:**
- Consistent time/segment across durations = good scaling
- Increasing time/segment = poor scaling or overhead
- High memory for first duration = model loading cost

#### 4. Performance Comparison
Direct comparisons. Shows:
- **Configuration Comparison (10s audio):** Which config is fastest
- **CPU vs GPU Performance:** Speedup from GPU (if available)
- **Scaling Analysis:** How well performance scales

**Scaling Quality:**
- **Linear:** Time per segment is constant (ideal)
- **Sublinear:** Slight increase with duration (acceptable)
- **Poor:** Large increase with duration (investigate)

#### 5. Recommendations
Automated suggestions based on results:
- **Optimal Configuration:** Best overall performer
- **Memory Requirements:** RAM needed for deployment
- **Configuration-Specific Notes:** Guidance per config

### JSON Report

Best for: Programmatic access, automated processing, long-term storage

**Structure:**
```json
{
  "benchmark_info": {
    "timestamp": "...",
    "system_info": {...}
  },
  "results": [
    {
      "config": {...},
      "execution_time_seconds": 2.16,
      "peak_memory_mb": 1203.6,
      ...
    }
  ]
}
```

**How to use:**
```python
import json

with open('benchmark_results.json') as f:
    data = json.load(f)

# Extract specific metrics
for result in data['results']:
    config = result['config']['config_mode']
    time = result['time_per_segment_ms']
    print(f"{config}: {time:.1f}ms")
```

## Performance Targets

### For Different Use Cases

#### Real-Time Streaming
**Target:** Segments/sec > 1.0
**Time/segment:** <1000ms for 5s segments
**Recommended Configs:** `prosody`, `full`

**Acceptable Performance:**
```
Prosody: 28-500ms/segment ✓
Full: 400-900ms/segment ✓
```

#### Batch Processing
**Target:** Complete within acceptable time budget
**Time/segment:** <3000ms
**Recommended Configs:** `full`, `categorical`

**Acceptable Performance:**
```
Full: 400-1500ms/segment ✓
Categorical: 1000-3000ms/segment ✓
```

#### Research/Analysis
**Target:** Maximum feature quality
**Time/segment:** Any (quality over speed)
**Recommended Configs:** `categorical`

**Acceptable Performance:**
```
Categorical: 1000-5000ms/segment ✓
(Use GPU for faster processing)
```

### By Hardware

#### Consumer Laptop (4-8 cores, 8GB RAM)
**Expected Performance:**
- Prosody: 500-1000ms/segment
- Emotion: 1500-2500ms/segment
- Full: 2000-3000ms/segment
- Categorical: 4000-6000ms/segment (may struggle)

#### Desktop Workstation (8-16 cores, 16GB+ RAM)
**Expected Performance:**
- Prosody: 100-500ms/segment
- Emotion: 500-1000ms/segment
- Full: 800-1500ms/segment
- Categorical: 2000-3000ms/segment

#### Server/High-End (16+ cores, 32GB+ RAM, GPU)
**Expected Performance:**
- Prosody: 50-200ms/segment
- Emotion (GPU): 200-500ms/segment
- Full (GPU): 300-700ms/segment
- Categorical (GPU): 500-1500ms/segment

## Troubleshooting

### Poor Performance

#### Symptom: Very slow (>5000ms/segment)
**Possible Causes:**
1. System under load (check CPU/RAM usage)
2. Thermal throttling (check temperatures)
3. Wrong Python environment (check `which python`)
4. Missing optimizations (check compiled libraries)

**Solutions:**
```bash
# Check system resources
top
htop

# Verify environment
python -c "import torch; print(torch.__version__)"

# Try simpler config
python benchmark_audio_enrich.py --config prosody
```

#### Symptom: Increasing time/segment with duration
**Possible Causes:**
1. Memory pressure (swapping)
2. Resource contention
3. Inefficient scaling

**Solutions:**
```bash
# Monitor memory during benchmark
watch -n 1 free -h

# Try shorter durations
python benchmark_audio_enrich.py --duration 10

# Check for memory leaks
python -m memory_profiler benchmark_audio_enrich.py
```

#### Symptom: Success rate < 100%
**Possible Causes:**
1. Audio file issues
2. Model loading failures
3. Resource constraints

**Solutions:**
```bash
# Run with verbose logging
python benchmark_audio_enrich.py --verbose --duration 10

# Check test audio
file benchmarks/test_audio/*.wav

# Verify models downloaded
ls -lh ~/.cache/huggingface/
```

### High Memory Usage

#### Symptom: Out of memory errors
**Solutions:**
1. Use `prosody` config (lowest memory)
2. Process shorter segments
3. Close other applications
4. Add swap space
5. Upgrade RAM

#### Symptom: Memory leak (increasing over time)
**Solutions:**
1. Report as bug (include benchmark results)
2. Process in batches with restarts
3. Monitor with `python -m memory_profiler`

### Inconsistent Results

#### Symptom: Results vary between runs
**Possible Causes:**
1. System load variations
2. Thermal throttling
3. Background processes

**Solutions:**
```bash
# Run multiple times and average
for i in {1..3}; do
    python benchmark_audio_enrich.py --quick
done

# Close background apps
# Wait for system to cool down
# Rerun benchmark
```

## Optimization Guide

### Quick Wins

#### 1. Use Appropriate Configuration
```bash
# For speed: prosody only
python benchmark_audio_enrich.py --config prosody

# For balance: full pipeline
python benchmark_audio_enrich.py --config full

# For quality: categorical (with GPU)
python benchmark_audio_enrich.py --config categorical --device cuda
```

#### 2. Pre-download Models
```bash
# Download models before benchmarking
python -c "from transcription.emotion import get_emotion_recognizer; get_emotion_recognizer()"
```

#### 3. Use GPU (if available)
```bash
# Verify CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Run with GPU
python benchmark_audio_enrich.py --device cuda
```

### Advanced Optimizations

#### 1. Model Quantization
Convert models to INT8 for 2-4x speedup:
```python
# (Requires implementation)
from transformers import AutoModelForAudioClassification
model = AutoModelForAudioClassification.from_pretrained(...)
quantized_model = torch.quantization.quantize_dynamic(model)
```

#### 2. Batch Processing
Process multiple segments together:
```python
# (Requires implementation)
results = enrich_segments_batch(audio_path, segments, batch_size=4)
```

#### 3. Parallel Processing
Use multiprocessing for prosody:
```python
from multiprocessing import Pool
with Pool(processes=4) as pool:
    results = pool.map(enrich_segment_audio, segments)
```

### Monitoring Performance

#### During Development
```bash
# Quick check after changes
python benchmark_audio_enrich.py --duration 10 --config prosody

# Full validation
python benchmark_audio_enrich.py --quick
```

#### In Production
```python
import time
import psutil

def monitor_enrichment(audio_path, segment):
    start = time.time()
    mem_before = psutil.Process().memory_info().rss / 1024 / 1024

    result = enrich_segment_audio(audio_path, segment)

    duration = time.time() - start
    mem_after = psutil.Process().memory_info().rss / 1024 / 1024

    logger.info(f"Enrichment took {duration:.2f}s, "
                f"memory delta: {mem_after - mem_before:.1f}MB")
    return result
```

## Conclusion

Benchmark results provide detailed insights into:
- **Performance:** How fast each configuration runs
- **Scaling:** How performance changes with duration
- **Resources:** Memory and CPU requirements
- **Reliability:** Success rates and error patterns

Use this guide to:
1. Understand what metrics mean
2. Choose appropriate configurations
3. Identify performance issues
4. Optimize for your use case

**Next Steps:**
- Review [README.md](README.md) for usage instructions
- Check [BASELINE_RESULTS.md](BASELINE_RESULTS.md) for reference performance
- Run benchmarks on your system: `python benchmark_audio_enrich.py --quick`
