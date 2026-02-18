#!/bin/bash
set -e

BENCH_ROOT=$(uv run python -c "from slower_whisper.pipeline.benchmarks import get_benchmarks_root; print(get_benchmarks_root())")
AMI_ROOT="$BENCH_ROOT/ami"

echo "Checking AMI setup at: $AMI_ROOT"
echo

# Check directories
echo "✓ Checking directories..."
test -d "$AMI_ROOT/audio" || { echo "✗ Missing audio/ directory"; exit 1; }
test -d "$AMI_ROOT/annotations" || { echo "✗ Missing annotations/ directory"; exit 1; }

# Count files
N_AUDIO=$(ls "$AMI_ROOT/audio/"*.wav 2>/dev/null | wc -l || echo 0)
N_ANNOT=$(ls "$AMI_ROOT/annotations/"*.json 2>/dev/null | wc -l || echo 0)

echo "  Audio files:      $N_AUDIO"
echo "  Annotation files: $N_ANNOT"

if [ "$N_AUDIO" -eq 0 ]; then
    echo "✗ No audio files found!"
    echo "  Expected: $AMI_ROOT/audio/*.wav"
    exit 1
fi

if [ "$N_ANNOT" -eq 0 ]; then
    echo "✗ No annotation files found!"
    echo "  Expected: $AMI_ROOT/annotations/*.json"
    exit 1
fi

# Test iteration
echo
echo "✓ Testing dataset iteration..."
uv run python -c "
from slower_whisper.pipeline.benchmarks import iter_ami_meetings
samples = list(iter_ami_meetings(limit=5))
if not samples:
    print('✗ No samples found!')
    exit(1)
print(f'  Found {len(samples)} samples:')
for s in samples:
    has_summary = '✓' if s.reference_summary else '✗'
    print(f'    {s.id}: summary={has_summary}')
"

echo
echo "✓ AMI setup looks good!"
echo "  Ready to run: uv run python benchmarks/eval_summaries.py --dataset ami --n $N_AUDIO"
