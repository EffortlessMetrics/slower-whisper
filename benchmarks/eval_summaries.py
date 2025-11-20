#!/usr/bin/env python3
"""Evaluate LLM-based meeting summarization using slower-whisper transcripts.

This script evaluates how well slower-whisper's structured JSON (with diarization
and audio cues) enables LLM-based meeting summarization compared to reference
summaries from benchmark datasets like AMI.

Workflow:
1. Load AMI meeting samples with reference summaries
2. Run slower-whisper transcription with diarization
3. Render transcript as LLM-ready context using render_conversation_for_llm()
4. Generate candidate summary using Claude
5. Score candidate vs reference using Claude-as-judge
6. Save results for analysis

Usage:
    # Evaluate 10 AMI test meetings
    python benchmarks/eval_summaries.py --dataset ami --split test --n 10

    # Use existing transcripts (skip re-transcription)
    python benchmarks/eval_summaries.py --dataset ami --use-cache

    # Analyze bad cases
    python benchmarks/eval_summaries.py --dataset ami --analyze-failures

Requirements:
    - ANTHROPIC_API_KEY environment variable
    - AMI corpus set up (see docs/AMI_SETUP.md)
    - uv sync --extra full (for diarization)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from transcription import load_transcript, transcribe_file
from transcription.benchmarks import iter_ami_meetings
from transcription.config import TranscriptionConfig
from transcription.llm_utils import render_conversation_for_llm


def check_anthropic_key() -> str:
    """Check for ANTHROPIC_API_KEY and return it."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable not set", file=sys.stderr)
        print("Get your API key from: https://console.anthropic.com/settings/keys")
        print("Then set it: export ANTHROPIC_API_KEY=sk-ant-...")
        sys.exit(1)
    return api_key


def generate_summary(context: str, api_key: str, model: str = "claude-3-5-sonnet-20241022") -> str:
    """Generate meeting summary from transcript context using Claude.

    Args:
        context: Rendered conversation text from render_conversation_for_llm()
        api_key: Anthropic API key
        model: Claude model to use

    Returns:
        Generated summary text
    """
    try:
        from anthropic import Anthropic
    except ImportError:
        print("Error: anthropic package not installed", file=sys.stderr)
        print("Install with: uv sync --extra dev")
        sys.exit(1)

    client = Anthropic(api_key=api_key)

    prompt = f"""You are analyzing a meeting transcript. Generate a concise summary that captures:
- Main topics discussed
- Key decisions made
- Action items (if any)
- Important outcomes

Keep the summary focused and factual. Use 3-5 bullet points.

Transcript:
{context}

Summary:"""

    response = client.messages.create(
        model=model,
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )

    return response.content[0].text


def judge_summary(
    reference: str,
    candidate: str,
    api_key: str,
    model: str = "claude-3-5-sonnet-20241022",
) -> dict[str, Any]:
    """Score candidate summary against reference using Claude-as-judge.

    Args:
        reference: Reference (ground truth) summary
        candidate: Generated summary to evaluate
        api_key: Anthropic API key
        model: Claude model to use

    Returns:
        Dict with scores and reasoning:
        {
            "faithfulness": int (0-10),
            "coverage": int (0-10),
            "clarity": int (0-10),
            "comments": str
        }
    """
    try:
        from anthropic import Anthropic
    except ImportError:
        print("Error: anthropic package not installed", file=sys.stderr)
        sys.exit(1)

    client = Anthropic(api_key=api_key)

    prompt = f"""You are evaluating a meeting summary against a reference summary.

Reference summary (ground truth):
---
{reference}
---

Candidate summary (to evaluate):
---
{candidate}
---

Score the candidate on three dimensions (0-10 scale):

1. **Faithfulness** (0-10): Does the candidate avoid hallucinations? Does it only include information that's actually in the meeting?
   - 10 = completely faithful, no false information
   - 5 = some minor inaccuracies
   - 0 = major hallucinations or fabrications

2. **Coverage** (0-10): Does the candidate capture the main points from the reference?
   - 10 = covers all important points
   - 5 = misses about half the key information
   - 0 = misses most/all key information

3. **Clarity** (0-10): Is the candidate well-structured and easy to understand?
   - 10 = excellent organization and readability
   - 5 = acceptable but could be clearer
   - 0 = confusing or poorly structured

Respond in JSON format:
{{
  "faithfulness": <score 0-10>,
  "coverage": <score 0-10>,
  "clarity": <score 0-10>,
  "comments": "<1-2 sentence analysis of main strengths/weaknesses>"
}}"""

    response = client.messages.create(
        model=model,
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )

    try:
        result = json.loads(response.content[0].text)
        # Ensure all required fields present
        result.setdefault("faithfulness", 0)
        result.setdefault("coverage", 0)
        result.setdefault("clarity", 0)
        result.setdefault("comments", "")
        return result
    except json.JSONDecodeError:
        print(f"Warning: Failed to parse judge response as JSON: {response.content[0].text}")
        return {
            "faithfulness": 0,
            "coverage": 0,
            "clarity": 0,
            "comments": "Failed to parse response",
        }


def analyze_failures(
    results: list[dict[str, Any]],
    api_key: str,
    threshold: int = 6,
) -> dict[str, Any]:
    """Use Claude to categorize failure patterns in bad cases.

    Args:
        results: List of evaluation results from eval_ami_summaries()
        api_key: Anthropic API key
        threshold: Score threshold below which cases are considered "bad"

    Returns:
        Dict with categorized failure modes:
        {
            "bad_cases": list of meeting IDs,
            "categories": {
                "missing_info": count,
                "hallucination": count,
                "wrong_speaker": count,
                "style_only": count
            },
            "recommendations": [list of improvement suggestions]
        }
    """
    try:
        from anthropic import Anthropic
    except ImportError:
        return {}

    # Filter bad cases
    bad_cases = [
        r
        for r in results
        if r["score"]["faithfulness"] < threshold
        or r["score"]["coverage"] < threshold
        or r["score"]["clarity"] < threshold
    ]

    if not bad_cases:
        return {
            "bad_cases": [],
            "categories": {},
            "recommendations": ["All summaries scored above threshold - no failures to analyze!"],
        }

    # Prepare summary of failures for Claude
    failure_summary = []
    for i, case in enumerate(bad_cases[:10], 1):  # Limit to 10 cases for token budget
        failure_summary.append(
            f"Case {i}: {case['meeting_id']}\n"
            f"  Scores: faithfulness={case['score']['faithfulness']}, "
            f"coverage={case['score']['coverage']}, clarity={case['score']['clarity']}\n"
            f"  Judge comments: {case['score']['comments']}\n"
        )

    client = Anthropic(api_key=api_key)

    prompt = f"""Analyze these meeting summary evaluation failures and categorize the main issues:

{chr(10).join(failure_summary)}

Based on these failures, categorize the problems into:
- **missing_info**: Important information from the meeting was omitted
- **hallucination**: Summary includes information not supported by the transcript
- **wrong_speaker**: Actions/decisions misattributed to wrong speakers
- **style_only**: Only stylistic issues, but content is mostly correct

Also provide 2-3 actionable recommendations for improving the summarization prompt or pipeline.

Respond in JSON:
{{
  "categories": {{
    "missing_info": <count>,
    "hallucination": <count>,
    "wrong_speaker": <count>,
    "style_only": <count>
  }},
  "recommendations": [
    "recommendation 1",
    "recommendation 2",
    "recommendation 3"
  ]
}}"""

    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )

    try:
        analysis = json.loads(response.content[0].text)
        analysis["bad_cases"] = [c["meeting_id"] for c in bad_cases]
        return analysis
    except json.JSONDecodeError:
        return {
            "bad_cases": [c["meeting_id"] for c in bad_cases],
            "categories": {},
            "recommendations": ["Failed to parse analysis response"],
        }


def eval_ami_summaries(
    split: str = "test",
    n: int = 10,
    use_cache: bool = False,
    include_audio_cues: bool = True,
    model: str = "claude-3-5-sonnet-20241022",
) -> list[dict[str, Any]]:
    """Evaluate AMI meeting summarization with Claude.

    Args:
        split: AMI dataset split ("train", "dev", "test")
        n: Number of meetings to evaluate
        use_cache: Skip transcription, use existing JSON files
        include_audio_cues: Include prosody/emotion cues in LLM context
        model: Claude model to use

    Returns:
        List of evaluation results:
        [
            {
                "meeting_id": "ES2002a",
                "reference_summary": "...",
                "candidate_summary": "...",
                "score": {"faithfulness": 8, "coverage": 7, "clarity": 9, "comments": "..."},
                "transcript_path": "/path/to/json"
            },
            ...
        ]
    """
    api_key = check_anthropic_key()
    results = []

    print(f"Evaluating AMI {split} split ({n} meetings)")
    print(f"Model: {model}")
    print(f"Audio cues: {include_audio_cues}")
    print()

    for i, sample in enumerate(iter_ami_meetings(split=split, require_summary=True), 1):
        if i > n:
            break

        print(f"[{i}/{n}] Processing {sample.id}...")

        # Step 1: Transcribe (or load from cache)
        json_path = Path("whisper_json") / f"{sample.id}.json"

        if use_cache and json_path.exists():
            print(f"  ✓ Using cached transcript: {json_path}")
        else:
            print("  → Transcribing with diarization...")
            config = TranscriptionConfig(
                enable_diarization=True,
                min_speakers=2,
                max_speakers=6,  # AMI meetings typically have 4 participants
            )
            try:
                transcribe_file(
                    audio_path=sample.audio_path,
                    config=config,
                    output_dir=Path("whisper_json"),
                )
                print("  ✓ Transcription complete")
            except Exception as e:
                print(f"  ✗ Transcription failed: {e}")
                continue

        # Step 2: Load transcript and render for LLM
        try:
            transcript = load_transcript(json_path)
        except Exception as e:
            print(f"  ✗ Failed to load transcript: {e}")
            continue

        # Map speaker IDs to roles if available in metadata
        speaker_labels = None
        if sample.metadata and "roles" in sample.metadata:
            speaker_labels = sample.metadata["roles"]

        context = render_conversation_for_llm(
            transcript,
            mode="turns",
            include_audio_cues=include_audio_cues,
            include_timestamps=True,
            speaker_labels=speaker_labels,
        )

        # Step 3: Generate summary
        print("  → Generating summary...")
        try:
            candidate_summary = generate_summary(context, api_key, model)
            print(f"  ✓ Summary generated ({len(candidate_summary)} chars)")
        except Exception as e:
            print(f"  ✗ Summary generation failed: {e}")
            continue

        # Step 4: Judge summary
        if sample.reference_summary:
            print("  → Scoring against reference...")
            try:
                score = judge_summary(sample.reference_summary, candidate_summary, api_key, model)
                print(
                    f"  ✓ Scores: faithfulness={score['faithfulness']}, "
                    f"coverage={score['coverage']}, clarity={score['clarity']}"
                )
            except Exception as e:
                print(f"  ✗ Scoring failed: {e}")
                score = {"faithfulness": 0, "coverage": 0, "clarity": 0, "comments": str(e)}
        else:
            print("  ⚠ No reference summary available")
            score = None

        # Collect results
        results.append(
            {
                "meeting_id": sample.id,
                "reference_summary": sample.reference_summary,
                "candidate_summary": candidate_summary,
                "score": score,
                "transcript_path": str(json_path),
                "audio_cues_used": include_audio_cues,
                "model": model,
            }
        )

        print()

    return results


def main() -> int:
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Evaluate LLM-based meeting summarization with AMI corpus",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate 10 test meetings
  python benchmarks/eval_summaries.py --dataset ami --split test --n 10

  # Use cached transcripts
  python benchmarks/eval_summaries.py --dataset ami --use-cache

  # Analyze failure patterns
  python benchmarks/eval_summaries.py --dataset ami --analyze-failures

  # Compare with/without audio cues
  python benchmarks/eval_summaries.py --dataset ami --no-audio-cues
""",
    )

    parser.add_argument(
        "--dataset",
        choices=["ami"],
        default="ami",
        help="Benchmark dataset (currently only AMI supported)",
    )
    parser.add_argument(
        "--split",
        choices=["train", "dev", "test", "all"],
        default="test",
        help="Dataset split to evaluate",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=10,
        help="Number of meetings to evaluate (default: 10)",
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Use existing transcripts, skip re-transcription",
    )
    parser.add_argument(
        "--no-audio-cues",
        action="store_true",
        help="Exclude audio cues (prosody/emotion) from LLM context",
    )
    parser.add_argument(
        "--model",
        default="claude-3-5-sonnet-20241022",
        help="Claude model to use (default: claude-3-5-sonnet-20241022)",
    )
    parser.add_argument(
        "--analyze-failures",
        action="store_true",
        help="Analyze failure patterns after evaluation",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output path for results JSON (default: benchmarks/results/ami_summaries_<date>.json)",
    )

    args = parser.parse_args()

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        results_dir = Path("benchmarks/results")
        results_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = results_dir / f"ami_summaries_{timestamp}.json"

    # Run evaluation
    results = eval_ami_summaries(
        split=args.split,
        n=args.n,
        use_cache=args.use_cache,
        include_audio_cues=not args.no_audio_cues,
        model=args.model,
    )

    if not results:
        print("No results collected. Check AMI setup and ensure meetings have reference summaries.")
        return 1

    # Compute aggregate stats
    scores = [r["score"] for r in results if r["score"]]
    if scores:
        avg_faithfulness = sum(s["faithfulness"] for s in scores) / len(scores)
        avg_coverage = sum(s["coverage"] for s in scores) / len(scores)
        avg_clarity = sum(s["clarity"] for s in scores) / len(scores)

        print("=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Meetings evaluated: {len(results)}")
        print(f"Average faithfulness: {avg_faithfulness:.1f}/10")
        print(f"Average coverage: {avg_coverage:.1f}/10")
        print(f"Average clarity: {avg_clarity:.1f}/10")
        print()

    # Analyze failures if requested
    if args.analyze_failures:
        print("=" * 60)
        print("FAILURE ANALYSIS")
        print("=" * 60)
        api_key = check_anthropic_key()
        analysis = analyze_failures(results, api_key)

        if analysis.get("bad_cases"):
            print(f"Bad cases ({len(analysis['bad_cases'])}): {', '.join(analysis['bad_cases'])}")
            print()
            print("Failure categories:")
            for category, count in analysis.get("categories", {}).items():
                print(f"  {category}: {count}")
            print()
            print("Recommendations:")
            for i, rec in enumerate(analysis.get("recommendations", []), 1):
                print(f"  {i}. {rec}")
        else:
            print("No failures detected (all summaries above threshold)")
        print()

    # Save results
    output_data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "dataset": args.dataset,
            "split": args.split,
            "n_meetings": len(results),
            "model": args.model,
            "audio_cues_used": not args.no_audio_cues,
        },
        "aggregate_scores": {
            "avg_faithfulness": avg_faithfulness if scores else None,
            "avg_coverage": avg_coverage if scores else None,
            "avg_clarity": avg_clarity if scores else None,
        },
        "results": results,
    }

    if args.analyze_failures:
        output_data["failure_analysis"] = analysis

    output_path.write_text(json.dumps(output_data, indent=2))
    print(f"Results saved to: {output_path}")
    print()
    print("Next steps:")
    print(f"  1. Review results: jq . {output_path} | less")
    print("  2. Compare different configurations (with/without audio cues)")
    print("  3. Iterate on prompts based on failure analysis")

    return 0


if __name__ == "__main__":
    sys.exit(main())
