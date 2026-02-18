"""
Example: Integrating emotion recognition with transcription pipeline.

This example shows how to add emotion analysis to transcribed segments.
"""

import json
import logging
from pathlib import Path

import numpy as np
import soundfile as sf

from slower_whisper.pipeline.emotion import extract_emotion_categorical, extract_emotion_dimensional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_audio_segment(audio_file: Path, start: float, end: float) -> tuple:
    """
    Load a specific time segment from an audio file.

    Args:
        audio_file: Path to audio file
        start: Start time in seconds
        end: End time in seconds

    Returns:
        Tuple of (audio_array, sample_rate)
    """
    # Read audio file
    audio, sr = sf.read(audio_file)

    # Convert stereo to mono if needed
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    # Extract segment
    start_sample = int(start * sr)
    end_sample = int(end * sr)
    segment = audio[start_sample:end_sample]

    return segment.astype(np.float32), sr


def enrich_transcript_with_emotions(json_file: Path, audio_file: Path, output_file: Path = None):
    """
    Enrich an existing transcript JSON with emotion analysis.

    Args:
        json_file: Path to transcript JSON file
        audio_file: Path to corresponding audio file
        output_file: Optional path for enriched JSON (defaults to json_file)
    """
    if output_file is None:
        output_file = json_file

    # Load transcript
    logger.info(f"Loading transcript: {json_file}")
    with open(json_file, encoding="utf-8") as f:
        transcript = json.load(f)

    # Process each segment
    total_segments = len(transcript["segments"])
    logger.info(f"Processing {total_segments} segments...")

    for i, segment in enumerate(transcript["segments"], 1):
        start = segment["start"]
        end = segment["end"]
        duration = end - start

        logger.info(f"Segment {i}/{total_segments}: {start:.1f}s - {end:.1f}s ({duration:.1f}s)")

        try:
            # Load audio segment
            audio, sr = load_audio_segment(audio_file, start, end)

            # Extract emotions (both dimensional and categorical)
            dim_emotions = extract_emotion_dimensional(audio, sr)
            cat_emotions = extract_emotion_categorical(audio, sr)

            # Add to segment
            segment["emotion"] = {
                "dimensional": dim_emotions,
                "categorical": cat_emotions["categorical"],
            }

            # Log primary emotion
            primary = cat_emotions["categorical"]["primary"]
            confidence = cat_emotions["categorical"]["confidence"]
            valence = dim_emotions["valence"]["level"]
            arousal = dim_emotions["arousal"]["level"]

            logger.info(
                f"  Emotion: {primary} ({confidence:.2f}) | Valence: {valence}, Arousal: {arousal}"
            )

        except Exception as e:
            logger.error(f"  Failed to extract emotion: {e}")
            segment["emotion"] = None

    # Update schema version to indicate enrichment
    transcript["schema_version"] = 2
    transcript["meta"]["enriched_with"] = ["emotion_dimensional", "emotion_categorical"]

    # Save enriched transcript
    logger.info(f"Saving enriched transcript: {output_file}")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(transcript, f, indent=2, ensure_ascii=False)

    logger.info("Done!")


def analyze_transcript_emotions(json_file: Path):
    """
    Analyze and summarize emotions in an enriched transcript.

    Args:
        json_file: Path to enriched transcript JSON
    """
    with open(json_file, encoding="utf-8") as f:
        transcript = json.load(f)

    # Check if enriched
    if "enriched_with" not in transcript.get("meta", {}):
        logger.warning("Transcript has not been enriched with emotions")
        return

    # Collect emotion statistics
    emotions_count = {}
    valence_scores = []
    arousal_scores = []

    for segment in transcript["segments"]:
        if segment.get("emotion"):
            # Categorical
            primary = segment["emotion"]["categorical"]["primary"]
            emotions_count[primary] = emotions_count.get(primary, 0) + 1

            # Dimensional
            valence_scores.append(segment["emotion"]["dimensional"]["valence"]["score"])
            arousal_scores.append(segment["emotion"]["dimensional"]["arousal"]["score"])

    # Print summary
    print("\n" + "=" * 60)
    print("EMOTION ANALYSIS SUMMARY")
    print("=" * 60)

    print(f"\nFile: {transcript['file']}")
    print(f"Duration: {transcript['meta']['audio_duration_sec']:.1f}s")
    print(f"Segments: {len(transcript['segments'])}")

    print("\nCategorical Emotion Distribution:")
    total = sum(emotions_count.values())
    for emotion, count in sorted(emotions_count.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total) * 100
        bar_length = int(percentage / 2)
        bar = "█" * bar_length
        print(f"  {emotion:12s} {bar:25s} {count:3d} ({percentage:5.1f}%)")

    if valence_scores:
        avg_valence = sum(valence_scores) / len(valence_scores)
        avg_arousal = sum(arousal_scores) / len(arousal_scores)

        print("\nDimensional Averages:")
        print(
            f"  Average Valence: {avg_valence:.3f} ({'positive' if avg_valence > 0.5 else 'negative'})"
        )
        print(
            f"  Average Arousal: {avg_arousal:.3f} ({'high energy' if avg_arousal > 0.5 else 'low energy'})"
        )

    print("=" * 60 + "\n")


def main():
    """Example usage."""
    import sys

    if len(sys.argv) < 3:
        print("Usage:")
        print("  Enrich: python emotion_integration.py enrich <json_file> <audio_file>")
        print("  Analyze: python emotion_integration.py analyze <enriched_json_file>")
        sys.exit(1)

    command = sys.argv[1]

    if command == "enrich":
        json_file = Path(sys.argv[2])
        audio_file = Path(sys.argv[3])
        output_file = json_file.parent / f"{json_file.stem}_enriched.json"

        enrich_transcript_with_emotions(json_file, audio_file, output_file)

    elif command == "analyze":
        json_file = Path(sys.argv[2])
        analyze_transcript_emotions(json_file)

    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
