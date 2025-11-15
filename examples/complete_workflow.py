#!/usr/bin/env python3
"""
Complete Workflow Example: Transcribe, Enrich, Query, and Report.

This example demonstrates the full pipeline:
1. Transcribe audio files (calls transcribe_pipeline)
2. Enrich with audio features (calls audio_enrich)
3. Query/analyze enriched transcripts
4. Generate comprehensive reports

The workflow shows:
- How to analyze emotion distribution across segments
- How to find segments by audio characteristics (prosody, energy, speech rate)
- How to create text-only transcripts with audio annotations
- How to export enriched data to CSV for further analysis
"""

import json
import csv
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
from collections import Counter
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from transcription.models import Transcript, Segment
from transcription.writers import load_transcript_from_json
from transcription.audio_utils import AudioSegmentExtractor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TranscriptAnalyzer:
    """
    Analyze and query enriched transcripts with multiple analysis capabilities.
    """

    def __init__(self, json_path: Path, audio_path: Optional[Path] = None):
        """
        Initialize the analyzer with a transcript and optional audio file.

        Args:
            json_path: Path to the enriched JSON transcript
            audio_path: Path to the corresponding audio file (optional)
        """
        self.json_path = Path(json_path)
        self.audio_path = Path(audio_path) if audio_path else None
        self.transcript = load_transcript_from_json(self.json_path)
        self.audio_extractor = None

        if self.audio_path and self.audio_path.exists():
            try:
                self.audio_extractor = AudioSegmentExtractor(self.audio_path)
                logger.info(f"Audio file loaded: {self.audio_path.name}")
            except Exception as e:
                logger.warning(f"Could not load audio file: {e}")

    def analyze_audio_characteristics(self) -> Dict[str, Any]:
        """
        Analyze audio characteristics across all segments.

        Returns:
            Dictionary with statistics about prosody, energy, speech rate, etc.
        """
        stats = {
            'total_segments': len(self.transcript.segments),
            'total_duration': 0.0,
            'segments_with_audio_state': 0,
            'pitch': {'min': float('inf'), 'max': 0, 'mean': 0, 'levels': []},
            'energy': {'min': float('inf'), 'max': 0, 'mean': 0, 'levels': []},
            'rate': {'min': float('inf'), 'max': 0, 'mean': 0, 'levels': []},
            'pauses': {'total': 0, 'mean_count': 0, 'densities': []},
            'contour_distribution': Counter(),
        }

        pitch_values = []
        energy_values = []
        rate_values = []
        pause_counts = []

        for segment in self.transcript.segments:
            duration = segment.end - segment.start
            stats['total_duration'] += duration

            if segment.audio_state:
                stats['segments_with_audio_state'] += 1

                # Pitch analysis
                if 'pitch' in segment.audio_state:
                    pitch_info = segment.audio_state['pitch']
                    stats['pitch']['levels'].append(pitch_info.get('level', 'unknown'))
                    if 'mean_hz' in pitch_info:
                        pitch_values.append(pitch_info['mean_hz'])
                        stats['pitch']['min'] = min(
                            stats['pitch']['min'], pitch_info['mean_hz']
                        )
                        stats['pitch']['max'] = max(
                            stats['pitch']['max'], pitch_info['mean_hz']
                        )
                    if 'contour' in pitch_info:
                        stats['contour_distribution'][pitch_info['contour']] += 1

                # Energy analysis
                if 'energy' in segment.audio_state:
                    energy_info = segment.audio_state['energy']
                    stats['energy']['levels'].append(energy_info.get('level', 'unknown'))
                    if 'db_rms' in energy_info:
                        energy_values.append(energy_info['db_rms'])
                        stats['energy']['min'] = min(
                            stats['energy']['min'], energy_info['db_rms']
                        )
                        stats['energy']['max'] = max(
                            stats['energy']['max'], energy_info['db_rms']
                        )

                # Speech rate analysis
                if 'rate' in segment.audio_state:
                    rate_info = segment.audio_state['rate']
                    stats['rate']['levels'].append(rate_info.get('level', 'unknown'))
                    if 'syllables_per_sec' in rate_info:
                        rate_values.append(rate_info['syllables_per_sec'])
                        stats['rate']['min'] = min(
                            stats['rate']['min'], rate_info['syllables_per_sec']
                        )
                        stats['rate']['max'] = max(
                            stats['rate']['max'], rate_info['syllables_per_sec']
                        )

                # Pause analysis
                if 'pauses' in segment.audio_state:
                    pause_info = segment.audio_state['pauses']
                    if 'count' in pause_info:
                        pause_counts.append(pause_info['count'])
                        stats['pauses']['total'] += pause_info['count']
                    if 'density' in pause_info:
                        stats['pauses']['densities'].append(pause_info['density'])

        # Calculate means
        if pitch_values:
            stats['pitch']['mean'] = sum(pitch_values) / len(pitch_values)
        if energy_values:
            stats['energy']['mean'] = sum(energy_values) / len(energy_values)
        if rate_values:
            stats['rate']['mean'] = sum(rate_values) / len(rate_values)
        if pause_counts:
            stats['pauses']['mean_count'] = sum(pause_counts) / len(pause_counts)

        # Count levels
        stats['pitch']['level_counts'] = dict(Counter(stats['pitch']['levels']))
        stats['energy']['level_counts'] = dict(Counter(stats['energy']['levels']))
        stats['rate']['level_counts'] = dict(Counter(stats['rate']['levels']))

        return stats

    def find_excited_moments(self, threshold_energy: str = 'high',
                             threshold_pitch: str = 'high',
                             threshold_rate: str = 'fast') -> List[Segment]:
        """
        Find segments with excited speech characteristics.

        Excited moments typically have:
        - High pitch
        - High energy
        - Fast speech rate

        Args:
            threshold_energy: 'low', 'medium', or 'high'
            threshold_pitch: 'low', 'medium', or 'high'
            threshold_rate: 'slow', 'normal', or 'fast'

        Returns:
            List of matching segments
        """
        excited_segments = []

        for segment in self.transcript.segments:
            if not segment.audio_state:
                continue

            matches = True

            # Check pitch
            if 'pitch' in segment.audio_state:
                pitch_level = segment.audio_state['pitch'].get('level', '')
                if threshold_pitch not in pitch_level.lower():
                    matches = False

            # Check energy
            if 'energy' in segment.audio_state:
                energy_level = segment.audio_state['energy'].get('level', '')
                if threshold_energy not in energy_level.lower():
                    matches = False

            # Check rate
            if 'rate' in segment.audio_state:
                rate_level = segment.audio_state['rate'].get('level', '')
                if threshold_rate not in rate_level.lower():
                    matches = False

            if matches:
                excited_segments.append(segment)

        return excited_segments

    def find_calm_moments(self, threshold_energy: str = 'low',
                         threshold_pitch: str = 'low',
                         threshold_rate: str = 'slow') -> List[Segment]:
        """
        Find segments with calm speech characteristics.

        Calm moments typically have:
        - Low pitch
        - Low energy
        - Slow speech rate
        - Potential pauses

        Args:
            threshold_energy: 'low', 'medium', or 'high'
            threshold_pitch: 'low', 'medium', or 'high'
            threshold_rate: 'slow', 'normal', or 'fast'

        Returns:
            List of matching segments
        """
        calm_segments = []

        for segment in self.transcript.segments:
            if not segment.audio_state:
                continue

            matches = True

            # Check pitch
            if 'pitch' in segment.audio_state:
                pitch_level = segment.audio_state['pitch'].get('level', '')
                if threshold_pitch not in pitch_level.lower():
                    matches = False

            # Check energy
            if 'energy' in segment.audio_state:
                energy_level = segment.audio_state['energy'].get('level', '')
                if threshold_energy not in energy_level.lower():
                    matches = False

            # Check rate
            if 'rate' in segment.audio_state:
                rate_level = segment.audio_state['rate'].get('level', '')
                if threshold_rate not in rate_level.lower():
                    matches = False

            if matches:
                calm_segments.append(segment)

        return calm_segments

    def find_emotional_segments(self, min_valence_magnitude: float = 0.3,
                               min_arousal_magnitude: float = 0.3) -> List[Segment]:
        """
        Find segments with strong emotional indicators.

        Emotional segments have significant deviations from neutral:
        - Non-neutral valence (positive or negative)
        - High arousal (emotional intensity)

        Args:
            min_valence_magnitude: Minimum absolute valence magnitude (0-1)
            min_arousal_magnitude: Minimum arousal magnitude (0-1)

        Returns:
            List of emotionally expressive segments
        """
        emotional_segments = []

        for segment in self.transcript.segments:
            if not segment.audio_state:
                continue

            if 'emotion' not in segment.audio_state:
                continue

            emotion_data = segment.audio_state['emotion']

            # Check dimensional emotion
            if 'dimensional' in emotion_data:
                dim = emotion_data['dimensional']
                valence = dim.get('valence', {}).get('score', 0.5)
                arousal = dim.get('arousal', {}).get('score', 0.5)

                # Check if emotional (away from neutral)
                valence_magnitude = abs(valence - 0.5)
                arousal_magnitude = abs(arousal - 0.5)

                if (valence_magnitude >= min_valence_magnitude or
                    arousal_magnitude >= min_arousal_magnitude):
                    emotional_segments.append(segment)

        return emotional_segments

    def find_hesitant_segments(self, min_pause_count: int = 2,
                              min_pause_density: float = 0.1) -> List[Segment]:
        """
        Find segments with hesitation indicators.

        Hesitant segments typically have:
        - Multiple pauses
        - High pause density
        - Slower speech rate

        Args:
            min_pause_count: Minimum number of pauses in segment
            min_pause_density: Minimum pause density (pauses/second)

        Returns:
            List of hesitant segments
        """
        hesitant_segments = []

        for segment in self.transcript.segments:
            if not segment.audio_state:
                continue

            if 'pauses' not in segment.audio_state:
                continue

            pauses = segment.audio_state['pauses']
            pause_count = pauses.get('count', 0)
            pause_density = pauses.get('density', 0)

            if pause_count >= min_pause_count or pause_density >= min_pause_density:
                hesitant_segments.append(segment)

        return hesitant_segments

    def create_annotated_transcript(self, output_path: Path) -> None:
        """
        Create a text-only transcript with audio feature annotations.

        This creates a readable transcript with inline annotations showing
        prosodic and emotional characteristics for each segment.

        Args:
            output_path: Path to write the annotated transcript
        """
        with output_path.open('w', encoding='utf-8') as f:
            f.write(f"# Annotated Transcript: {self.transcript.file_name}\n")
            f.write(f"# Language: {self.transcript.language}\n")
            f.write(f"# Generated from enriched transcript\n\n")

            for segment in self.transcript.segments:
                # Timestamp and text
                f.write(f"[{segment.start:6.2f}s - {segment.end:6.2f}s]\n")
                f.write(f"{segment.text}\n")

                # Audio annotations
                if segment.audio_state:
                    annotations = []

                    # Prosody
                    if 'pitch' in segment.audio_state:
                        pitch_info = segment.audio_state['pitch']
                        level = pitch_info.get('level', 'unknown')
                        contour = pitch_info.get('contour', '')
                        annotations.append(f"Pitch: {level} ({contour})")

                    if 'energy' in segment.audio_state:
                        energy_info = segment.audio_state['energy']
                        level = energy_info.get('level', 'unknown')
                        db_rms = energy_info.get('db_rms', 0)
                        annotations.append(f"Energy: {level} ({db_rms:.1f} dB)")

                    if 'rate' in segment.audio_state:
                        rate_info = segment.audio_state['rate']
                        level = rate_info.get('level', 'unknown')
                        syl_per_sec = rate_info.get('syllables_per_sec', 0)
                        annotations.append(
                            f"Rate: {level} ({syl_per_sec:.2f} syl/sec)"
                        )

                    # Emotion
                    if 'emotion' in segment.audio_state:
                        emotion = segment.audio_state['emotion']
                        if 'categorical' in emotion:
                            primary = emotion['categorical'].get('primary', 'neutral')
                            confidence = emotion['categorical'].get('confidence', 0)
                            annotations.append(
                                f"Emotion: {primary} ({confidence:.1%})"
                            )

                    # Pauses
                    if 'pauses' in segment.audio_state:
                        pauses = segment.audio_state['pauses']
                        count = pauses.get('count', 0)
                        if count > 0:
                            annotations.append(f"Pauses: {count}")

                    if annotations:
                        f.write("  >> ")
                        f.write(" | ".join(annotations))
                        f.write("\n")

                f.write("\n")

    def export_to_csv(self, output_path: Path) -> None:
        """
        Export enriched transcript to CSV for spreadsheet analysis.

        Each row represents a segment with all available metadata and features.

        Args:
            output_path: Path to write the CSV file
        """
        fieldnames = [
            'segment_id', 'start_time', 'end_time', 'duration', 'text',
            'pitch_level', 'pitch_mean_hz', 'pitch_contour',
            'energy_level', 'energy_db_rms',
            'rate_level', 'syllables_per_sec',
            'pause_count', 'pause_density',
            'emotion_category', 'emotion_confidence',
            'emotion_valence', 'emotion_arousal'
        ]

        with output_path.open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for segment in self.transcript.segments:
                row = {
                    'segment_id': segment.id,
                    'start_time': f"{segment.start:.3f}",
                    'end_time': f"{segment.end:.3f}",
                    'duration': f"{segment.end - segment.start:.3f}",
                    'text': segment.text,
                }

                # Extract audio state features
                if segment.audio_state:
                    # Pitch
                    if 'pitch' in segment.audio_state:
                        pitch = segment.audio_state['pitch']
                        row['pitch_level'] = pitch.get('level', '')
                        row['pitch_mean_hz'] = pitch.get('mean_hz', '')
                        row['pitch_contour'] = pitch.get('contour', '')

                    # Energy
                    if 'energy' in segment.audio_state:
                        energy = segment.audio_state['energy']
                        row['energy_level'] = energy.get('level', '')
                        row['energy_db_rms'] = energy.get('db_rms', '')

                    # Rate
                    if 'rate' in segment.audio_state:
                        rate = segment.audio_state['rate']
                        row['rate_level'] = rate.get('level', '')
                        row['syllables_per_sec'] = rate.get('syllables_per_sec', '')

                    # Pauses
                    if 'pauses' in segment.audio_state:
                        pauses = segment.audio_state['pauses']
                        row['pause_count'] = pauses.get('count', '')
                        row['pause_density'] = pauses.get('density', '')

                    # Emotion
                    if 'emotion' in segment.audio_state:
                        emotion = segment.audio_state['emotion']
                        if 'categorical' in emotion:
                            cat = emotion['categorical']
                            row['emotion_category'] = cat.get('primary', '')
                            row['emotion_confidence'] = cat.get('confidence', '')
                        if 'dimensional' in emotion:
                            dim = emotion['dimensional']
                            row['emotion_valence'] = dim.get('valence', {}).get('score', '')
                            row['emotion_arousal'] = dim.get('arousal', {}).get('score', '')

                writer.writerow(row)

        logger.info(f"Exported {len(self.transcript.segments)} segments to {output_path}")


def print_analysis_report(analyzer: TranscriptAnalyzer) -> None:
    """
    Print a comprehensive analysis report to the console.

    Args:
        analyzer: TranscriptAnalyzer instance with loaded transcript
    """
    print("\n" + "=" * 80)
    print("TRANSCRIPT ANALYSIS REPORT")
    print("=" * 80)

    print(f"\nFile: {analyzer.transcript.file_name}")
    print(f"Language: {analyzer.transcript.language}")

    # Audio characteristics
    audio_stats = analyzer.analyze_audio_characteristics()

    print(f"\n--- Audio Characteristics ---")
    print(f"Total segments: {audio_stats['total_segments']}")
    print(f"Total duration: {audio_stats['total_duration']:.1f}s")
    print(f"Segments with audio features: {audio_stats['segments_with_audio_state']}")

    if audio_stats['pitch']['mean'] != 0:
        print(f"\nPitch Analysis:")
        print(f"  Range: {audio_stats['pitch']['min']:.1f} - {audio_stats['pitch']['max']:.1f} Hz")
        print(f"  Mean: {audio_stats['pitch']['mean']:.1f} Hz")
        print(f"  Distribution: {audio_stats['pitch']['level_counts']}")
        if audio_stats['contour_distribution']:
            print(f"  Contours: {dict(audio_stats['contour_distribution'])}")

    if audio_stats['energy']['mean'] != 0:
        print(f"\nEnergy Analysis:")
        print(f"  Range: {audio_stats['energy']['min']:.1f} - {audio_stats['energy']['max']:.1f} dB")
        print(f"  Mean: {audio_stats['energy']['mean']:.1f} dB")
        print(f"  Distribution: {audio_stats['energy']['level_counts']}")

    if audio_stats['rate']['mean'] != 0:
        print(f"\nSpeech Rate Analysis:")
        print(f"  Range: {audio_stats['rate']['min']:.2f} - {audio_stats['rate']['max']:.2f} syl/sec")
        print(f"  Mean: {audio_stats['rate']['mean']:.2f} syl/sec")
        print(f"  Distribution: {audio_stats['rate']['level_counts']}")

    if audio_stats['pauses']['total'] > 0:
        print(f"\nPause Analysis:")
        print(f"  Total pauses: {audio_stats['pauses']['total']}")
        print(f"  Mean pauses per segment: {audio_stats['pauses']['mean_count']:.2f}")

    # Excited moments
    excited = analyzer.find_excited_moments()
    if excited:
        print(f"\n--- Excited Moments ({len(excited)}) ---")
        for seg in excited[:3]:  # Show first 3
            print(f"  [{seg.start:.1f}s] {seg.text[:60]}...")
        if len(excited) > 3:
            print(f"  ... and {len(excited) - 3} more")

    # Calm moments
    calm = analyzer.find_calm_moments()
    if calm:
        print(f"\n--- Calm Moments ({len(calm)}) ---")
        for seg in calm[:3]:  # Show first 3
            print(f"  [{seg.start:.1f}s] {seg.text[:60]}...")
        if len(calm) > 3:
            print(f"  ... and {len(calm) - 3} more")

    # Emotional segments
    emotional = analyzer.find_emotional_segments()
    if emotional:
        print(f"\n--- Emotionally Expressive Segments ({len(emotional)}) ---")
        for seg in emotional[:3]:  # Show first 3
            emotion = seg.audio_state.get('emotion', {})
            cat_emotion = emotion.get('categorical', {}).get('primary', '?')
            print(f"  [{seg.start:.1f}s] {cat_emotion}: {seg.text[:50]}...")
        if len(emotional) > 3:
            print(f"  ... and {len(emotional) - 3} more")

    # Hesitant segments
    hesitant = analyzer.find_hesitant_segments()
    if hesitant:
        print(f"\n--- Hesitant Segments ({len(hesitant)}) ---")
        for seg in hesitant[:3]:  # Show first 3
            pauses = seg.audio_state.get('pauses', {})
            pause_count = pauses.get('count', 0)
            print(f"  [{seg.start:.1f}s] ({pause_count} pauses) {seg.text[:50]}...")
        if len(hesitant) > 3:
            print(f"  ... and {len(hesitant) - 3} more")

    print("\n" + "=" * 80)


def main():
    """
    Main entry point demonstrating the complete workflow.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description='Complete workflow: transcribe, enrich, and analyze transcripts'
    )
    parser.add_argument('json_file', type=Path, help='Path to enriched JSON transcript')
    parser.add_argument('--audio', type=Path, help='Path to audio file')
    parser.add_argument(
        '--output-annotated', type=Path,
        help='Output path for annotated transcript'
    )
    parser.add_argument(
        '--output-csv', type=Path,
        help='Output path for CSV export'
    )

    args = parser.parse_args()

    # Validate input
    if not args.json_file.exists():
        print(f"Error: JSON file not found: {args.json_file}")
        sys.exit(1)

    # Initialize analyzer
    logger.info(f"Loading transcript: {args.json_file}")
    analyzer = TranscriptAnalyzer(args.json_file, args.audio)

    # Print analysis report
    print_analysis_report(analyzer)

    # Generate outputs
    if args.output_annotated:
        logger.info(f"Generating annotated transcript: {args.output_annotated}")
        analyzer.create_annotated_transcript(args.output_annotated)
        print(f"Annotated transcript saved to: {args.output_annotated}")

    if args.output_csv:
        logger.info(f"Generating CSV export: {args.output_csv}")
        analyzer.export_to_csv(args.output_csv)
        print(f"CSV export saved to: {args.output_csv}")

    print("\nWorkflow complete!")


if __name__ == '__main__':
    main()
