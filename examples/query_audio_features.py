#!/usr/bin/env python3
"""
Query Audio Features: Utilities for enriched transcripts.

This module provides specialized query functions for finding segments based on
audio characteristics:
- Excited moments (high pitch, high energy, fast speech)
- Calm moments (low pitch, low energy, slow speech)
- Emotional segments (non-neutral valence/arousal)
- Hesitant segments (many pauses, stuttering patterns)
- Summary statistics and trend analysis
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict
from statistics import mean, stdev
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from transcription.models import Transcript, Segment
from transcription.writers import load_transcript_from_json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AudioFeatureQuery:
    """
    Query interface for audio features in enriched transcripts.
    """

    def __init__(self, json_path: Path):
        """
        Initialize query interface with a transcript.

        Args:
            json_path: Path to enriched JSON transcript
        """
        self.json_path = Path(json_path)
        self.transcript = load_transcript_from_json(self.json_path)
        self._compute_statistics()

    def _compute_statistics(self) -> None:
        """Compute baseline statistics for all audio features."""
        self.pitch_values = []
        self.energy_values = []
        self.rate_values = []
        self.pause_counts = []
        self.pause_densities = []

        for segment in self.transcript.segments:
            if not segment.audio_state:
                continue

            if 'pitch' in segment.audio_state:
                val = segment.audio_state['pitch'].get('mean_hz')
                if val is not None:
                    self.pitch_values.append(val)

            if 'energy' in segment.audio_state:
                val = segment.audio_state['energy'].get('db_rms')
                if val is not None:
                    self.energy_values.append(val)

            if 'rate' in segment.audio_state:
                val = segment.audio_state['rate'].get('syllables_per_sec')
                if val is not None:
                    self.rate_values.append(val)

            if 'pauses' in segment.audio_state:
                val = segment.audio_state['pauses'].get('count')
                if val is not None:
                    self.pause_counts.append(val)
                val = segment.audio_state['pauses'].get('density')
                if val is not None:
                    self.pause_densities.append(val)

    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        Generate comprehensive summary statistics for all audio features.

        Returns:
            Dictionary with statistics for pitch, energy, speech rate, and pauses
        """
        def compute_stats(values: List[float]) -> Dict[str, float]:
            if not values:
                return {}
            return {
                'count': len(values),
                'mean': mean(values),
                'stdev': stdev(values) if len(values) > 1 else 0,
                'min': min(values),
                'max': max(values),
                'median': sorted(values)[len(values) // 2],
                'q25': sorted(values)[len(values) // 4],
                'q75': sorted(values)[3 * len(values) // 4],
            }

        return {
            'file': self.transcript.file_name,
            'total_segments': len(self.transcript.segments),
            'segments_with_features': sum(
                1 for seg in self.transcript.segments if seg.audio_state
            ),
            'pitch': compute_stats(self.pitch_values),
            'energy': compute_stats(self.energy_values),
            'rate': compute_stats(self.rate_values),
            'pauses': {
                'counts': compute_stats(self.pause_counts),
                'densities': compute_stats(self.pause_densities),
            },
        }

    def find_excited_moments(
        self,
        high_energy_percentile: float = 0.75,
        high_pitch_percentile: float = 0.75,
        fast_rate_percentile: float = 0.75
    ) -> List[Tuple[Segment, Dict[str, float]]]:
        """
        Find segments with excited speech characteristics.

        Uses percentile thresholds to identify segments that are above average
        in energy, pitch, and speech rate simultaneously.

        Args:
            high_energy_percentile: Threshold for energy (0-1)
            high_pitch_percentile: Threshold for pitch (0-1)
            fast_rate_percentile: Threshold for speech rate (0-1)

        Returns:
            List of (segment, feature_scores) tuples
        """
        if not (self.energy_values and self.pitch_values and self.rate_values):
            logger.warning("Not enough data for excited moment detection")
            return []

        # Calculate thresholds
        energy_threshold = sorted(self.energy_values)[
            int(len(self.energy_values) * high_energy_percentile)
        ]
        pitch_threshold = sorted(self.pitch_values)[
            int(len(self.pitch_values) * high_pitch_percentile)
        ]
        rate_threshold = sorted(self.rate_values)[
            int(len(self.rate_values) * fast_rate_percentile)
        ]

        excited = []

        for segment in self.transcript.segments:
            if not segment.audio_state:
                continue

            scores = {}
            match_count = 0

            if 'energy' in segment.audio_state:
                energy = segment.audio_state['energy'].get('db_rms')
                if energy and energy >= energy_threshold:
                    scores['energy'] = energy
                    match_count += 1

            if 'pitch' in segment.audio_state:
                pitch = segment.audio_state['pitch'].get('mean_hz')
                if pitch and pitch >= pitch_threshold:
                    scores['pitch'] = pitch
                    match_count += 1

            if 'rate' in segment.audio_state:
                rate = segment.audio_state['rate'].get('syllables_per_sec')
                if rate and rate >= rate_threshold:
                    scores['rate'] = rate
                    match_count += 1

            # Must match at least 2 out of 3 criteria
            if match_count >= 2:
                excited.append((segment, scores))

        return sorted(excited, key=lambda x: x[0].start)

    def find_calm_moments(
        self,
        low_energy_percentile: float = 0.25,
        low_pitch_percentile: float = 0.25,
        slow_rate_percentile: float = 0.25
    ) -> List[Tuple[Segment, Dict[str, float]]]:
        """
        Find segments with calm speech characteristics.

        Uses percentile thresholds to identify segments that are below average
        in energy, pitch, and speech rate.

        Args:
            low_energy_percentile: Threshold for energy (0-1)
            low_pitch_percentile: Threshold for pitch (0-1)
            slow_rate_percentile: Threshold for speech rate (0-1)

        Returns:
            List of (segment, feature_scores) tuples
        """
        if not (self.energy_values and self.pitch_values and self.rate_values):
            logger.warning("Not enough data for calm moment detection")
            return []

        # Calculate thresholds
        energy_threshold = sorted(self.energy_values)[
            int(len(self.energy_values) * low_energy_percentile)
        ]
        pitch_threshold = sorted(self.pitch_values)[
            int(len(self.pitch_values) * low_pitch_percentile)
        ]
        rate_threshold = sorted(self.rate_values)[
            int(len(self.rate_values) * slow_rate_percentile)
        ]

        calm = []

        for segment in self.transcript.segments:
            if not segment.audio_state:
                continue

            scores = {}
            match_count = 0

            if 'energy' in segment.audio_state:
                energy = segment.audio_state['energy'].get('db_rms')
                if energy and energy <= energy_threshold:
                    scores['energy'] = energy
                    match_count += 1

            if 'pitch' in segment.audio_state:
                pitch = segment.audio_state['pitch'].get('mean_hz')
                if pitch and pitch <= pitch_threshold:
                    scores['pitch'] = pitch
                    match_count += 1

            if 'rate' in segment.audio_state:
                rate = segment.audio_state['rate'].get('syllables_per_sec')
                if rate and rate <= rate_threshold:
                    scores['rate'] = rate
                    match_count += 1

            # Must match at least 2 out of 3 criteria
            if match_count >= 2:
                calm.append((segment, scores))

        return sorted(calm, key=lambda x: x[0].start)

    def find_emotional_segments(
        self,
        min_valence_magnitude: float = 0.2,
        min_arousal_magnitude: float = 0.2
    ) -> List[Tuple[Segment, Dict[str, Any]]]:
        """
        Find segments with strong emotional indicators.

        Segments with high emotional intensity in either dimensional or
        categorical emotion representations.

        Args:
            min_valence_magnitude: Minimum magnitude away from neutral valence
            min_arousal_magnitude: Minimum magnitude away from neutral arousal

        Returns:
            List of (segment, emotion_data) tuples
        """
        emotional = []

        for segment in self.transcript.segments:
            if not segment.audio_state or 'emotion' not in segment.audio_state:
                continue

            emotion = segment.audio_state['emotion']
            emotion_data = {}

            # Check dimensional emotion
            if 'dimensional' in emotion:
                dim = emotion['dimensional']
                valence = dim.get('valence', {}).get('score', 0.5)
                arousal = dim.get('arousal', {}).get('score', 0.5)

                valence_mag = abs(valence - 0.5)
                arousal_mag = abs(arousal - 0.5)

                if valence_mag >= min_valence_magnitude or arousal_mag >= min_arousal_magnitude:
                    emotion_data['valence'] = valence
                    emotion_data['arousal'] = arousal

            # Check categorical emotion
            if 'categorical' in emotion:
                cat = emotion['categorical']
                primary = cat.get('primary')
                confidence = cat.get('confidence', 0)
                if primary and confidence > 0.5:
                    emotion_data['category'] = primary
                    emotion_data['confidence'] = confidence

            if emotion_data:
                emotional.append((segment, emotion_data))

        return sorted(emotional, key=lambda x: x[0].start)

    def find_hesitant_segments(
        self,
        min_pause_count: int = 1,
        min_pause_density: float = 0.05
    ) -> List[Tuple[Segment, Dict[str, Any]]]:
        """
        Find segments with hesitation indicators.

        Hesitant segments show multiple pauses, high pause density, or
        stuttering-like patterns.

        Args:
            min_pause_count: Minimum number of pauses in segment
            min_pause_density: Minimum pause density (pauses/second)

        Returns:
            List of (segment, pause_data) tuples
        """
        hesitant = []

        for segment in self.transcript.segments:
            if not segment.audio_state or 'pauses' not in segment.audio_state:
                continue

            pauses = segment.audio_state['pauses']
            pause_count = pauses.get('count', 0)
            pause_density = pauses.get('density', 0)

            if pause_count >= min_pause_count or pause_density >= min_pause_density:
                hesitant.append((
                    segment,
                    {
                        'pause_count': pause_count,
                        'pause_density': pause_density
                    }
                ))

        return sorted(hesitant, key=lambda x: x[0].start)

    def find_high_pitch_moments(self, percentile: float = 0.75) -> List[Segment]:
        """
        Find all segments with notably high pitch.

        Args:
            percentile: Percentile threshold (0-1)

        Returns:
            List of segments
        """
        if not self.pitch_values:
            return []

        threshold = sorted(self.pitch_values)[int(len(self.pitch_values) * percentile)]
        result = []

        for segment in self.transcript.segments:
            if segment.audio_state and 'pitch' in segment.audio_state:
                pitch = segment.audio_state['pitch'].get('mean_hz')
                if pitch and pitch >= threshold:
                    result.append(segment)

        return result

    def find_low_pitch_moments(self, percentile: float = 0.25) -> List[Segment]:
        """
        Find all segments with notably low pitch.

        Args:
            percentile: Percentile threshold (0-1)

        Returns:
            List of segments
        """
        if not self.pitch_values:
            return []

        threshold = sorted(self.pitch_values)[int(len(self.pitch_values) * percentile)]
        result = []

        for segment in self.transcript.segments:
            if segment.audio_state and 'pitch' in segment.audio_state:
                pitch = segment.audio_state['pitch'].get('mean_hz')
                if pitch and pitch <= threshold:
                    result.append(segment)

        return result

    def get_emotion_distribution(self) -> Dict[str, int]:
        """
        Get distribution of categorical emotions across transcript.

        Returns:
            Dictionary mapping emotion categories to counts
        """
        distribution = defaultdict(int)

        for segment in self.transcript.segments:
            if segment.audio_state and 'emotion' in segment.audio_state:
                emotion = segment.audio_state['emotion']
                if 'categorical' in emotion:
                    category = emotion['categorical'].get('primary', 'unknown')
                    distribution[category] += 1

        return dict(distribution)

    def get_pitch_contour_distribution(self) -> Dict[str, int]:
        """
        Get distribution of pitch contours (rising, falling, flat).

        Returns:
            Dictionary mapping contour types to counts
        """
        distribution = defaultdict(int)

        for segment in self.transcript.segments:
            if segment.audio_state and 'pitch' in segment.audio_state:
                contour = segment.audio_state['pitch'].get('contour', 'unknown')
                distribution[contour] += 1

        return dict(distribution)

    def analyze_temporal_trends(self, window_size: int = 5) -> Dict[str, List[float]]:
        """
        Analyze temporal trends in audio features using a sliding window.

        Args:
            window_size: Size of the sliding window (segments)

        Returns:
            Dictionary with trend data for each feature
        """
        trends = {
            'pitch_trend': [],
            'energy_trend': [],
            'rate_trend': [],
        }

        segments_list = self.transcript.segments

        for i in range(len(segments_list) - window_size + 1):
            window = segments_list[i:i + window_size]

            # Pitch trend
            pitch_vals = [
                seg.audio_state['pitch'].get('mean_hz')
                for seg in window
                if seg.audio_state and 'pitch' in seg.audio_state
                and seg.audio_state['pitch'].get('mean_hz')
            ]
            if pitch_vals:
                trends['pitch_trend'].append(mean(pitch_vals))

            # Energy trend
            energy_vals = [
                seg.audio_state['energy'].get('db_rms')
                for seg in window
                if seg.audio_state and 'energy' in seg.audio_state
                and seg.audio_state['energy'].get('db_rms')
            ]
            if energy_vals:
                trends['energy_trend'].append(mean(energy_vals))

            # Rate trend
            rate_vals = [
                seg.audio_state['rate'].get('syllables_per_sec')
                for seg in window
                if seg.audio_state and 'rate' in seg.audio_state
                and seg.audio_state['rate'].get('syllables_per_sec')
            ]
            if rate_vals:
                trends['rate_trend'].append(mean(rate_vals))

        return trends


def print_summary_report(query: AudioFeatureQuery) -> None:
    """
    Print a comprehensive summary report.

    Args:
        query: AudioFeatureQuery instance
    """
    print("\n" + "=" * 80)
    print("AUDIO FEATURE SUMMARY REPORT")
    print("=" * 80)

    stats = query.get_summary_statistics()

    print(f"\nFile: {stats['file']}")
    print(f"Total segments: {stats['total_segments']}")
    print(f"Segments with audio features: {stats['segments_with_features']}")

    # Pitch statistics
    if stats['pitch']:
        print(f"\n--- Pitch ---")
        for key, val in stats['pitch'].items():
            if isinstance(val, float):
                print(f"  {key:15s}: {val:8.1f} Hz")

    # Energy statistics
    if stats['energy']:
        print(f"\n--- Energy ---")
        for key, val in stats['energy'].items():
            if isinstance(val, float):
                print(f"  {key:15s}: {val:8.1f} dB")

    # Rate statistics
    if stats['rate']:
        print(f"\n--- Speech Rate ---")
        for key, val in stats['rate'].items():
            if isinstance(val, float):
                print(f"  {key:15s}: {val:8.2f} syl/sec")

    # Pause statistics
    if stats['pauses']['counts']:
        print(f"\n--- Pause Counts ---")
        for key, val in stats['pauses']['counts'].items():
            if isinstance(val, float):
                print(f"  {key:15s}: {val:8.2f}")

    # Feature distributions
    emotion_dist = query.get_emotion_distribution()
    if emotion_dist:
        print(f"\n--- Emotion Distribution ---")
        total = sum(emotion_dist.values())
        for emotion, count in sorted(emotion_dist.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total) * 100
            bar = "█" * int(percentage / 5)
            print(f"  {emotion:15s}: {bar:20s} {count:3d} ({percentage:5.1f}%)")

    contour_dist = query.get_pitch_contour_distribution()
    if contour_dist:
        print(f"\n--- Pitch Contour Distribution ---")
        total = sum(contour_dist.values())
        for contour, count in sorted(contour_dist.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total) * 100
            bar = "█" * int(percentage / 5)
            print(f"  {contour:15s}: {bar:20s} {count:3d} ({percentage:5.1f}%)")

    print("\n" + "=" * 80)


def main():
    """
    Main entry point with command-line interface.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description='Query and analyze audio features in enriched transcripts'
    )
    parser.add_argument('json_file', type=Path, help='Path to enriched JSON transcript')
    parser.add_argument(
        '--excited', action='store_true',
        help='Find excited moments (high pitch, high energy, fast speech)'
    )
    parser.add_argument(
        '--calm', action='store_true',
        help='Find calm moments (low pitch, low energy, slow speech)'
    )
    parser.add_argument(
        '--emotional', action='store_true',
        help='Find emotionally expressive segments'
    )
    parser.add_argument(
        '--hesitant', action='store_true',
        help='Find hesitant segments (many pauses)'
    )
    parser.add_argument(
        '--summary', action='store_true',
        help='Print summary statistics'
    )
    parser.add_argument(
        '--all', action='store_true',
        help='Run all analyses'
    )

    args = parser.parse_args()

    # Validate input
    if not args.json_file.exists():
        print(f"Error: JSON file not found: {args.json_file}")
        sys.exit(1)

    # Initialize query
    logger.info(f"Loading transcript: {args.json_file}")
    query = AudioFeatureQuery(args.json_file)

    # Default to summary if no specific analysis requested
    if not any([args.excited, args.calm, args.emotional, args.hesitant, args.summary, args.all]):
        args.summary = True

    # Print summary
    if args.summary or args.all:
        print_summary_report(query)

    # Find excited moments
    if args.excited or args.all:
        excited = query.find_excited_moments()
        if excited:
            print(f"\n--- Excited Moments ({len(excited)}) ---")
            for segment, scores in excited[:5]:
                print(f"  [{segment.start:6.2f}s] {segment.text[:70]}")
                for feature, value in scores.items():
                    print(f"    {feature}: {value:.2f}")
            if len(excited) > 5:
                print(f"  ... and {len(excited) - 5} more")
        else:
            print("\nNo excited moments found.")

    # Find calm moments
    if args.calm or args.all:
        calm = query.find_calm_moments()
        if calm:
            print(f"\n--- Calm Moments ({len(calm)}) ---")
            for segment, scores in calm[:5]:
                print(f"  [{segment.start:6.2f}s] {segment.text[:70]}")
                for feature, value in scores.items():
                    print(f"    {feature}: {value:.2f}")
            if len(calm) > 5:
                print(f"  ... and {len(calm) - 5} more")
        else:
            print("\nNo calm moments found.")

    # Find emotional segments
    if args.emotional or args.all:
        emotional = query.find_emotional_segments()
        if emotional:
            print(f"\n--- Emotionally Expressive Segments ({len(emotional)}) ---")
            for segment, emotion_data in emotional[:5]:
                category = emotion_data.get('category', '?')
                valence = emotion_data.get('valence', '?')
                print(f"  [{segment.start:6.2f}s] {category} - {segment.text[:70]}")
                if isinstance(valence, float):
                    print(f"    Valence: {valence:.2f}, Arousal: {emotion_data.get('arousal', 0):.2f}")
            if len(emotional) > 5:
                print(f"  ... and {len(emotional) - 5} more")
        else:
            print("\nNo emotionally expressive segments found.")

    # Find hesitant segments
    if args.hesitant or args.all:
        hesitant = query.find_hesitant_segments()
        if hesitant:
            print(f"\n--- Hesitant Segments ({len(hesitant)}) ---")
            for segment, pause_data in hesitant[:5]:
                pauses = pause_data.get('pause_count', 0)
                print(f"  [{segment.start:6.2f}s] ({pauses} pauses) {segment.text[:70]}")
            if len(hesitant) > 5:
                print(f"  ... and {len(hesitant) - 5} more")
        else:
            print("\nNo hesitant segments found.")


if __name__ == '__main__':
    main()
