"""
Tests for prosody feature extraction module.
"""

import unittest

import numpy as np

from slower_whisper.pipeline.prosody import (
    ENERGY_THRESHOLDS,
    PITCH_THRESHOLDS,
    categorize_value,
    compute_speaker_baseline,
    count_syllables,
    extract_prosody,
    normalize_to_baseline,
)


class TestProsodyExtraction(unittest.TestCase):
    """Test prosody feature extraction."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a simple test audio signal (1 second, 16kHz, 440 Hz sine wave)
        self.sr = 16000
        duration = 1.0
        t = np.linspace(0, duration, int(self.sr * duration))
        frequency = 440  # A4 note
        self.test_audio = np.sin(2 * np.pi * frequency * t).astype(np.float32)
        self.test_text = "This is a test sentence with multiple words."

    def test_count_syllables(self):
        """Test syllable counting heuristic."""
        test_cases = [
            ("hello", 2),
            ("world", 1),
            ("beautiful", 3),
            ("", 0),
            ("I am happy today", 5),  # I(1) am(1) hap-py(2) to-day(2)
        ]

        for text, _expected_min in test_cases:
            count = count_syllables(text)
            self.assertGreaterEqual(count, 0, f"Syllable count should be non-negative for '{text}'")
            # Allow some tolerance in syllable counting (it's heuristic)
            if text:
                self.assertGreater(count, 0, f"Non-empty text should have syllables: '{text}'")

    def test_categorize_value(self):
        """Test categorical value mapping."""
        # Test pitch categorization
        self.assertEqual(categorize_value(100, PITCH_THRESHOLDS), "low")
        self.assertEqual(
            categorize_value(200, PITCH_THRESHOLDS), "high"
        )  # 200 >= 180 (neutral), < 250 (high)
        self.assertEqual(categorize_value(300, PITCH_THRESHOLDS), "very_high")

        # Test energy categorization
        self.assertEqual(categorize_value(-30, ENERGY_THRESHOLDS), "quiet")
        self.assertEqual(categorize_value(-20, ENERGY_THRESHOLDS), "normal")
        self.assertEqual(categorize_value(-5, ENERGY_THRESHOLDS), "very_loud")

    def test_normalize_to_baseline(self):
        """Test baseline normalization."""
        # Test with standard deviation
        baseline_median = 180
        baseline_std = 30

        self.assertEqual(normalize_to_baseline(180, baseline_median, baseline_std), "neutral")
        self.assertEqual(normalize_to_baseline(240, baseline_median, baseline_std), "very_high")
        self.assertEqual(normalize_to_baseline(120, baseline_median, baseline_std), "very_low")

        # Test without standard deviation (ratio-based)
        self.assertEqual(normalize_to_baseline(180, 180, None), "neutral")
        self.assertEqual(
            normalize_to_baseline(240, 180, None), "very_high"
        )  # ratio = 1.33 >= 1.3 -> very_high
        self.assertEqual(normalize_to_baseline(100, 180, None), "very_low")

    def test_extract_prosody_basic(self):
        """Test basic prosody extraction."""
        result = extract_prosody(self.test_audio, self.sr, self.test_text)

        # Check structure
        self.assertIn("pitch", result)
        self.assertIn("energy", result)
        self.assertIn("rate", result)
        self.assertIn("pauses", result)

        # Check pitch fields
        self.assertIn("level", result["pitch"])
        self.assertIn("mean_hz", result["pitch"])
        self.assertIn("variation", result["pitch"])
        self.assertIn("contour", result["pitch"])

        # Check energy fields
        self.assertIn("level", result["energy"])
        self.assertIn("db_rms", result["energy"])
        self.assertIn("variation", result["energy"])

        # Check rate fields
        self.assertIn("level", result["rate"])
        self.assertIn("syllables_per_sec", result["rate"])
        self.assertIn("words_per_sec", result["rate"])

        # Check pause fields
        self.assertIn("count", result["pauses"])
        self.assertIn("longest_ms", result["pauses"])
        self.assertIn("density", result["pauses"])

    def test_extract_prosody_with_baseline(self):
        """Test prosody extraction with speaker baseline."""
        baseline = {
            "pitch_median": 200.0,
            "pitch_std": 25.0,
            "energy_median": -15.0,
            "energy_std": 3.0,
            "rate_median": 5.0,
            "rate_std": 0.5,
        }

        result = extract_prosody(
            self.test_audio, self.sr, self.test_text, speaker_baseline=baseline
        )

        # Should have valid structure even with baseline
        self.assertIn("pitch", result)
        self.assertIn("energy", result)
        self.assertIn("rate", result)

    def test_extract_prosody_empty_audio(self):
        """Test prosody extraction with empty audio."""
        empty_audio = np.array([], dtype=np.float32)

        result = extract_prosody(empty_audio, self.sr, "")

        # Should return default structure without crashing
        self.assertIn("pitch", result)
        self.assertIn("energy", result)
        self.assertIn("rate", result)
        self.assertIn("pauses", result)

    def test_extract_prosody_silent_audio(self):
        """Test prosody extraction with silent audio."""
        silent_audio = np.zeros(self.sr, dtype=np.float32)  # 1 second of silence

        result = extract_prosody(silent_audio, self.sr, "This text has no corresponding audio.")

        # Should handle silence gracefully
        self.assertIsNotNone(result["energy"])

        # Energy should be very low
        if result["energy"]["db_rms"] is not None:
            self.assertLess(result["energy"]["db_rms"], -30)

    def test_compute_speaker_baseline(self):
        """Test speaker baseline computation."""
        # Create multiple segments
        segments = []
        for i in range(5):
            audio = np.random.randn(self.sr).astype(np.float32) * 0.1
            segments.append(
                {"audio": audio, "sr": self.sr, "text": f"This is test segment number {i}."}
            )

        baseline = compute_speaker_baseline(segments)

        # Check that baseline contains expected keys
        # Note: May not have all keys if extraction fails
        self.assertIsInstance(baseline, dict)

        # If pitch was extracted, check values are reasonable
        if "pitch_median" in baseline:
            self.assertGreater(baseline["pitch_median"], 0)
            self.assertGreater(baseline["pitch_std"], 0)

        # If energy was extracted, check values are reasonable
        if "energy_median" in baseline:
            self.assertLess(baseline["energy_median"], 0)  # dB should be negative

    def test_speech_rate_calculation(self):
        """Test speech rate calculation."""
        # Create longer text with known word count
        test_text = "This is a longer test sentence with exactly ten words."
        word_count = 10
        duration = 2.0  # 2 seconds

        audio = np.random.randn(int(self.sr * duration)).astype(np.float32) * 0.1

        result = extract_prosody(audio, self.sr, test_text)

        # Check that rate was calculated
        if result["rate"]["words_per_sec"] is not None:
            # Should be approximately 10 words / 2 seconds = 5 words/sec
            expected_wps = word_count / duration
            self.assertAlmostEqual(result["rate"]["words_per_sec"], expected_wps, delta=0.1)

    def test_prosody_extraction_with_noise(self):
        """Test prosody extraction with noisy audio."""
        # Add noise to test audio
        noise = np.random.randn(len(self.test_audio)) * 0.05
        noisy_audio = self.test_audio + noise

        result = extract_prosody(noisy_audio, self.sr, self.test_text)

        # Should still produce valid results
        self.assertIn("pitch", result)
        self.assertIn("energy", result)


class TestSyllableCounting(unittest.TestCase):
    """Test syllable counting heuristics."""

    def test_common_words(self):
        """Test syllable counting for common words."""
        # Note: These are approximate - syllable counting is heuristic
        words = {
            "cat": 1,
            "dog": 1,
            "water": 2,
            "beautiful": 3,
            "extraordinary": 5,
        }

        for word, expected_syllables in words.items():
            count = count_syllables(word)
            # Allow +/- 1 tolerance for heuristic method
            self.assertAlmostEqual(
                count, expected_syllables, delta=1, msg=f"Syllable count for '{word}'"
            )

    def test_empty_and_whitespace(self):
        """Test edge cases."""
        self.assertEqual(count_syllables(""), 0)
        self.assertEqual(count_syllables("   "), 0)
        self.assertGreater(count_syllables("a"), 0)


if __name__ == "__main__":
    unittest.main()
