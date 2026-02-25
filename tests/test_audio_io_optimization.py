"""Tests for audio normalization performance optimization."""

import logging
import time
from unittest.mock import MagicMock, patch

from transcription import audio_io
from transcription.config import Paths

# Configure logging to verify log messages
logging.basicConfig(level=logging.INFO)


def test_normalize_all_skips_submission_for_uptodate_files(tmp_path, caplog):
    """
    Verify that normalize_all checks timestamps BEFORE submitting to ThreadPoolExecutor.
    This ensures we don't spawn threads/tasks for up-to-date files.
    """
    paths = Paths(root=tmp_path)
    audio_io.ensure_dirs(paths)

    # Create a source file and a corresponding up-to-date norm file
    src = paths.raw_dir / "test.wav"
    src.touch()
    dst = paths.norm_dir / "test.wav"
    dst.touch()

    # Set mtimes so dst is newer than src
    import os

    t = time.time()
    os.utime(src, (t, t))
    os.utime(dst, (t + 10, t + 10))

    # Mock check_ffmpeg_installation to avoid actual check
    with patch("transcription.audio_io.check_ffmpeg_installation"):
        # Mock ThreadPoolExecutor
        with patch("concurrent.futures.ThreadPoolExecutor") as MockExecutor:
            mock_executor_instance = MockExecutor.return_value
            mock_executor_instance.__enter__.return_value = mock_executor_instance

            # Mock submit to return a fake future if called (should not be called)
            fake_future = MagicMock()
            fake_future.result.return_value = MagicMock(abort=False, error=None)
            mock_executor_instance.submit.return_value = fake_future

            # Mock as_completed to behave like the real one (iterating over keys)
            # This ensures robustness whether submit is called or not.
            def side_effect(fs, **kwargs):
                return iter(fs)

            with patch("concurrent.futures.as_completed", side_effect=side_effect):
                # Run normalization
                with caplog.at_level(logging.INFO):
                    audio_io.normalize_all(paths)

            # Verification:
            # executor.submit should NOT be called because file is up-to-date
            mock_executor_instance.submit.assert_not_called()

            # Verify that we logged the skip
            assert "Skipped 1 already normalized files" in caplog.text
