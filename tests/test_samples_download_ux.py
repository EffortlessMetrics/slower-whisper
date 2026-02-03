import os
from pathlib import Path
from unittest.mock import patch

import pytest

from transcription.cli import main
from transcription.samples import SampleDataset


class TestSamplesDownloadUX:
    """Test UX interactions for samples download command."""

    @pytest.fixture
    def mock_datasets(self):
        """Mock SAMPLE_DATASETS to include a downloadable test dataset."""
        test_dataset = SampleDataset(
            name="test_dataset",
            description="Test Dataset",
            url="http://example.com/test.zip",
            sha256="",
            archive_format="zip",
            test_files=["test.wav"],
            source_url="http://example.com",
            license="MIT",
        )
        with patch.dict("transcription.samples.SAMPLE_DATASETS", {"test_dataset": test_dataset}):
            yield test_dataset

    @patch("transcription.samples.extract_archive")
    @patch("transcription.samples.verify_sha256")
    @patch("urllib.request.urlretrieve")
    def test_download_progress_bar(
        self, mock_retrieve, mock_verify, mock_extract, mock_datasets, capsys, tmp_path
    ):
        """Verify progress bar is displayed during download."""

        # Mock urlretrieve to call the reporthook
        def side_effect(url, dest, reporthook=None, data=None):
            if reporthook:
                # Simulate progress: 0%, 50%, 100%
                # block_num, block_size, total_size
                # total = 10MB (10 * 1024 * 1024)
                total = 10 * 1024 * 1024
                # 0 bytes
                reporthook(0, 1024, total)
                # 5MB (approx 5120 blocks of 1024 bytes)
                reporthook(5120, 1024, total)
                # 10MB
                reporthook(10240, 1024, total)

            # Create dummy file
            Path(dest).touch()
            return dest, None

        mock_retrieve.side_effect = side_effect
        mock_verify.return_value = True

        # Simulate extraction creating the test file
        def extract_side_effect(archive, dest, fmt, members=None):
            dest.mkdir(parents=True, exist_ok=True)
            (dest / "test.wav").touch()

        mock_extract.side_effect = extract_side_effect

        # Override cache location to tmp_path
        with patch.dict(os.environ, {"SLOWER_WHISPER_SAMPLES": str(tmp_path)}):
            exit_code = main(["samples", "download", "test_dataset", "--force"])

        assert exit_code == 0

        captured = capsys.readouterr()

        # Check for progress bar output
        # Note: capsys captures all output including carriage returns
        assert "Downloading:   0.0%" in captured.out
        assert "Downloading:  50.0%" in captured.out
        assert "Downloading: 100.0%" in captured.out
        assert "Test files ready:" in captured.out
