"""
Tests for batch result models in transcription/models.py.

This test validates the BatchResult and EnrichmentResult classes:
- BatchFileResult: Result of processing a single file in batch transcription
- BatchProcessingResult: Summary of batch transcription operation
- EnrichmentFileResult: Result of enriching a single transcript file
- EnrichmentBatchResult: Summary of batch enrichment operation

Test coverage:
1. Creation with success/error/partial statuses
2. .to_dict() serialization (excludes transcript field)
3. .get_failures() filtering
4. .get_transcripts() filtering
5. Aggregation logic (successful/failed counts)
6. Edge cases (empty results, all failures)
"""

from transcription.models import (
    BatchFileResult,
    BatchProcessingResult,
    EnrichmentBatchResult,
    EnrichmentFileResult,
    Segment,
    Transcript,
)

# ============================================================================
# BatchFileResult tests
# ============================================================================


def test_batch_file_result_success() -> None:
    """BatchFileResult should properly represent a successful transcription."""
    seg = Segment(id=0, start=0.0, end=1.5, text="Hello world")
    transcript = Transcript(file_name="test.wav", language="en", segments=[seg])

    result = BatchFileResult(
        file_path="/path/to/test.wav",
        status="success",
        transcript=transcript,
    )

    assert result.file_path == "/path/to/test.wav"
    assert result.status == "success"
    assert result.transcript is not None
    assert result.transcript.file_name == "test.wav"
    assert result.error_type is None
    assert result.error_message is None


def test_batch_file_result_error() -> None:
    """BatchFileResult should properly represent a failed transcription."""
    result = BatchFileResult(
        file_path="/path/to/broken.wav",
        status="error",
        transcript=None,
        error_type="FileNotFoundError",
        error_message="Audio file not found",
    )

    assert result.file_path == "/path/to/broken.wav"
    assert result.status == "error"
    assert result.transcript is None
    assert result.error_type == "FileNotFoundError"
    assert result.error_message == "Audio file not found"


def test_batch_file_result_to_dict_success() -> None:
    """BatchFileResult.to_dict() should exclude transcript field (too large)."""
    seg = Segment(id=0, start=0.0, end=1.0, text="Test")
    transcript = Transcript(file_name="test.wav", language="en", segments=[seg])

    result = BatchFileResult(
        file_path="/path/to/test.wav",
        status="success",
        transcript=transcript,
    )

    d = result.to_dict()

    assert d["file_path"] == "/path/to/test.wav"
    assert d["status"] == "success"
    assert d["error_type"] is None
    assert d["error_message"] is None
    assert "transcript" not in d  # Should be excluded from serialization


def test_batch_file_result_to_dict_error() -> None:
    """BatchFileResult.to_dict() should include error information."""
    result = BatchFileResult(
        file_path="/path/to/broken.wav",
        status="error",
        error_type="ASRError",
        error_message="Transcription failed: CUDA out of memory",
    )

    d = result.to_dict()

    assert d["file_path"] == "/path/to/broken.wav"
    assert d["status"] == "error"
    assert d["error_type"] == "ASRError"
    assert d["error_message"] == "Transcription failed: CUDA out of memory"
    assert "transcript" not in d


# ============================================================================
# BatchProcessingResult tests
# ============================================================================


def test_batch_processing_result_creation() -> None:
    """BatchProcessingResult should properly aggregate counts."""
    seg1 = Segment(id=0, start=0.0, end=1.0, text="Success 1")
    seg2 = Segment(id=0, start=0.0, end=1.0, text="Success 2")
    transcript1 = Transcript(file_name="file1.wav", language="en", segments=[seg1])
    transcript2 = Transcript(file_name="file2.wav", language="en", segments=[seg2])

    results = [
        BatchFileResult(file_path="/path/file1.wav", status="success", transcript=transcript1),
        BatchFileResult(file_path="/path/file2.wav", status="success", transcript=transcript2),
        BatchFileResult(
            file_path="/path/file3.wav",
            status="error",
            error_type="FileNotFoundError",
            error_message="Not found",
        ),
    ]

    batch_result = BatchProcessingResult(
        total_files=3,
        successful=2,
        failed=1,
        results=results,
    )

    assert batch_result.total_files == 3
    assert batch_result.successful == 2
    assert batch_result.failed == 1
    assert len(batch_result.results) == 3


def test_batch_processing_result_get_failures() -> None:
    """BatchProcessingResult.get_failures() should return only failed results."""
    results = [
        BatchFileResult(
            file_path="/path/file1.wav",
            status="success",
            transcript=Transcript(file_name="file1.wav", language="en", segments=[]),
        ),
        BatchFileResult(
            file_path="/path/file2.wav",
            status="error",
            error_type="ASRError",
            error_message="Failed",
        ),
        BatchFileResult(
            file_path="/path/file3.wav",
            status="error",
            error_type="TimeoutError",
            error_message="Timeout",
        ),
    ]

    batch_result = BatchProcessingResult(total_files=3, successful=1, failed=2, results=results)

    failures = batch_result.get_failures()

    assert len(failures) == 2
    assert all(f.status == "error" for f in failures)
    assert failures[0].file_path == "/path/file2.wav"
    assert failures[1].file_path == "/path/file3.wav"


def test_batch_processing_result_get_transcripts() -> None:
    """BatchProcessingResult.get_transcripts() should return only successful transcripts."""
    transcript1 = Transcript(file_name="file1.wav", language="en", segments=[])
    transcript2 = Transcript(file_name="file2.wav", language="en", segments=[])

    results = [
        BatchFileResult(file_path="/path/file1.wav", status="success", transcript=transcript1),
        BatchFileResult(
            file_path="/path/file2.wav",
            status="error",
            error_type="ASRError",
            error_message="Failed",
        ),
        BatchFileResult(file_path="/path/file3.wav", status="success", transcript=transcript2),
    ]

    batch_result = BatchProcessingResult(total_files=3, successful=2, failed=1, results=results)

    transcripts = batch_result.get_transcripts()

    assert len(transcripts) == 2
    assert transcripts[0].file_name == "file1.wav"
    assert transcripts[1].file_name == "file2.wav"


def test_batch_processing_result_to_dict() -> None:
    """BatchProcessingResult.to_dict() should serialize results without transcripts."""
    results = [
        BatchFileResult(
            file_path="/path/file1.wav",
            status="success",
            transcript=Transcript(file_name="file1.wav", language="en", segments=[]),
        ),
        BatchFileResult(
            file_path="/path/file2.wav",
            status="error",
            error_type="ASRError",
            error_message="Failed",
        ),
    ]

    batch_result = BatchProcessingResult(total_files=2, successful=1, failed=1, results=results)

    d = batch_result.to_dict()

    assert d["total_files"] == 2
    assert d["successful"] == 1
    assert d["failed"] == 1
    assert len(d["results"]) == 2
    assert d["results"][0]["status"] == "success"
    assert d["results"][1]["status"] == "error"
    assert "transcript" not in d["results"][0]  # Should be excluded


def test_batch_processing_result_empty() -> None:
    """BatchProcessingResult should handle empty results list."""
    batch_result = BatchProcessingResult(total_files=0, successful=0, failed=0, results=[])

    assert batch_result.total_files == 0
    assert batch_result.successful == 0
    assert batch_result.failed == 0
    assert len(batch_result.results) == 0
    assert len(batch_result.get_failures()) == 0
    assert len(batch_result.get_transcripts()) == 0


def test_batch_processing_result_all_failures() -> None:
    """BatchProcessingResult should handle all-failure scenario."""
    results = [
        BatchFileResult(
            file_path="/path/file1.wav",
            status="error",
            error_type="Error1",
            error_message="Failed 1",
        ),
        BatchFileResult(
            file_path="/path/file2.wav",
            status="error",
            error_type="Error2",
            error_message="Failed 2",
        ),
    ]

    batch_result = BatchProcessingResult(total_files=2, successful=0, failed=2, results=results)

    assert batch_result.successful == 0
    assert batch_result.failed == 2
    assert len(batch_result.get_failures()) == 2
    assert len(batch_result.get_transcripts()) == 0


# ============================================================================
# EnrichmentFileResult tests
# ============================================================================


def test_enrichment_file_result_success() -> None:
    """EnrichmentFileResult should properly represent successful enrichment."""
    transcript = Transcript(file_name="test.wav", language="en", segments=[])

    result = EnrichmentFileResult(
        transcript_path="/path/to/test.json",
        status="success",
        enriched_transcript=transcript,
    )

    assert result.transcript_path == "/path/to/test.json"
    assert result.status == "success"
    assert result.enriched_transcript is not None
    assert result.error_type is None
    assert result.error_message is None
    assert len(result.warnings) == 0


def test_enrichment_file_result_partial() -> None:
    """EnrichmentFileResult should handle partial enrichment with warnings."""
    transcript = Transcript(file_name="test.wav", language="en", segments=[])

    result = EnrichmentFileResult(
        transcript_path="/path/to/test.json",
        status="partial",
        enriched_transcript=transcript,
        warnings=["Prosody extraction failed for segment 3", "Emotion model unavailable"],
    )

    assert result.status == "partial"
    assert result.enriched_transcript is not None
    assert len(result.warnings) == 2
    assert "Prosody extraction failed" in result.warnings[0]
    assert "Emotion model unavailable" in result.warnings[1]


def test_enrichment_file_result_error() -> None:
    """EnrichmentFileResult should properly represent failed enrichment."""
    result = EnrichmentFileResult(
        transcript_path="/path/to/broken.json",
        status="error",
        enriched_transcript=None,
        error_type="AudioNotFoundError",
        error_message="WAV file not found for enrichment",
    )

    assert result.transcript_path == "/path/to/broken.json"
    assert result.status == "error"
    assert result.enriched_transcript is None
    assert result.error_type == "AudioNotFoundError"
    assert result.error_message == "WAV file not found for enrichment"


def test_enrichment_file_result_to_dict_success() -> None:
    """EnrichmentFileResult.to_dict() should exclude enriched_transcript field."""
    transcript = Transcript(file_name="test.wav", language="en", segments=[])

    result = EnrichmentFileResult(
        transcript_path="/path/to/test.json",
        status="success",
        enriched_transcript=transcript,
    )

    d = result.to_dict()

    assert d["transcript_path"] == "/path/to/test.json"
    assert d["status"] == "success"
    assert d["error_type"] is None
    assert d["error_message"] is None
    assert d["warnings"] == []
    assert "enriched_transcript" not in d  # Should be excluded


def test_enrichment_file_result_to_dict_partial() -> None:
    """EnrichmentFileResult.to_dict() should include warnings for partial enrichment."""
    transcript = Transcript(file_name="test.wav", language="en", segments=[])

    result = EnrichmentFileResult(
        transcript_path="/path/to/test.json",
        status="partial",
        enriched_transcript=transcript,
        warnings=["Warning 1", "Warning 2"],
    )

    d = result.to_dict()

    assert d["status"] == "partial"
    assert d["warnings"] == ["Warning 1", "Warning 2"]
    assert "enriched_transcript" not in d


def test_enrichment_file_result_to_dict_error() -> None:
    """EnrichmentFileResult.to_dict() should include error information."""
    result = EnrichmentFileResult(
        transcript_path="/path/to/broken.json",
        status="error",
        error_type="ValidationError",
        error_message="Invalid transcript schema",
        warnings=["Schema version mismatch"],
    )

    d = result.to_dict()

    assert d["transcript_path"] == "/path/to/broken.json"
    assert d["status"] == "error"
    assert d["error_type"] == "ValidationError"
    assert d["error_message"] == "Invalid transcript schema"
    assert d["warnings"] == ["Schema version mismatch"]


# ============================================================================
# EnrichmentBatchResult tests
# ============================================================================


def test_enrichment_batch_result_creation() -> None:
    """EnrichmentBatchResult should properly aggregate counts."""
    transcript1 = Transcript(file_name="file1.wav", language="en", segments=[])
    transcript2 = Transcript(file_name="file2.wav", language="en", segments=[])

    results = [
        EnrichmentFileResult(
            transcript_path="/path/file1.json", status="success", enriched_transcript=transcript1
        ),
        EnrichmentFileResult(
            transcript_path="/path/file2.json",
            status="partial",
            enriched_transcript=transcript2,
            warnings=["Warning"],
        ),
        EnrichmentFileResult(
            transcript_path="/path/file3.json",
            status="error",
            error_type="Error",
            error_message="Failed",
        ),
    ]

    batch_result = EnrichmentBatchResult(
        total_files=3,
        successful=1,
        partial=1,
        failed=1,
        results=results,
    )

    assert batch_result.total_files == 3
    assert batch_result.successful == 1
    assert batch_result.partial == 1
    assert batch_result.failed == 1
    assert len(batch_result.results) == 3


def test_enrichment_batch_result_get_failures() -> None:
    """EnrichmentBatchResult.get_failures() should return only error results."""
    results = [
        EnrichmentFileResult(
            transcript_path="/path/file1.json",
            status="success",
            enriched_transcript=Transcript(file_name="file1.wav", language="en", segments=[]),
        ),
        EnrichmentFileResult(
            transcript_path="/path/file2.json",
            status="partial",
            enriched_transcript=Transcript(file_name="file2.wav", language="en", segments=[]),
            warnings=["Warning"],
        ),
        EnrichmentFileResult(
            transcript_path="/path/file3.json",
            status="error",
            error_type="Error1",
            error_message="Failed 1",
        ),
        EnrichmentFileResult(
            transcript_path="/path/file4.json",
            status="error",
            error_type="Error2",
            error_message="Failed 2",
        ),
    ]

    batch_result = EnrichmentBatchResult(
        total_files=4, successful=1, partial=1, failed=2, results=results
    )

    failures = batch_result.get_failures()

    assert len(failures) == 2
    assert all(f.status == "error" for f in failures)
    assert failures[0].transcript_path == "/path/file3.json"
    assert failures[1].transcript_path == "/path/file4.json"


def test_enrichment_batch_result_get_transcripts() -> None:
    """EnrichmentBatchResult.get_transcripts() should return success and partial transcripts."""
    transcript1 = Transcript(file_name="file1.wav", language="en", segments=[])
    transcript2 = Transcript(file_name="file2.wav", language="en", segments=[])

    results = [
        EnrichmentFileResult(
            transcript_path="/path/file1.json", status="success", enriched_transcript=transcript1
        ),
        EnrichmentFileResult(
            transcript_path="/path/file2.json",
            status="partial",
            enriched_transcript=transcript2,
            warnings=["Warning"],
        ),
        EnrichmentFileResult(
            transcript_path="/path/file3.json",
            status="error",
            error_type="Error",
            error_message="Failed",
        ),
    ]

    batch_result = EnrichmentBatchResult(
        total_files=3, successful=1, partial=1, failed=1, results=results
    )

    transcripts = batch_result.get_transcripts()

    assert len(transcripts) == 2  # Should include both success and partial
    assert transcripts[0].file_name == "file1.wav"
    assert transcripts[1].file_name == "file2.wav"


def test_enrichment_batch_result_to_dict() -> None:
    """EnrichmentBatchResult.to_dict() should serialize results without transcripts."""
    results = [
        EnrichmentFileResult(
            transcript_path="/path/file1.json",
            status="success",
            enriched_transcript=Transcript(file_name="file1.wav", language="en", segments=[]),
        ),
        EnrichmentFileResult(
            transcript_path="/path/file2.json",
            status="partial",
            enriched_transcript=Transcript(file_name="file2.wav", language="en", segments=[]),
            warnings=["Partial warning"],
        ),
        EnrichmentFileResult(
            transcript_path="/path/file3.json",
            status="error",
            error_type="Error",
            error_message="Failed",
        ),
    ]

    batch_result = EnrichmentBatchResult(
        total_files=3, successful=1, partial=1, failed=1, results=results
    )

    d = batch_result.to_dict()

    assert d["total_files"] == 3
    assert d["successful"] == 1
    assert d["partial"] == 1
    assert d["failed"] == 1
    assert len(d["results"]) == 3
    assert d["results"][0]["status"] == "success"
    assert d["results"][1]["status"] == "partial"
    assert d["results"][2]["status"] == "error"
    assert "enriched_transcript" not in d["results"][0]  # Should be excluded


def test_enrichment_batch_result_empty() -> None:
    """EnrichmentBatchResult should handle empty results list."""
    batch_result = EnrichmentBatchResult(
        total_files=0, successful=0, partial=0, failed=0, results=[]
    )

    assert batch_result.total_files == 0
    assert batch_result.successful == 0
    assert batch_result.partial == 0
    assert batch_result.failed == 0
    assert len(batch_result.results) == 0
    assert len(batch_result.get_failures()) == 0
    assert len(batch_result.get_transcripts()) == 0


def test_enrichment_batch_result_all_partial() -> None:
    """EnrichmentBatchResult should handle all-partial scenario."""
    transcript1 = Transcript(file_name="file1.wav", language="en", segments=[])
    transcript2 = Transcript(file_name="file2.wav", language="en", segments=[])

    results = [
        EnrichmentFileResult(
            transcript_path="/path/file1.json",
            status="partial",
            enriched_transcript=transcript1,
            warnings=["Warning 1"],
        ),
        EnrichmentFileResult(
            transcript_path="/path/file2.json",
            status="partial",
            enriched_transcript=transcript2,
            warnings=["Warning 2"],
        ),
    ]

    batch_result = EnrichmentBatchResult(
        total_files=2, successful=0, partial=2, failed=0, results=results
    )

    assert batch_result.successful == 0
    assert batch_result.partial == 2
    assert batch_result.failed == 0
    assert len(batch_result.get_failures()) == 0
    assert len(batch_result.get_transcripts()) == 2  # Partial results still return transcripts
