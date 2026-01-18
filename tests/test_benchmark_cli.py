import pytest
from unittest.mock import MagicMock, patch, ANY
from pathlib import Path
from transcription.benchmark_cli import StreamingBenchmarkRunner
from transcription.benchmarks import EvalSample
from transcription.streaming import StreamChunk
import types

@pytest.fixture
def mock_asr_engine():
    with patch("transcription.asr_engine.TranscriptionEngine") as mock:
        engine_instance = mock.return_value
        # Mock _transcribe_with_model to return a generator of mock segments
        segment1 = MagicMock(start=0.0, end=1.0, text="hello")
        segment2 = MagicMock(start=1.0, end=2.0, text="world")
        info = MagicMock(duration=2.0)

        # Configure _transcribe_with_model to return (generator, info)
        engine_instance._transcribe_with_model.return_value = ([segment1, segment2], info)
        yield engine_instance

@pytest.fixture
def mock_streaming_session():
    # Since StreamingEnrichmentSession is imported inside evaluate_sample from .streaming_enrich,
    # and we can't easily patch local imports without patching sys.modules or using other tricks,
    # the easiest way is to patch transcription.streaming_enrich.StreamingEnrichmentSession

    with patch("transcription.streaming_enrich.StreamingEnrichmentSession") as mock:
        session_instance = mock.return_value
        session_instance._extractor.duration_seconds = 2.0
        yield session_instance

def test_streaming_benchmark_runner_evaluate_sample(mock_asr_engine, mock_streaming_session):
    # Setup
    runner = StreamingBenchmarkRunner(track="streaming", dataset="test_ds")
    sample = EvalSample(
        dataset="test_ds",
        id="test_001",
        audio_path=Path("dummy.wav"),
        reference_transcript="hello world"
    )

    # Run
    result = runner.evaluate_sample(sample)

    # Verify ASR engine called
    mock_asr_engine._transcribe_with_model.assert_called_once_with(Path("dummy.wav"))

    # Verify chunks ingested
    assert mock_streaming_session.ingest_chunk.call_count == 2

    # Check first chunk argument
    call_args_list = mock_streaming_session.ingest_chunk.call_args_list
    chunk1 = call_args_list[0][0][0] # First arg of first call
    assert chunk1["text"] == "hello"
    assert chunk1["start"] == 0.0

    # Verify end_of_stream called
    mock_streaming_session.end_of_stream.assert_called_once()

    # Verify result structure
    assert result["id"] == "test_001"
    assert "latency_first_token_ms" in result
    assert "latency_total_ms" in result
    assert result["audio_duration_s"] == 2.0
    assert "rtf" in result
