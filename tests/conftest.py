"""
Pytest configuration and fixtures for tests.

This module provides:
- Mock objects for optional dependencies
- Common test fixtures
- Test configuration
"""

import sys
from unittest.mock import Mock, MagicMock


# ============================================================================
# Mock unavailable dependencies to allow tests to run
# ============================================================================

def mock_module(module_name, attrs=None):
    """Create a mock module with optional attributes."""
    mock = MagicMock()
    if attrs:
        for attr_name, attr_value in attrs.items():
            setattr(mock, attr_name, attr_value)
    return mock


# Mock faster_whisper if not available
try:
    import faster_whisper
except ImportError:
    sys.modules['faster_whisper'] = mock_module('faster_whisper', {
        'WhisperModel': MagicMock
    })


# Mock transformers if not available (for emotion tests)
try:
    import transformers
except ImportError:
    sys.modules['transformers'] = mock_module('transformers', {
        'AutoModelForAudioClassification': MagicMock,
        'Wav2Vec2Processor': MagicMock,
        'pipeline': MagicMock
    })


# Mock torch if not available (for emotion tests)
try:
    import torch
except ImportError:
    sys.modules['torch'] = mock_module('torch', {
        'cuda': mock_module('torch.cuda', {'is_available': lambda: False}),
        'no_grad': lambda: MagicMock(__enter__=lambda x: None, __exit__=lambda *args: None),
        'nn': mock_module('torch.nn', {
            'functional': mock_module('torch.nn.functional', {
                'softmax': lambda x, dim: x
            })
        })
    })


# Mock parselmouth if not available (for prosody tests)
try:
    import parselmouth
except ImportError:
    sys.modules['parselmouth'] = mock_module('parselmouth', {
        'Sound': MagicMock,
        'praat': mock_module('parselmouth.praat', {
            'call': MagicMock(return_value=None)
        })
    })


# Mock librosa if not available (for prosody tests)
try:
    import librosa
except ImportError:
    sys.modules['librosa'] = mock_module('librosa', {
        'feature': mock_module('librosa.feature', {
            'rms': MagicMock(return_value=[[0.1]])
        }),
        'frames_to_time': lambda frames, sr, hop_length: frames * hop_length / sr
    })


# Mock soundfile if not available
SOUNDFILE_AVAILABLE = False
try:
    import soundfile
    SOUNDFILE_AVAILABLE = True
except ImportError:
    import numpy as np

    class MockSoundFile:
        def __init__(self, path, mode='r'):
            self.samplerate = 16000
            self.channels = 1

        def __len__(self):
            return 16000  # 1 second

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def seek(self, frame):
            pass

        def read(self, frames, dtype='float32'):
            return np.zeros(frames, dtype=dtype)

    def mock_read(path, dtype='float32'):
        return (np.zeros(16000, dtype=dtype), 16000)

    def mock_write(path, data, sr):
        # Create an empty file so file existence checks pass
        import pathlib
        pathlib.Path(path).touch()

    sys.modules['soundfile'] = mock_module('soundfile', {
        'SoundFile': MockSoundFile,
        'read': mock_read,
        'write': mock_write
    })


# Make sure numpy is available (it's usually in the base system)
try:
    import numpy
except ImportError:
    # If numpy is not available, we can't really run tests
    raise ImportError(
        "numpy is required for tests. "
        "Please install with: pip install numpy"
    )
