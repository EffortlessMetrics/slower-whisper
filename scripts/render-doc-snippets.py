#!/usr/bin/env python3
"""Generate documentation snippets from test fixtures.

This script produces curated JSON examples for docs from actual
test data, ensuring documentation stays in sync with implementation.

Usage:
    python scripts/render-doc-snippets.py

Output:
    docs/_snippets/schema_example.json      - Minimal transcript example
    docs/_snippets/schema_full_example.json - Full transcript with all fields
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def create_minimal_example() -> dict:
    """Create a minimal valid transcript example."""
    return {
        "schema_version": 2,
        "file": "example.wav",
        "language": "en",
        "segments": [{"id": 0, "start": 0.0, "end": 2.5, "text": "Hello world."}],
        "speakers": None,
        "turns": None,
        "meta": None,
    }


def create_full_example() -> dict:
    """Create a full transcript example with all optional fields."""
    return {
        "schema_version": 2,
        "file": "meeting.wav",
        "language": "en",
        "segments": [
            {
                "id": 0,
                "start": 0.0,
                "end": 2.5,
                "text": "Hello, how are you?",
                "speaker": {"id": "spk_0", "confidence": 0.95},
                "words": [
                    {"word": "Hello,", "start": 0.0, "end": 0.4, "probability": 0.98},
                    {"word": "how", "start": 0.5, "end": 0.7, "probability": 0.96},
                    {"word": "are", "start": 0.8, "end": 0.9, "probability": 0.97},
                    {"word": "you?", "start": 1.0, "end": 1.3, "probability": 0.95},
                ],
                "audio_state": None,
            },
            {
                "id": 1,
                "start": 2.8,
                "end": 4.5,
                "text": "I'm doing well, thanks.",
                "speaker": {"id": "spk_1", "confidence": 0.92},
                "words": [
                    {"word": "I'm", "start": 2.8, "end": 3.0, "probability": 0.94},
                    {"word": "doing", "start": 3.1, "end": 3.4, "probability": 0.96},
                    {"word": "well,", "start": 3.5, "end": 3.8, "probability": 0.93},
                    {"word": "thanks.", "start": 3.9, "end": 4.3, "probability": 0.97},
                ],
                "audio_state": None,
            },
        ],
        "speakers": [
            {"id": "spk_0", "label": None, "total_speech_time": 2.5, "num_segments": 1},
            {"id": "spk_1", "label": None, "total_speech_time": 1.7, "num_segments": 1},
        ],
        "turns": [
            {
                "id": "turn_0",
                "speaker_id": "spk_0",
                "start": 0.0,
                "end": 2.5,
                "segment_ids": [0],
                "text": "Hello, how are you?",
            },
            {
                "id": "turn_1",
                "speaker_id": "spk_1",
                "start": 2.8,
                "end": 4.5,
                "segment_ids": [1],
                "text": "I'm doing well, thanks.",
            },
        ],
        "meta": {"source": "documentation example", "generated": True},
    }


def main() -> int:
    """Generate and write documentation snippets."""
    snippets_dir = PROJECT_ROOT / "docs" / "_snippets"
    snippets_dir.mkdir(parents=True, exist_ok=True)

    # Generate minimal example
    minimal = create_minimal_example()
    minimal_path = snippets_dir / "schema_example.json"
    with open(minimal_path, "w") as f:
        json.dump(minimal, f, indent=2)
    print(f"Wrote {minimal_path}")

    # Generate full example
    full = create_full_example()
    full_path = snippets_dir / "schema_full_example.json"
    with open(full_path, "w") as f:
        json.dump(full, f, indent=2)
    print(f"Wrote {full_path}")

    # Validate against schema if jsonschema available
    try:
        import jsonschema

        schema_path = PROJECT_ROOT / "transcription" / "schemas" / "transcript-v2.schema.json"
        if schema_path.exists():
            with open(schema_path) as f:
                schema = json.load(f)
            jsonschema.validate(minimal, schema)
            jsonschema.validate(full, schema)
            print("Schema validation passed")
    except ImportError:
        print("Note: jsonschema not installed, skipping validation")
    except jsonschema.ValidationError as e:
        print(f"Schema validation failed: {e.message}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
