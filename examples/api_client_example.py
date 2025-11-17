#!/usr/bin/env python3
"""
Example client for the slower-whisper FastAPI service.

This script demonstrates how to use the REST API to transcribe and enrich audio files.

Usage:
    # Start the API service first:
    uvicorn transcription.service:app --host 0.0.0.0 --port 8000

    # Then run this client:
    python examples/api_client_example.py

Requirements:
    pip install requests
"""

import sys
from pathlib import Path

try:
    import requests
except ImportError:
    print("Error: requests library not found. Install with: pip install requests")
    sys.exit(1)


class SlowerWhisperClient:
    """Client for the slower-whisper REST API."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize the API client.

        Args:
            base_url: Base URL of the API service (default: http://localhost:8000)
        """
        self.base_url = base_url.rstrip("/")

    def health_check(self) -> dict:
        """
        Check if the API service is healthy.

        Returns:
            Health status dictionary

        Raises:
            requests.exceptions.RequestException: If the service is unreachable
        """
        response = requests.get(f"{self.base_url}/health", timeout=5)
        response.raise_for_status()
        return response.json()

    def transcribe(
        self,
        audio_path: str | Path,
        model: str = "large-v3",
        language: str | None = None,
        device: str = "cpu",
        compute_type: str = "float32",
        task: str = "transcribe",
    ) -> dict:
        """
        Transcribe an audio file.

        Args:
            audio_path: Path to audio file (mp3, wav, m4a, etc.)
            model: Whisper model size (tiny, base, small, medium, large-v3)
            language: Language code (e.g., 'en', 'es') or None for auto-detect
            device: Device to use ('cuda' or 'cpu')
            compute_type: Precision (float16, float32, int8)
            task: Task type ('transcribe' or 'translate')

        Returns:
            Transcript dictionary with segments and metadata

        Raises:
            FileNotFoundError: If audio file doesn't exist
            requests.exceptions.HTTPError: If API request fails
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Prepare request
        with open(audio_path, "rb") as f:
            files = {"audio": (audio_path.name, f, "audio/mpeg")}

            params = {
                "model": model,
                "device": device,
                "compute_type": compute_type,
                "task": task,
            }
            if language:
                params["language"] = language

            # Send request
            response = requests.post(
                f"{self.base_url}/transcribe",
                files=files,
                params=params,
                timeout=300,  # 5 minutes timeout
            )
            response.raise_for_status()

        return response.json()

    def enrich(
        self,
        transcript_path: str | Path,
        audio_path: str | Path,
        enable_prosody: bool = True,
        enable_emotion: bool = True,
        enable_categorical_emotion: bool = False,
        device: str = "cpu",
    ) -> dict:
        """
        Enrich a transcript with audio features.

        Args:
            transcript_path: Path to transcript JSON file
            audio_path: Path to audio WAV file (16kHz mono)
            enable_prosody: Extract prosodic features (pitch, energy, rate)
            enable_emotion: Extract dimensional emotion features (valence, arousal)
            enable_categorical_emotion: Extract categorical emotions (happy, sad, etc.)
            device: Device for emotion models ('cuda' or 'cpu')

        Returns:
            Enriched transcript dictionary with audio_state for each segment

        Raises:
            FileNotFoundError: If transcript or audio file doesn't exist
            requests.exceptions.HTTPError: If API request fails
        """
        transcript_path = Path(transcript_path)
        audio_path = Path(audio_path)

        if not transcript_path.exists():
            raise FileNotFoundError(f"Transcript file not found: {transcript_path}")
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Prepare request
        with open(transcript_path, "rb") as t, open(audio_path, "rb") as a:
            files = {
                "transcript": (transcript_path.name, t, "application/json"),
                "audio": (audio_path.name, a, "audio/wav"),
            }

            params = {
                "enable_prosody": enable_prosody,
                "enable_emotion": enable_emotion,
                "enable_categorical_emotion": enable_categorical_emotion,
                "device": device,
            }

            # Send request
            response = requests.post(
                f"{self.base_url}/enrich",
                files=files,
                params=params,
                timeout=600,  # 10 minutes timeout for emotion models
            )
            response.raise_for_status()

        return response.json()


def main():
    """Example usage of the SlowerWhisperClient."""
    # Initialize client
    client = SlowerWhisperClient(base_url="http://localhost:8000")

    print("=== Slower-Whisper API Client Example ===\n")

    # Step 1: Health check
    print("1. Checking API health...")
    try:
        health = client.health_check()
        print(f"   ✓ Service is healthy: {health['status']}")
        print(f"   Version: {health['version']}")
        print(f"   Schema: v{health['schema_version']}\n")
    except requests.exceptions.RequestException as e:
        print(f"   ✗ Service is not reachable: {e}")
        print("   Make sure the API service is running:")
        print("   uvicorn transcription.service:app --host 0.0.0.0 --port 8000\n")
        sys.exit(1)

    # Step 2: Transcribe audio (example)
    print("2. Transcribing audio (example)...")
    print("   Note: This is a demonstration. Replace with your actual audio file.\n")

    # Example with a hypothetical audio file
    # Uncomment and modify the path to use with a real file:
    #
    # try:
    #     transcript = client.transcribe(
    #         audio_path="path/to/your/audio.mp3",
    #         model="large-v3",
    #         language="en",
    #         device="cpu",
    #     )
    #
    #     print(f"   ✓ Transcription completed")
    #     print(f"   File: {transcript['file_name']}")
    #     print(f"   Language: {transcript['language']}")
    #     print(f"   Segments: {len(transcript['segments'])}")
    #
    #     if transcript['segments']:
    #         print(f"   First segment: \"{transcript['segments'][0]['text']}\"")
    #
    #     # Save transcript
    #     output_path = Path("transcript.json")
    #     output_path.write_text(json.dumps(transcript, indent=2))
    #     print(f"   Saved to: {output_path}\n")
    #
    # except FileNotFoundError as e:
    #     print(f"   ✗ {e}\n")
    # except requests.exceptions.HTTPError as e:
    #     print(f"   ✗ API error: {e}\n")

    # Step 3: Enrich transcript (example)
    print("3. Enriching transcript (example)...")
    print("   Note: This requires a transcript JSON and matching audio WAV.\n")

    # Example with hypothetical files
    # Uncomment and modify the paths to use with real files:
    #
    # try:
    #     enriched = client.enrich(
    #         transcript_path="transcript.json",
    #         audio_path="path/to/audio.wav",
    #         enable_prosody=True,
    #         enable_emotion=True,
    #         device="cpu",
    #     )
    #
    #     print(f"   ✓ Enrichment completed")
    #     print(f"   Segments: {len(enriched['segments'])}")
    #
    #     if enriched['segments'] and enriched['segments'][0].get('audio_state'):
    #         audio_state = enriched['segments'][0]['audio_state']
    #         print(f"   Audio rendering: {audio_state.get('rendering', 'N/A')}")
    #
    #     # Save enriched transcript
    #     output_path = Path("enriched.json")
    #     output_path.write_text(json.dumps(enriched, indent=2))
    #     print(f"   Saved to: {output_path}\n")
    #
    # except FileNotFoundError as e:
    #     print(f"   ✗ {e}\n")
    # except requests.exceptions.HTTPError as e:
    #     print(f"   ✗ API error: {e}\n")

    # Step 4: Usage instructions
    print("=== Usage Instructions ===\n")
    print("To use this client with your own audio files:")
    print()
    print("1. Start the API service:")
    print("   uvicorn transcription.service:app --host 0.0.0.0 --port 8000")
    print()
    print("2. Modify this script to use your audio file:")
    print("   # Uncomment and update the paths in the transcribe() call above")
    print()
    print("3. Run the script:")
    print("   python examples/api_client_example.py")
    print()
    print("Or use curl directly:")
    print('   curl -X POST -F "audio=@your_audio.mp3" \\')
    print('     "http://localhost:8000/transcribe?model=large-v3&language=en"')
    print()


if __name__ == "__main__":
    main()
