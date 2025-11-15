"""
Convenience entrypoint script for audio enrichment.

Usage (from the directory containing this file):

    python audio_enrich.py

or with options, for example:

    python audio_enrich.py --root /path/to/project --skip-existing --device cpu
    python audio_enrich.py --file whisper_json/meeting1.json
    python audio_enrich.py --enable-categorical-emotion
"""

from transcription.audio_enrich_cli import main

if __name__ == "__main__":
    main()
