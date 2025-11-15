"""
Convenience entrypoint script.

Usage (from the directory containing this file):

    python transcribe_pipeline.py

or with options, for example:

    python transcribe_pipeline.py --root C:\transcription_toolkit --model large-v3 --language en
"""

from transcription.cli import main

if __name__ == "__main__":
    main()
