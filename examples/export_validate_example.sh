#!/bin/sh
# Run with slower-whisper on PATH. No audio or model download is needed.
set -eu
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

cat > "$work/transcript.json" <<'JSON'
{
  "schema_version": 2,
  "file": "hello.wav",
  "language": "en",
  "segments": [{"id": 0, "start": 0.0, "end": 1.0, "text": "Hello!"}]
}
JSON

# Both commands should exit 0. The exported subtitle contains one one-second cue.
slower-whisper validate "$work/transcript.json"
slower-whisper export "$work/transcript.json" --format vtt --output "$work/hello.vtt"
cat "$work/hello.vtt"

# Remove a required property while keeping the input syntactically valid JSON.
sed '/"language":/d' "$work/transcript.json" > "$work/invalid.json"
status=0
slower-whisper validate "$work/invalid.json" || status=$?
echo "Invalid transcript validation exited $status (expected 1)."
test "$status" -eq 1
