import sys
import toml

with open('pyproject.toml', 'r') as f:
    content = f.read()

content = content.replace(
    'packages = ["transcription", "transcription.historian", "transcription.historian.analyzers", "transcription.integrations", "scripts", "integrations", "slower_whisper"]',
    'packages = ["transcription", "transcription.historian", "transcription.historian.analyzers", "transcription.integrations", "scripts", "integrations", "slower_whisper", "transcription.store"]'
)

with open('pyproject.toml', 'w') as f:
    f.write(content)
