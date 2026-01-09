# Slower-Whisper REST API Service

This document describes the FastAPI-based REST API service for slower-whisper, enabling transcription and audio enrichment via HTTP endpoints.

## Quick Start

### Installation

```bash
# Install API dependencies
uv sync --extra api

# For full features (transcription + enrichment)
uv sync --extra api --extra full
```

### Running Locally

```bash
# Development mode (with auto-reload)
uv run uvicorn transcription.service:app --reload --host 0.0.0.0 --port 8000

# Production mode (4 workers)
uv run uvicorn transcription.service:app --host 0.0.0.0 --port 8000 --workers 4
```

### Running with Docker

```bash
# Build the API image
docker build -f Dockerfile.api -t slower-whisper:api .

# Run the service
docker run --rm -p 8000:8000 slower-whisper:api

# Or use Docker Compose
docker-compose -f docker-compose.api.yml up -d
```

## API Endpoints

### Health Check

**GET** `/health`

Check if the service is running and responsive.

```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "service": "slower-whisper-api",
  "version": "1.0.0",
  "schema_version": "2"
}
```

### Transcribe Audio

**POST** `/transcribe`

Upload an audio file and receive a transcription in JSON format.

**Parameters:**
- `audio` (file, required): Audio file to transcribe (mp3, wav, m4a, etc.)
- `model` (query, optional): Whisper model size (default: `large-v3`)
  - Options: `tiny`, `base`, `small`, `medium`, `large-v3`
- `language` (query, optional): Language code (default: `null` for auto-detect)
  - Examples: `en`, `es`, `fr`, `de`, `zh`
- `device` (query, optional): Device for inference (default: `cpu`)
  - Options: `cpu`, `cuda`
- `compute_type` (query, optional): Precision override (default: auto — `float16` on CUDA, `int8` on CPU)
  - Options: `float16`, `float32`, `int8`
- `task` (query, optional): Task type (default: `transcribe`)
  - Options: `transcribe`, `translate` (to English)
- `word_timestamps` (query, optional): Enable word-level timestamps (default: `false`)
  - When `true`, each segment includes a `words` array with per-word timing
- `enable_diarization` (query, optional): Run speaker diarization (default: `false`)
- `diarization_device` (query, optional): Device for diarization (default: `auto`)
  - Options: `cpu`, `cuda`, `auto`
- `min_speakers` / `max_speakers` (query, optional): Speaker count hints for diarization
- `overlap_threshold` (query, optional): Minimum overlap ratio (0.0–1.0) to assign a speaker (default: `0.3`)

**ASR runtime behavior and metadata:**
- If the requested model/device fails (e.g., CUDA not available), the service retries on CPU with a safer `compute_type` before falling back to a lightweight dummy model.
- Response metadata includes the actual ASR runtime used (`asr_backend`, `asr_device`, `asr_compute_type`) plus any load warnings or fallback reasons.

**Example:**
```bash
curl -X POST -F "audio=@interview.mp3" \
  "http://localhost:8000/transcribe?model=large-v3&language=en&device=cpu"

# With word-level timestamps
curl -X POST -F "audio=@interview.mp3" \
  "http://localhost:8000/transcribe?model=large-v3&word_timestamps=true"

# With diarization enabled
curl -X POST -F "audio=@meeting.wav" \
  "http://localhost:8000/transcribe?enable_diarization=true&min_speakers=2&max_speakers=4"
```

**Response:**
```json
{
  "schema_version": 2,
  "file": "uploaded_audio.mp3",
  "language": "en",
  "meta": {
    "generated_at": "2025-11-15T12:34:56Z",
    "model_name": "large-v3",
    "device": "cpu",
    "compute_type": "int8",
    "asr_backend": "faster-whisper",
    "asr_device": "cpu",
    "asr_compute_type": "int8",
    "asr_model_load_warnings": [
      "cuda (float16) load failed: CUDA unavailable"
    ]
  },
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 4.2,
      "text": "Hello, this is a transcription example.",
      "speaker": null,
      "tone": null,
      "audio_state": null
    }
  ]
}
```

**Response with word_timestamps=true:**
```json
{
  "schema_version": 2,
  "file": "uploaded_audio.mp3",
  "language": "en",
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 4.2,
      "text": "Hello, this is a transcription example.",
      "words": [
        {"word": "Hello,", "start": 0.0, "end": 0.5, "probability": 0.98},
        {"word": "this", "start": 0.5, "end": 0.7, "probability": 0.99},
        {"word": "is", "start": 0.7, "end": 0.9, "probability": 0.97},
        {"word": "a", "start": 0.9, "end": 1.0, "probability": 0.95},
        {"word": "transcription", "start": 1.0, "end": 1.8, "probability": 0.92},
        {"word": "example.", "start": 1.8, "end": 2.3, "probability": 0.96}
      ]
    }
  ]
}
```

### Enrich Transcript

**POST** `/enrich`

Upload a transcript JSON and corresponding audio WAV to add prosodic and emotional features.

**Parameters:**
- `transcript` (file, required): Transcript JSON file (output from `/transcribe`)
- `audio` (file, required): Audio WAV file (16kHz mono, matching the transcript)
- `enable_prosody` (query, optional): Extract prosodic features (default: `true`)
- `enable_emotion` (query, optional): Extract dimensional emotion features (default: `true`)
- `enable_categorical_emotion` (query, optional): Extract categorical emotions (default: `false`)
- `device` (query, optional): Device for emotion models (default: `cpu`)
  - Options: `cpu`, `cuda`

**Example:**
```bash
curl -X POST \
  -F "transcript=@transcript.json" \
  -F "audio=@audio.wav" \
  "http://localhost:8000/enrich?enable_prosody=true&enable_emotion=true&device=cpu"
```

**Response:**
```json
{
  "schema_version": 2,
  "file": "audio.wav",
  "language": "en",
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 4.2,
      "text": "Hello, this is a transcription example.",
      "audio_state": {
        "prosody": {
          "pitch": {"level": "high", "mean_hz": 245.3},
          "energy": {"level": "loud", "db_rms": -8.2},
          "rate": {"level": "fast", "syllables_per_sec": 6.3}
        },
        "emotion": {
          "valence": {"level": "positive", "score": 0.72},
          "arousal": {"level": "high", "score": 0.68}
        },
        "rendering": "[audio: high pitch, loud volume, fast speech]"
      }
    }
  ]
}
```

## Interactive Documentation

The API includes auto-generated interactive documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI Schema**: http://localhost:8000/openapi.json

## Docker Deployment

### Basic Deployment

```bash
# Build the image
docker build -f Dockerfile.api -t slower-whisper:api .

# Run the service
docker run --rm -p 8000:8000 slower-whisper:api
```

### With GPU Support

```bash
# Requires nvidia-docker
docker run --rm --gpus all -p 8000:8000 slower-whisper:api \
  uvicorn transcription.service:app --host 0.0.0.0 --port 8000
```

### Docker Compose

```bash
# Start the service with Docker Compose
docker-compose -f docker-compose.api.yml up -d

# View logs
docker-compose -f docker-compose.api.yml logs -f

# Stop the service
docker-compose -f docker-compose.api.yml down
```

### Persistent Model Cache

To avoid re-downloading models on container restart:

```bash
docker run --rm -p 8000:8000 \
  -v ~/.cache/huggingface:/home/appuser/.cache/huggingface \
  slower-whisper:api
```

## Configuration

### Environment Variables

You can configure the service using environment variables:

```bash
# Transcription configuration
export SLOWER_WHISPER_MODEL=large-v3
export SLOWER_WHISPER_DEVICE=cuda
export SLOWER_WHISPER_LANGUAGE=en

# Enrichment configuration
export SLOWER_WHISPER_ENRICH_DEVICE=cuda

# Uvicorn configuration
export UVICORN_WORKERS=8
export UVICORN_HOST=0.0.0.0
export UVICORN_PORT=8000
```

### Uvicorn Options

You can customize the Uvicorn server when running the container:

```bash
docker run --rm -p 8000:8000 slower-whisper:api \
  --host 0.0.0.0 --port 8000 --workers 8 --timeout-keep-alive 120
```

Common options:
- `--workers N`: Number of worker processes (default: 4)
- `--timeout-keep-alive N`: Keep-alive timeout in seconds
- `--limit-concurrency N`: Maximum concurrent connections
- `--log-level LEVEL`: Logging level (debug, info, warning, error)

## Performance Considerations

### CPU vs GPU

- **CPU mode**: Slower but works everywhere
  - Transcription: ~1-2x realtime (1 minute audio = 1-2 minutes processing)
  - Emotion models: ~5-10x realtime
- **GPU mode** (CUDA): Much faster, requires NVIDIA GPU
  - Transcription: ~10-20x realtime
  - Emotion models: ~50-100x realtime

### Worker Configuration

- **CPU**: Use `--workers N` where N = number of CPU cores
- **GPU**: Use `--workers 1` or `--workers 2` (GPU is not shared across workers)
- For high-concurrency workloads, consider using multiple workers with CPU

### Model Caching

Models are downloaded on first use and cached:
- Whisper models: ~150MB (base) to ~3GB (large-v3)
- Emotion models: ~4GB total

Use persistent volumes to avoid re-downloading:
```yaml
volumes:
  - model-cache:/home/appuser/.cache/huggingface
```

## Error Handling

The API returns appropriate HTTP status codes:

- `200`: Success
- `400`: Invalid request (bad configuration, unsupported format, etc.)
- `422`: Validation error (missing required parameters)
- `500`: Internal server error (transcription/enrichment failed)

**Example error response:**
```json
{
  "detail": "Invalid device 'gpu'. Must be 'cuda' or 'cpu'."
}
```

## Security Considerations

### Authentication

The basic API does not include authentication. For production deployments, consider:

1. **Nginx reverse proxy** with HTTP Basic Auth:
   ```nginx
   location / {
       auth_basic "Slower-Whisper API";
       auth_basic_user_file /etc/nginx/.htpasswd;
       proxy_pass http://localhost:8000;
   }
   ```

2. **API Gateway** (AWS API Gateway, Kong, etc.) with OAuth2/JWT

3. **VPN/Network isolation**: Deploy within a private network

### Rate Limiting

For production, add rate limiting to prevent abuse:

1. **Nginx rate limiting**:
   ```nginx
   limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
   limit_req zone=api burst=20;
   ```

2. **Application-level**: Use `slowapi` middleware:
   ```python
   from slowapi import Limiter
   limiter = Limiter(key_func=get_remote_address)
   app.state.limiter = limiter
   ```

### File Upload Limits

The API accepts file uploads. Consider setting limits:

```python
# In service.py, add middleware:
app.add_middleware(
    RequestSizeLimitMiddleware,
    max_request_size=100 * 1024 * 1024  # 100MB
)
```

Or configure Nginx:
```nginx
client_max_body_size 100M;
```

## Development

### Running in Development Mode

```bash
# Install dev dependencies
uv sync --extra dev --extra api --extra full

# Run with auto-reload
uv run uvicorn transcription.service:app --reload --log-level debug
```

### Testing the API

```bash
# Health check
curl http://localhost:8000/health

# Transcribe a sample file
curl -X POST -F "audio=@examples/sample.mp3" \
  "http://localhost:8000/transcribe?model=base&device=cpu"

# Test with httpie (more readable)
pip install httpie
http POST localhost:8000/transcribe audio@examples/sample.mp3 model==base device==cpu
```

### Debugging

Enable debug logging:
```bash
uv run uvicorn transcription.service:app --log-level debug
```

Check Docker logs:
```bash
docker logs -f slower-whisper-api
```

## Example Client Code

### Python

```python
import requests

# Transcribe
with open("interview.mp3", "rb") as f:
    response = requests.post(
        "http://localhost:8000/transcribe",
        files={"audio": f},
        params={"model": "large-v3", "language": "en"},
    )
    transcript = response.json()
    print(transcript["segments"][0]["text"])

# Enrich
with open("transcript.json", "rb") as t, open("audio.wav", "rb") as a:
    response = requests.post(
        "http://localhost:8000/enrich",
        files={"transcript": t, "audio": a},
        params={"enable_prosody": True, "enable_emotion": True},
    )
    enriched = response.json()
    print(enriched["segments"][0]["audio_state"]["rendering"])
```

### JavaScript/Node.js

```javascript
const FormData = require('form-data');
const fs = require('fs');
const axios = require('axios');

async function transcribe() {
  const form = new FormData();
  form.append('audio', fs.createReadStream('interview.mp3'));

  const response = await axios.post('http://localhost:8000/transcribe', form, {
    params: { model: 'large-v3', language: 'en' },
    headers: form.getHeaders(),
  });

  console.log(response.data.segments[0].text);
}
```

### cURL

```bash
# Transcribe
curl -X POST -F "audio=@interview.mp3" \
  "http://localhost:8000/transcribe?model=large-v3&language=en" \
  -o transcript.json

# Enrich
curl -X POST \
  -F "transcript=@transcript.json" \
  -F "audio=@audio.wav" \
  "http://localhost:8000/enrich?enable_prosody=true" \
  -o enriched.json
```

## Troubleshooting

### Port Already in Use

```bash
# Find process using port 8000
lsof -i :8000

# Kill the process
kill -9 <PID>

# Or use a different port
docker run -p 8001:8000 slower-whisper:api --port 8000
```

### Out of Memory

Reduce workers or use smaller models:
```bash
docker run -p 8000:8000 slower-whisper:api --workers 2
```

Or increase Docker memory limit:
```yaml
deploy:
  resources:
    limits:
      memory: 16G
```

### CUDA Not Available

Ensure nvidia-docker is installed:
```bash
# Check GPU availability
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

# Run with GPU
docker run --rm --gpus all -p 8000:8000 slower-whisper:api
```

### Models Not Downloading

Check internet connection and HuggingFace availability:
```bash
# Test model download
docker run --rm slower-whisper:api python -c "
from transformers import AutoModel
AutoModel.from_pretrained('audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim')
"
```

## License

This API service is part of slower-whisper and is licensed under the Apache 2.0 License.
