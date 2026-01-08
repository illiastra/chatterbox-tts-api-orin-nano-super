# Chatterbox TTS API — Jetson Orin Nano Super Edition

<p align="center">
  <img src="https://lm17s1uz51.ufs.sh/f/EsgO8cDHBTOU5bjcd6giJaPhnlpTZysr24u6k9WGqwIjNgQo" alt="Chatterbox API TTS header">
</p>

> **Fork of [travisvn/chatterbox-tts-api](https://github.com/travisvn/chatterbox-tts-api)** — optimized for NVIDIA Jetson Orin Nano Super's 8GB unified memory architecture.

This fork adds comprehensive memory optimizations that enable the full Chatterbox TTS API to run efficiently on Jetson Orin Nano Super and similar memory-constrained edge devices.

## What's Different in This Fork

| Optimization | Description | Impact |
|--------------|-------------|--------|
| **Low Memory Mode** | Loads model piece-by-piece with GC between moves | Avoids 2x memory spike during loading |
| **FP16 Precision** | Half-precision inference (except tokenizer) | ~49% memory reduction (3.5GB → 1.8GB) |
| **Unified Memory Aware** | Jetson-specific loading strategy for shared CPU/GPU RAM | Optimized for Tegra architecture |
| **Auto Jetson Detection** | Detects Jetson hardware via `/etc/nv_tegra_release` | Enables optimizations automatically |
| **Aggressive GC** | `malloc_trim()` + triple garbage collection | Forces OS memory reclamation |
| **CUDA Allocator Tuning** | `max_split_size_mb`, `garbage_collection_threshold` | Reduces memory fragmentation |
| **L4T Docker Image** | Uses `dustynv/l4t-pytorch:r36.4.0` base | Native ARM64 with Jetson-optimized PyTorch |
| **Conservative Defaults** | Single concurrent job, smaller chunks, English model | Fits comfortably in 8GB |

### Performance on Jetson Orin Nano Super

- **Inference Speed**: ~7-9 iterations/second
- **GPU Memory**: ~1.8GB (FP16 mode)
- **Power Draw**: ~15-25W
- **First Audio Latency**: ~2-3 seconds for short text

## Quick Start (Jetson)

### Docker (Recommended)

```bash
git clone https://github.com/illiastra/chatterbox-tts-api-orin-nano-super
cd chatterbox-tts-api-orin-nano-super

# Use Jetson-optimized environment
cp .env.example.docker .env

# Start with Jetson-optimized container
docker compose -f docker/docker-compose.jetson.yml up -d

# Watch logs (first startup downloads the model)
docker logs chatterbox-tts-orin -f

# Test the API
curl -X POST http://localhost:4123/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello from Jetson!"}' \
  --output test.wav
```

### Local Installation

```bash
git clone https://github.com/illiastra/chatterbox-tts-api-orin-nano-super
cd chatterbox-tts-api-orin-nano-super

# Use system Python with Jetson's pre-installed PyTorch
pip3 install -r requirements.txt --no-deps torch torchvision torchaudio

cp .env.example .env

# Enable Jetson optimizations
echo "LOW_MEMORY_MODE=true" >> .env
echo "USE_FP16=true" >> .env
echo "USE_MULTILINGUAL_MODEL=false" >> .env

python3 main.py
```

For detailed Jetson setup instructions, troubleshooting, and power optimization tips, see **[ORIN_NANO_SETUP.md](ORIN_NANO_SETUP.md)**.

## Jetson-Specific Configuration

These environment variables enable the memory optimizations:

| Variable | Jetson Default | Description |
|----------|----------------|-------------|
| `LOW_MEMORY_MODE` | `true` | Enable piece-by-piece model loading |
| `USE_FP16` | `true` | Use half-precision inference |
| `USE_MULTILINGUAL_MODEL` | `false` | Use smaller English-only model |
| `WARMUP_RUNS` | `1` | Fewer warmup runs to save memory |
| `LONG_TEXT_MAX_CONCURRENT_JOBS` | `1` | Single job to prevent memory exhaustion |
| `MAX_CHUNK_LENGTH` | `280` | Smaller text chunks |
| `PYTORCH_CUDA_ALLOC_CONF` | `max_split_size_mb:32,...` | CUDA allocator tuning |

## Monitoring Memory

```bash
# Inside container or on host
tegrastats  # Jetson-specific monitoring

# Or install jtop for a nice TUI
pip install jetson-stats
jtop

# API memory endpoint
curl http://localhost:4123/memory
```

---

## Full Feature Set

This fork maintains all features from the upstream project:

- **OpenAI-Compatible API** — Drop-in replacement for OpenAI's TTS API
- **Voice Cloning** — Use your own voice samples
- **Voice Library Management** — Upload, manage, and reuse custom voices
- **Real-time Streaming** — SSE and raw audio streaming
- **Long Text Synthesis** — Background processing for long-form content
- **React Frontend** — Web UI at port 4321 (with `--profile frontend`)
- **22 Language Support** — When `USE_MULTILINGUAL_MODEL=true` (requires more memory)

## API Usage

### Basic Text-to-Speech

```bash
curl -X POST http://localhost:4123/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Your text here"}' \
  --output speech.wav
```

### With Parameters

```bash
curl -X POST http://localhost:4123/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Expressive speech!", "exaggeration": 0.8, "cfg_weight": 0.5}' \
  --output expressive.wav
```

### Custom Voice Upload

```bash
curl -X POST http://localhost:4123/v1/audio/speech/upload \
  -F "input=Hello with my voice!" \
  -F "voice_file=@my_voice.mp3" \
  --output custom.wav
```

### Voice Library

```bash
# Upload voice to library
curl -X POST http://localhost:4123/voices \
  -F "voice_file=@my_voice.wav" \
  -F "voice_name=my-voice"

# Use by name
curl -X POST http://localhost:4123/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello!", "voice": "my-voice"}' \
  --output output.wav

# List voices
curl http://localhost:4123/voices
```

### Streaming

```bash
# Raw audio streaming
curl -X POST http://localhost:4123/v1/audio/speech/stream \
  -H "Content-Type: application/json" \
  -d '{"input": "This streams in real-time!"}' \
  --output streaming.wav

# SSE streaming (OpenAI compatible)
curl -X POST http://localhost:4123/v1/audio/speech \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{"input": "SSE streaming!", "stream_format": "sse"}' \
  --no-buffer
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/audio/speech` | POST | Generate speech (OpenAI compatible) |
| `/v1/audio/speech/upload` | POST | Generate with voice upload |
| `/v1/audio/speech/stream` | POST | Streaming generation |
| `/voices` | GET/POST | Voice library management |
| `/languages` | GET | Supported languages (if multilingual enabled) |
| `/health` | GET | Health check |
| `/memory` | GET | Memory status |
| `/status` | GET | Processing status |
| `/config` | GET | Current configuration |
| `/docs` | GET | Interactive API docs |

## Parameters

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| `exaggeration` | 0.25-2.0 | 0.5 | Emotion intensity |
| `cfg_weight` | 0.0-1.0 | 0.5 | Pace control |
| `temperature` | 0.05-5.0 | 0.8 | Randomness |

## Running with Frontend

```bash
# Docker with frontend
docker compose -f docker/docker-compose.jetson.yml --profile frontend up -d

# Access:
# - API: http://localhost:4123
# - Web UI: http://localhost:4321
```

## Other Deployment Options

This fork also includes all upstream Docker configurations for non-Jetson hardware:

```bash
# Standard GPU (discrete NVIDIA)
docker compose -f docker/docker-compose.gpu.yml up -d

# CPU only
docker compose -f docker/docker-compose.cpu.yml up -d

# Blackwell (50XX) GPUs
docker compose -f docker/docker-compose.blackwell.yml up -d
```

## Troubleshooting (Jetson)

### Out of Memory During Model Load

```bash
# Ensure low memory mode is enabled
echo "LOW_MEMORY_MODE=true" >> .env
echo "USE_FP16=true" >> .env

# Use English-only model
echo "USE_MULTILINGUAL_MODEL=false" >> .env

# Restart
docker compose -f docker/docker-compose.jetson.yml down
docker compose -f docker/docker-compose.jetson.yml up -d
```

### Slow First Inference

This is normal — cuDNN benchmarking runs on first inference. Subsequent requests are faster.

### Model Download Fails

```bash
# Clear cache and retry
docker volume rm chatterbox-tts-api-orin-nano-super_chatterbox-models
docker compose -f docker/docker-compose.jetson.yml up -d
```

### Check GPU Memory

```bash
tegrastats | grep -E "RAM|GR3D"
# Or use the memory endpoint
curl http://localhost:4123/memory | jq
```

See **[ORIN_NANO_SETUP.md](ORIN_NANO_SETUP.md)** for comprehensive troubleshooting.

## Documentation

- **[ORIN_NANO_SETUP.md](ORIN_NANO_SETUP.md)** — Jetson-specific setup guide
- **[docs/API_README.md](docs/API_README.md)** — Full API documentation
- **[docs/STREAMING_API.md](docs/STREAMING_API.md)** — Streaming documentation
- **[docs/VOICE_LIBRARY_MANAGEMENT.md](docs/VOICE_LIBRARY_MANAGEMENT.md)** — Voice management
- **[docs/MULTILINGUAL.md](docs/MULTILINGUAL.md)** — Multilingual support (requires more memory)

## Credits

This project is a fork of **[travisvn/chatterbox-tts-api](https://github.com/travisvn/chatterbox-tts-api)**, which provides the FastAPI wrapper around [Resemble AI's Chatterbox TTS](https://github.com/resemble-ai/chatterbox).

**Upstream features maintained:**
- OpenAI-compatible API
- Voice cloning and library management
- React frontend
- Multilingual support
- SSE streaming
- Long text synthesis

**This fork adds:**
- NVIDIA Jetson Orin Nano Super support
- Low memory mode with unified memory awareness
- FP16 inference
- Jetson-optimized Docker configuration
- Aggressive memory management

## License

MIT License — see [LICENSE](LICENSE)

## Support

- **Jetson Issues**: Open an issue on [this repo](https://github.com/illiastra/chatterbox-tts-api-orin-nano-super/issues)
- **General Chatterbox TTS API**: See [upstream repo](https://github.com/travisvn/chatterbox-tts-api)
- **Chatterbox TTS Engine**: See [Resemble AI's repo](https://github.com/resemble-ai/chatterbox)
