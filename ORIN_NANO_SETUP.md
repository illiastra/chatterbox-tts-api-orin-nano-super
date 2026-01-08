# Chatterbox TTS API - Orin Nano Super Setup Guide

This guide explains how to deploy the Chatterbox TTS API with FP16 optimization on your NVIDIA Orin Nano Super (8GB).

## Why Orin Nano Super?

| Feature | GTX 1070 | Orin Nano Super |
|---------|----------|-----------------|
| Architecture | Pascal | Ampere |
| Tensor Cores | No | Yes |
| FP16 Performance | Emulated (slower) | Native (faster) |
| Memory | 8GB VRAM | 8GB Unified |
| Power | ~150W | ~25W |

The Orin's Ampere tensor cores will accelerate FP16 inference, potentially making it **faster** than the GTX 1070 while using a fraction of the power.

---

## Prerequisites

### On your Orin Nano Super:

1. **JetPack 6.x installed** (includes CUDA, cuDNN, TensorRT)
   ```bash
   # Check JetPack version
   cat /etc/nv_tegra_release

   # Check CUDA version
   nvcc --version
   ```

2. **Docker with NVIDIA runtime**
   ```bash
   # Verify Docker is installed
   docker --version

   # Verify NVIDIA runtime
   docker info | grep -i runtime
   ```

3. **Sufficient storage** (~10GB for Docker image + models)

4. **Swap space configured** (CRITICAL for model loading)
   ```bash
   # Check current swap
   free -h

   # If swap is less than 8GB, increase it:
   sudo swapoff -a
   sudo fallocate -l 8G /swapfile
   sudo chmod 600 /swapfile
   sudo mkswap /swapfile
   sudo swapon /swapfile

   # Make permanent (add to /etc/fstab)
   echo '/swapfile swap swap defaults 0 0' | sudo tee -a /etc/fstab
   ```

   > **Note:** The model loads on CPU first (in low-memory mode), then converts to FP16 before moving to GPU. This requires ~6-8GB of CPU RAM during loading.

---

## Step 1: Transfer Files to Orin

### Option A: Using SCP (from your Windows machine)

```bash
# From Windows PowerShell or Git Bash
# Replace <orin-ip> with your Orin's IP address
# Replace <username> with your Orin username

scp -r "C:\Docker Containers\chatterbox-tts-api" <username>@<orin-ip>:~/chatterbox-tts-api
```

### Option B: Using Git (if you've committed changes)

```bash
# On the Orin
git clone <your-repo-url> ~/chatterbox-tts-api
```

### Option C: Using rsync (recommended for large transfers)

```bash
# From Windows WSL or Linux
rsync -avz --progress "/mnt/c/Docker Containers/chatterbox-tts-api/" <username>@<orin-ip>:~/chatterbox-tts-api/
```

---

## Step 2: Create Jetson-Compatible Dockerfile

The Orin uses ARM64 architecture and requires NVIDIA L4T base images. Create this new Dockerfile:

```bash
# On the Orin, create the Jetson Dockerfile
cd ~/chatterbox-tts-api
nano docker/Dockerfile.jetson
```

Paste the following content:

```dockerfile
# Dockerfile.jetson - For NVIDIA Jetson Orin Nano Super
# Uses L4T (Linux for Tegra) base image with PyTorch pre-installed

FROM nvcr.io/nvidia/l4t-pytorch:r36.2.0-pth2.1-py3

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Install pip and upgrade
RUN pip3 install --upgrade pip

# Set working directory
WORKDIR /app

# Install Python dependencies
RUN pip3 install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    python-dotenv \
    python-multipart \
    requests \
    psutil \
    pydub \
    sse-starlette \
    torchaudio

# Install chatterbox-tts
RUN pip3 install --no-cache-dir git+https://github.com/travisvn/chatterbox-multilingual.git@exp

# Copy application code
COPY app/ ./app/
COPY main.py ./

# Copy voice sample
COPY voice-sample.mp3 ./voice-sample.mp3

# Create directories
RUN mkdir -p /cache /voices /data/long_text_jobs

# Set default environment variables
ENV PORT=4123
ENV HOST=0.0.0.0
ENV DEVICE=cuda
ENV USE_FP16=true
ENV TORCH_CUDNN_BENCHMARK=true
ENV WARMUP_ON_STARTUP=true
ENV WARMUP_RUNS=3
ENV MODEL_CACHE_DIR=/cache
ENV VOICE_LIBRARY_DIR=/voices
ENV VOICE_SAMPLE_PATH=/app/voice-sample.mp3

# Long text settings
ENV LONG_TEXT_DATA_DIR=/data/long_text_jobs

# Expose port
EXPOSE ${PORT}

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5m --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Run the application
CMD ["python3", "main.py"]
```

---

## Step 3: Create Jetson Docker Compose File

The Jetson Docker Compose file is already provided at `docker/docker-compose.jetson.yml`. It includes:
- The main TTS API service with Jetson-specific optimizations
- Optional frontend service (activated via profiles)

---

## Step 4: Configure Environment

```bash
# Copy and edit the .env file
cp .env .env.backup
nano .env
```

Ensure these settings are configured:

```bash
# Key settings for Orin Nano Super
USE_FP16=true
TORCH_CUDNN_BENCHMARK=true
WARMUP_ON_STARTUP=true
WARMUP_RUNS=3
DEVICE=cuda

# Memory-optimized settings (Orin has 8GB unified memory)
MEMORY_CLEANUP_INTERVAL=3
CUDA_CACHE_CLEAR_INTERVAL=2
ENABLE_MEMORY_MONITORING=true

# Use standard (English-only) model to save memory
USE_MULTILINGUAL_MODEL=false
```

---

## Step 5: Build and Run

```bash
cd ~/chatterbox-tts-api

# Build the Docker image (this will take a while on first run)
docker compose -f docker/docker-compose.jetson.yml build

# Start the API service only
docker compose -f docker/docker-compose.jetson.yml up -d

# OR: Start with the frontend UI (port 4321)
docker compose -f docker/docker-compose.jetson.yml --profile frontend up -d

# Watch the logs
docker logs -f chatterbox-tts-orin
```

**Frontend Access:**
- API: `http://<orin-ip>:4123`
- Frontend UI: `http://<orin-ip>:4321` (when using `--profile frontend`)

---

## Step 6: Verify Installation

### Check health endpoint:
```bash
curl http://localhost:4123/health
```

Expected response should show:
- `"model_loaded": true`
- `"device": "cuda"`
- `"gpu_memory_allocated_mb"` around 1800-2000 MB

### Check for FP16 in logs:
```bash
docker logs chatterbox-tts-orin 2>&1 | grep -E "(FP16|GPU memory)"
```

You should see:
```
Converting model to FP16 for reduced memory usage...
✓ FP16 conversion complete - GPU memory: ~1775MB allocated
```

### Test TTS generation:
```bash
curl -X POST http://localhost:4123/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello from the Orin Nano Super!", "voice": "alloy"}' \
  -o test.wav

# Play the audio (if you have speakers connected)
aplay test.wav
```

---

## Step 7: Monitor Performance

### Check inference speed:
```bash
# Watch logs during TTS generation
docker logs -f chatterbox-tts-orin 2>&1 | grep "it/s"
```

**Expected speeds:**
- GTX 1070 (FP16): ~13-14 it/s
- Orin Nano Super (FP16): Should be similar or faster due to tensor cores

### Monitor memory usage:
```bash
# Check Jetson stats
tegrastats

# Or use jtop (install with: sudo pip3 install jetson-stats)
jtop
```

---

## Troubleshooting

### Issue: NvMapMemAllocInternalTagged error 12 (OOM during model loading)

This error means the system ran out of memory during model loading.

**Solution 1:** Ensure swap is configured (see Prerequisites step 4)
```bash
# Check swap is active
free -h
# Should show 8GB+ swap
```

**Solution 2:** Verify low-memory mode is enabled
```bash
# Check container logs
docker logs chatterbox-tts-orin 2>&1 | grep "Low memory mode"
# Should show: "⚡ Low memory mode: ENABLED"
```

**Solution 3:** Free up system memory before starting
```bash
# Stop other containers
docker stop $(docker ps -q)

# Clear system caches
sudo sync && echo 3 | sudo tee /proc/sys/vm/drop_caches

# Then start the container
docker compose -f docker/docker-compose.jetson.yml up -d
```

### Issue: Out of memory during inference

**Solution:** Reduce concurrent processing:
```bash
# In .env or docker-compose
LONG_TEXT_MAX_CONCURRENT_JOBS=1
MAX_CHUNK_LENGTH=200
```

### Issue: Slow model loading

**Solution:** The first load downloads the model (~3GB). Subsequent loads use cache. Ensure `/cache` volume is persistent.

### Issue: Docker build fails

**Solution:** Check JetPack version compatibility. The Dockerfile uses `dustynv/l4t-pytorch:r36.4.0` which requires JetPack 6.x.

### Issue: CUDA not available

**Solution:** Ensure NVIDIA runtime is default:
```bash
# Check /etc/docker/daemon.json
{
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    },
    "default-runtime": "nvidia"
}

# Restart Docker
sudo systemctl restart docker
```

---

## Network Access

To access from other devices on your network:

```bash
# Find Orin's IP
hostname -I

# Access from another device
curl http://<orin-ip>:4123/health
```

---

## Power Optimization (Optional)

For best performance on Orin Nano Super:

```bash
# Set to maximum performance mode
sudo nvpmodel -m 0

# Or for power-efficient mode (slightly slower)
sudo nvpmodel -m 1

# Check current mode
nvpmodel -q
```

---

## Quick Reference

| Command | Description |
|---------|-------------|
| `docker compose -f docker/docker-compose.jetson.yml up -d` | Start API only |
| `docker compose -f docker/docker-compose.jetson.yml --profile frontend up -d` | Start API + Frontend |
| `docker compose -f docker/docker-compose.jetson.yml down` | Stop the services |
| `docker logs -f chatterbox-tts-orin` | View API logs |
| `docker logs -f chatterbox-tts-frontend-jetson` | View frontend logs |
| `curl http://localhost:4123/health` | Check health |
| `tegrastats` | Monitor Jetson stats |

---

## Expected Results

With FP16 on Orin Nano Super:
- **Model memory:** ~1.8GB (fits easily in 8GB unified memory)
- **Inference speed:** ~13-20 it/s (depending on nvpmodel setting)
- **Power consumption:** ~15-25W (vs ~150W for GTX 1070)

The Orin Nano Super should provide comparable or better TTS performance while using a fraction of the power!
