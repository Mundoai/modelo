# LaMusica — XL-Turbo RunPod Worker Dockerfile
# ACE-Step 1.5 XL-Turbo (4B DiT) on CUDA 12.8.1 / PyTorch 2.9.1
#
# Build:  docker build -t ghcr.io/mundoai/lamusica-xl-worker:latest .
# Push:   docker push ghcr.io/mundoai/lamusica-xl-worker:latest
# Auth:   docker login ghcr.io -u USERNAME -p GITHUB_PAT
#
# The image is built and pushed automatically by the GitHub Actions workflow
# (.github/workflows/build-worker.yml) using GITHUB_TOKEN — no Docker Hub
# credentials are required.
#
# Model weights are NOT baked in — they are downloaded on first run
# and cached in a RunPod Network Volume at /runpod-volume/checkpoints.

FROM runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2204

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# ── Install Python dependencies ───────────────────────────────────────────────
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install flash-attention (CUDA only, optional but speeds up attention layers)
RUN pip install flash-attn --no-build-isolation || \
    echo "flash-attn install failed — continuing without it (non-fatal)"

# ── Copy ACE-Step source ──────────────────────────────────────────────────────
COPY ace-step-1.5/ ./ace-step-1.5/

# ── Copy worker handler ───────────────────────────────────────────────────────
COPY handler.py ./handler.py

# ── Model volume mount point ──────────────────────────────────────────────────
RUN mkdir -p /runpod-volume/checkpoints
ENV CHECKPOINTS_DIR=/runpod-volume/checkpoints

# ── Default env (XL-Turbo defaults) ──────────────────────────────────────────
ENV ACESTEP_DIT_MODEL=acestep-v15-xl-turbo
# LM Planner disabled — lyrics come from Groq/DeepInfra/OpenRouter/RunPod LLM endpoint chain
ENV ACESTEP_LM_MODEL=
ENV ACESTEP_CPU_OFFLOAD=false
ENV ACESTEP_AUDIO_FORMAT=mp3
ENV ACESTEP_QUANTIZATION=
ENV LORA_WEIGHT=0.8
ENV PYTHONUNBUFFERED=1
# RunPod init timeout: model download from HuggingFace on first cold start needs time
ENV RUNPOD_INIT_TIMEOUT=600

# ── Entrypoint ────────────────────────────────────────────────────────────────
CMD ["python", "-u", "handler.py"]
