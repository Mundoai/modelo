# LaMusica — XL-Turbo RunPod Worker Dockerfile
# ACE-Step 1.5 XL-Turbo (4B DiT) on CUDA 12.8.1 / PyTorch 2.9.1
#
# Model weights are BAKED IN during build — zero cold-start download.
# Image size ~15GB (base ~5GB + deps ~2GB + models ~8GB).

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

# ── Bake model weights into image (zero cold-start download) ─────────────────
# Downloads: vae (~500MB), Qwen3-Embedding-0.6B (~1.2GB) from main repo
#            acestep-v15-xl-turbo (~8GB) from sub-model repo
# No LM Planner — lyrics come from external LLM chain
ENV HF_HUB_ENABLE_HF_TRANSFER=1
RUN pip install --no-cache-dir hf_transfer && \
    python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('ACE-Step/Ace-Step1.5', local_dir='/app/ace-step-1.5/checkpoints', \
    local_dir_use_symlinks=False, allow_patterns=['vae/**','Qwen3-Embedding-0.6B/**']); \
snapshot_download('ACE-Step/acestep-v15-xl-turbo', local_dir='/app/ace-step-1.5/checkpoints/acestep-v15-xl-turbo', \
    local_dir_use_symlinks=False); \
print('Models baked successfully')"

# ── Copy worker handler ───────────────────────────────────────────────────────
COPY handler.py ./handler.py

# ── Default env (XL-Turbo defaults) ──────────────────────────────────────────
# CHECKPOINTS_DIR points to baked-in weights inside the image
ENV CHECKPOINTS_DIR=/app/ace-step-1.5/checkpoints
ENV ACESTEP_DIT_MODEL=acestep-v15-xl-turbo
ENV ACESTEP_LM_MODEL=
ENV ACESTEP_CPU_OFFLOAD=false
ENV ACESTEP_AUDIO_FORMAT=mp3
ENV ACESTEP_QUANTIZATION=
ENV LORA_WEIGHT=0.8
ENV PYTHONUNBUFFERED=1

# ── Entrypoint ────────────────────────────────────────────────────────────────
CMD ["python", "-u", "handler.py"]
