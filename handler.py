"""
LaMusica — RunPod Serverless Handler
ACE-Step 1.5 music generation endpoint

Input schema:
{
  "prompt": "salsa, brass, conga, upbeat, Latin, 170 bpm, energetic",
  "lyrics": "[verse]\nTu sonrisa...\n[chorus]\nBailamos...",
  "duration": 60.0,
  "quality": "standard",          // draft|standard|high|studio
  "vocal_language": "es",
  "vocal_gender": "male",         // male|female|both|null
  "vocal_timbre": "carlos",       // voice character name (matches voice_refs on R2)
  "reference_audio_url": "https://...", // presigned R2 URL for 30s vocal reference WAV
  "creativity": 50,               // 0-100 (maps to guidance_scale)
  "bpm": 170,                     // int or null for auto
  "key_scale": "A minor",         // or null for auto
  "time_signature": "4",          // 2|3|4|6 or null for auto
  "seed": -1,                     // -1 for random
  "use_windowed_attention": true,
  "lora_path": null               // optional: S3 path to .safetensors LoRA
}

Output:
{
  "audio_url": "https://s3...",
  "duration": 60.0,
  "format": "wav",
  "seed": 12345,
  "generation_time": 45.2,
  "metadata": { ... }
}
"""

import os
import sys
import time
import uuid
import json
import tempfile
import logging
import concurrent.futures

import runpod

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("lamusica.worker")

# ── ACE-Step path ─────────────────────────────────────────────────────────────
WORKER_DIR = os.path.dirname(os.path.abspath(__file__))
ACESTEP_DIR = os.path.join(WORKER_DIR, "ace-step-1.5")
if os.path.isdir(ACESTEP_DIR):
    sys.path.insert(0, ACESTEP_DIR)

# ── Config from environment ───────────────────────────────────────────────────
CHECKPOINTS_DIR = os.environ.get("CHECKPOINTS_DIR", os.path.join(WORKER_DIR, "checkpoints"))
DIT_MODEL = os.environ.get("ACESTEP_DIT_MODEL", "acestep-v15-turbo")
LM_MODEL = os.environ.get("ACESTEP_LM_MODEL", "acestep-5Hz-lm-4B")  # LM planner for CoT song blueprints
# "auto": enable CPU offload when GPU VRAM < 40GB (prevents OOM on RTX 4090/3090 24GB GPUs)
# "true": always enable | "false": always disable
_CPU_OFFLOAD_ENV = os.environ.get("ACESTEP_CPU_OFFLOAD", "auto").lower()
CPU_OFFLOAD = False  # resolved at init time in _initialize()
AUDIO_FORMAT = os.environ.get("ACESTEP_AUDIO_FORMAT", "mp3")
# INT8 quantization: "int8_weight_only" or "" (empty = bf16, no quantization)
QUANTIZATION = os.environ.get("ACESTEP_QUANTIZATION", "").strip() or None
LORA_PATH = os.environ.get("SPANISH_VOCALS_LORA_PATH", "")
LORA_WEIGHT = float(os.environ.get("LORA_WEIGHT", "0.5"))

# Generation timeout — kills hung generate_music() calls (seconds)
# Set to 0 to disable. Must be < RunPod execution timeout (900s).
GENERATION_TIMEOUT = int(os.environ.get("GENERATION_TIMEOUT", "720"))

# Maximum duration we will ever send to ACE-Step (seconds).
# Prevents runaway jobs that would exceed RunPod's execution window.
MAX_DURATION_SECONDS = int(os.environ.get("MAX_DURATION_SECONDS", "240"))

# S3 config
S3_ENDPOINT = os.environ.get("S3_ENDPOINT_URL")
S3_ACCESS_KEY = os.environ.get("S3_ACCESS_KEY_ID")
S3_SECRET_KEY = os.environ.get("S3_SECRET_ACCESS_KEY")
S3_BUCKET = os.environ.get("S3_BUCKET_NAME", "lamusica-audio")
S3_REGION = os.environ.get("S3_REGION", "us-east-1")

# Quality presets (matches Flask server.py QUALITY_PRESETS)
QUALITY_PRESETS = {
    "draft":    {"steps": 4,  "guidance_scale": 5.0},
    "standard": {"steps": 8,  "guidance_scale": 7.0},
    "high":     {"steps": 16, "guidance_scale": 10.0},
    "studio":   {"steps": 32, "guidance_scale": 15.0},
}

# ── Global model state (warm between RunPod calls) ────────────────────────────
_dit_handler = None
_llm_handler = None
_initialized = False
_init_error = None


def _creativity_to_guidance(creativity: int) -> float:
    """Map creativity 0–100 → guidance_scale 15.0 (constrained) to 3.0 (creative)."""
    creativity = max(0, min(100, creativity))
    return 15.0 - (creativity / 100.0) * 12.0


TIMBRE_PROMPTS = {
    # ── Legacy character keys (kept for backwards compat with existing jobs) ──
    "carlos":    "solo nasal tenor male vocals, distinctive vibrato, Caribbean salsa phrasing, close-mic warm recording, clear mid-range, emotional delivery, no harmony",
    "miguel":    "solo powerful belting tenor male vocals, modern Latin pop projection, studio-polished, bright high range, passionate crescendo, emotional vibrato, no harmony",
    "andres":    "solo deep warm bass-baritone male vocals, Venezuelan resonance, smooth legato delivery, rich low register, romantic crooner, velvety dark tone, no harmony",
    "javier":    "solo raw gritty baritone male vocals, street-corner storytelling, rough authoritative tone, staccato phrasing, urban edge, dry recording, no harmony",
    "frankie":   "solo sweet silky tenor male vocals, romantic salsa crooner, gentle smooth delivery, soft vibrato, intimate close-mic, tender phrasing, no harmony",
    "valentina": "solo powerful dramatic mezzo-soprano female vocals, passionate belting, intense emotional projection, bold Latina diva, dynamic range, live energy, no harmony",
    "sofia":     "solo bright alto female vocals, unique Cuban accent, joyful rhythmic energy, celebratory spirit, warm Caribbean timbre, iconic phrasing, no harmony",
    "camila":    "solo powerful dramatic alto female vocals, intense emotional belting, passionate Latin pop delivery, raw expressive vibrato, heartbreak intensity, no harmony",
    "lucia":     "solo warm deep contralto female vocals, ranchera phrasing, rich velvety tone, soulful slow vibrato, intimate romantic delivery, Mexican folk warmth, no harmony",
    # ── New voice characters (Phase 2) ──
    "tito":      "solo bright powerful tenor male vocals, operatic projection, soaring high notes, dynamic crescendo, bold resonant tone, no harmony",
    "lalo":      "solo sweet melodic tenor male vocals, gentle romantic phrasing, light airy delivery, tender vibrato, soft intimate tone, no harmony",
    "hildemaro": "solo soft velvety tenor male vocals, smooth romantic delivery, gentle whispered phrasing, intimate close-mic warmth, delicate vibrato, no harmony",
    "david":     "solo light airy tenor male vocals, modern pop-crossover delivery, crisp clean articulation, youthful bright tone, studio polish, no harmony",
    # ── Branded "Carácter" styles (genre-neutral vocal modifiers) ──
    "suave":     "smooth legato vocals, gentle silky delivery, flowing phrasing, soft dynamics, polished tone, no harmony",
    "fuego":     "powerful belting vocals, strong emotional projection, intense dynamic range, passionate crescendo, bold attack, no harmony",
    "humo":      "raspy gritty vocals, rough textured tone, raw edge, gravelly resonance, smoky delivery, no harmony",
    "brisa":     "breathy airy vocals, intimate whispered delivery, soft delicate tone, close-mic presence, gentle phrasing, no harmony",
    "cristal":   "bright clear vocals, crisp articulation, luminous high register, pristine tone, sparkling delivery, no harmony",
    "calido":    "warm deep vocals, rich low resonance, velvety smooth tone, honeyed delivery, soothing presence, no harmony",
}


def _build_voice_prompt(prompt: str, vocal_gender: str, vocal_timbre: str = None) -> str:
    """Append vocal timbre descriptor (or gender fallback) to prompt caption."""
    timbre_key = vocal_timbre.strip().lower() if vocal_timbre else ""
    timbre_desc = TIMBRE_PROMPTS.get(timbre_key, "")

    if timbre_desc:
        return f"{prompt}, {timbre_desc}"

    # Fallback: plain gender hint
    if not vocal_gender or vocal_gender == "both":
        return prompt
    hint = "male vocals" if vocal_gender.lower() == "male" else "female vocals"
    if hint.lower() not in prompt.lower():
        return f"{prompt}, {hint}"
    return prompt


def _setup_volume_symlink():
    """
    ACE-Step resolves checkpoints relative to the package root:
      /app/ace-step-1.5/checkpoints/
    But that path lives in ephemeral container storage — re-downloading 7 GB on
    every cold start.  We fix this by symlinking it to the persistent RunPod
    Network Volume (/runpod-volume/checkpoints) before the model is loaded.

    If the package already ships model weights (baked-in image), the symlink is
    skipped and the baked-in weights are used directly.
    """
    package_checkpoints = os.path.join(ACESTEP_DIR, "checkpoints")
    volume_checkpoints  = os.path.join(CHECKPOINTS_DIR, "")   # e.g. /runpod-volume/checkpoints

    # Already a symlink → nothing to do.
    if os.path.islink(package_checkpoints):
        logger.info(f"[setup] Checkpoints symlink already in place: {package_checkpoints}")
        return

    # Baked-in weights detected → use them as-is.
    baked = os.path.exists(os.path.join(package_checkpoints, DIT_MODEL, "config.json"))
    if baked:
        logger.info("[setup] Baked-in model weights detected — skipping volume symlink")
        return

    # Ensure volume dir exists.
    os.makedirs(volume_checkpoints, exist_ok=True)

    if os.path.isdir(package_checkpoints):
        # Copy small config / code files to volume first (if not already there).
        import shutil
        for item in os.listdir(package_checkpoints):
            src = os.path.join(package_checkpoints, item)
            dst = os.path.join(volume_checkpoints, item)
            if os.path.exists(dst):
                continue
            try:
                if os.path.isfile(src):
                    shutil.copy2(src, dst)
                elif os.path.isdir(src):
                    shutil.copytree(src, dst)
            except Exception as copy_err:
                logger.warning(f"[setup] Could not copy {src}: {copy_err}")
        shutil.rmtree(package_checkpoints)

    os.symlink(volume_checkpoints.rstrip("/"), package_checkpoints)
    logger.info(f"[setup] Symlinked {package_checkpoints} → {volume_checkpoints}")


def initialize_model():
    """
    Cold start: load ACE-Step 1.5 model into GPU memory.
    Called once per worker lifecycle. Subsequent jobs reuse the warm model.
    """
    global _dit_handler, _llm_handler, _initialized, _init_error

    if _initialized:
        return

    logger.info("[init] Starting ACE-Step 1.5 initialization...")
    start = time.time()

    try:
        import torch
        from acestep.handler import AceStepHandler
        from acestep.llm_inference import LLMHandler
        from acestep.gpu_config import get_gpu_config, set_global_gpu_config
        from pathlib import Path

        # ── Redirect checkpoints to persistent volume ─────────────────────
        _setup_volume_symlink()

        # ── Device detection ──────────────────────────────────────────────
        global CPU_OFFLOAD
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"[init] Device: {device}")
        if device == "cuda":
            _props = torch.cuda.get_device_properties(0)
            mem_gb = _props.total_memory / 1e9
            mem_mb = _props.total_memory // (1024 ** 2)
            logger.info(f"[init] GPU: {torch.cuda.get_device_name(0)} ({mem_gb:.1f}GB)")
            # Auto CPU offload: enable on GPUs < 40GB to prevent CUDA OOM during
            # VAE reference audio encoding with all models on GPU simultaneously.
            if _CPU_OFFLOAD_ENV == "auto":
                CPU_OFFLOAD = mem_mb < 40_000
                logger.info(
                    f"[init] CPU_OFFLOAD=auto → {'ENABLED' if CPU_OFFLOAD else 'DISABLED'} "
                    f"(GPU VRAM={mem_gb:.1f}GB, threshold=40GB)"
                )
            else:
                CPU_OFFLOAD = _CPU_OFFLOAD_ENV == "true"
                logger.info(f"[init] CPU_OFFLOAD={CPU_OFFLOAD} (from env)")

        # ── Model download check (volume path) ───────────────────────────
        # Check required components: DiT + VAE + Embedding + LM (if enabled).
        # Only downloads the specific LM model needed, skips all others.
        checkpoint_path = Path(CHECKPOINTS_DIR)
        _required = [DIT_MODEL, "vae", "Qwen3-Embedding-0.6B"]
        if LM_MODEL:
            _required.append(LM_MODEL)
        _missing = [c for c in _required if not (checkpoint_path / c).exists()]
        if _missing:
            logger.info(f"[init] Missing components: {_missing} — downloading from HuggingFace...")
            try:
                from huggingface_hub import snapshot_download as _hf_download
                from acestep.model_downloader import SUBMODEL_REGISTRY, MAIN_MODEL_REPO

                # Components from the main repo (vae, embedding, default turbo, default LM)
                _main_components = {"acestep-v15-turbo", "vae", "Qwen3-Embedding-0.6B", "acestep-5Hz-lm-1.7B"}
                _main_missing = [c for c in _missing if c in _main_components]
                _sub_missing = [c for c in _missing if c not in _main_components]

                # Download main repo components (if any are missing)
                if _main_missing:
                    _all_lm = ["acestep-5Hz-lm-0.6B", "acestep-5Hz-lm-1.7B", "acestep-5Hz-lm-4B"]
                    _ignore = ["*.gitattributes"]
                    if LM_MODEL:
                        for lm in _all_lm:
                            if lm != LM_MODEL:
                                _ignore.append(f"{lm}/**")
                    else:
                        _ignore.append("acestep-5Hz-lm-*/**")
                    logger.info(f"[init] Downloading main repo components: {_main_missing}")
                    _hf_download(
                        repo_id=MAIN_MODEL_REPO,
                        local_dir=str(checkpoint_path),
                        local_dir_use_symlinks=False,
                        ignore_patterns=_ignore,
                    )

                # Download sub-model repo components (XL DiT, extra LMs, etc.)
                for sub_name in _sub_missing:
                    if sub_name in SUBMODEL_REGISTRY:
                        sub_repo = SUBMODEL_REGISTRY[sub_name]
                        sub_dest = checkpoint_path / sub_name
                        logger.info(f"[init] Downloading sub-model {sub_name} from {sub_repo}")
                        _hf_download(
                            repo_id=sub_repo,
                            local_dir=str(sub_dest),
                            local_dir_use_symlinks=False,
                        )
                    else:
                        logger.warning(f"[init] Unknown component {sub_name} — not in registry, skipping")

                logger.info("[init] Model download complete")
            except Exception as dl_err:
                raise RuntimeError(f"Model download failed: {dl_err}") from dl_err

        # ── GPU config ────────────────────────────────────────────────────
        gpu_config = get_gpu_config()
        set_global_gpu_config(gpu_config)
        logger.info(f"[init] GPU tier: {gpu_config.tier}, max duration: {gpu_config.max_duration_without_lm}s")

        # ── DiT handler ───────────────────────────────────────────────────
        _dit_handler = AceStepHandler()
        project_root = ACESTEP_DIR if os.path.isdir(ACESTEP_DIR) else os.path.dirname(CHECKPOINTS_DIR)
        _use_compile = bool(QUANTIZATION)  # torchao INT8 requires torch.compile
        status_msg, success = _dit_handler.initialize_service(
            project_root=project_root,
            config_path=DIT_MODEL,
            device=device,
            use_flash_attention=(device == "cuda"),
            compile_model=_use_compile,
            offload_to_cpu=CPU_OFFLOAD,
            offload_dit_to_cpu=CPU_OFFLOAD,
            quantization=QUANTIZATION,
        )
        if not success:
            raise RuntimeError(f"DiT init failed: {status_msg}")
        logger.info(f"[init] DiT ready: {status_msg[:150]}")

        # ── LLM handler (optional) ────────────────────────────────────────
        _llm_handler = LLMHandler()
        if LM_MODEL:
            try:
                lm_status, lm_success = _llm_handler.initialize(
                    checkpoint_dir=CHECKPOINTS_DIR,
                    lm_model_path=LM_MODEL,
                    backend="pt",
                    device=device,
                    offload_to_cpu=CPU_OFFLOAD,
                )
                if lm_success:
                    logger.info(f"[init] LLM ready: {lm_status[:100]}")
                else:
                    logger.warning(f"[init] LLM init failed (non-fatal): {lm_status[:100]}")
            except Exception as e:
                logger.warning(f"[init] LLM error (non-fatal): {e}")
        else:
            logger.info("[init] LLM disabled (ACESTEP_LM_MODEL not set)")

        # ── LoRA (optional) ───────────────────────────────────────────────
        if LORA_PATH and os.path.exists(LORA_PATH):
            logger.info(f"[init] Loading LoRA from: {LORA_PATH}")
            lora_status = _dit_handler.load_lora(LORA_PATH)
            _dit_handler.set_lora_scale(LORA_WEIGHT)
            logger.info(f"[init] LoRA loaded (weight={LORA_WEIGHT}): {lora_status}")

        _initialized = True
        logger.info(f"[init] ACE-Step ready in {time.time() - start:.1f}s")

    except Exception as e:
        _init_error = str(e)
        _initialized = True   # Mark done so we don't retry and block
        logger.error(f"[init] FAILED: {e}")
        raise


def upload_to_s3(local_path: str, object_key: str) -> str:
    """Upload audio file to S3. Returns the public/presigned URL."""
    import boto3

    kwargs = {
        "region_name": S3_REGION,
        "aws_access_key_id": S3_ACCESS_KEY,
        "aws_secret_access_key": S3_SECRET_KEY,
    }
    if S3_ENDPOINT:
        kwargs["endpoint_url"] = S3_ENDPOINT

    s3 = boto3.client("s3", **kwargs)
    s3.upload_file(
        local_path,
        S3_BUCKET,
        object_key,
        ExtraArgs={
            "ContentType": "audio/wav" if object_key.endswith(".wav") else "audio/mpeg",
            "CacheControl": "public, max-age=86400",
        },
    )

    # Generate presigned URL (7 days)
    url = s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": S3_BUCKET, "Key": object_key},
        ExpiresIn=604800,
    )
    return url


def handler(job):
    """
    RunPod serverless handler. Called for every generation job.

    Args:
        job: dict with keys "id" and "input"

    Returns:
        dict with "audio_url", "duration", "seed", "generation_time", "error"
    """
    job_id = job.get("id", "unknown")
    input_data = job.get("input", {})

    logger.info(f"[job:{job_id}] Received: {json.dumps(input_data, default=str)[:300]}")

    # ── Ensure model is loaded ────────────────────────────────────────────────
    if not _initialized:
        try:
            initialize_model()
        except Exception as e:
            return {"error": f"Model initialization failed: {e}"}

    if _init_error:
        return {"error": f"Model in failed state: {_init_error}"}

    # ── Parse input ───────────────────────────────────────────────────────────
    prompt = input_data.get("prompt", "")
    lyrics = input_data.get("lyrics") or ""
    duration = float(input_data.get("duration", 60.0))
    quality = input_data.get("quality", "standard")
    vocal_language = input_data.get("vocal_language", "es")
    vocal_gender = input_data.get("vocal_gender")
    vocal_timbre = input_data.get("vocal_timbre")           # text prompt descriptor e.g. "carlos", "sofia"
    voice_character = input_data.get("voice_character")     # drives reference audio (true voice identity)
    reference_audio_url = input_data.get("reference_audio_url")  # presigned R2 URL for 30s vocal WAV
    creativity = input_data.get("creativity")               # int 0-100 or None
    bpm = input_data.get("bpm")                      # int or None
    key_scale = input_data.get("key_scale", "")
    time_signature = input_data.get("time_signature", "")
    seed = input_data.get("seed", -1)
    use_windowed_attention = input_data.get("use_windowed_attention", True)
    job_lora_path = input_data.get("lora_path")      # optional per-job LoRA override
    # Cover mode: re-voice a source audio with a different reference voice
    task_type = input_data.get("task_type", "text2music")
    src_audio_url = input_data.get("src_audio_url")         # presigned URL for source song (cover mode)
    audio_cover_strength = float(input_data.get("audio_cover_strength", 0.8))

    if not prompt:
        return {"error": "prompt is required"}

    # ── Cap duration to prevent jobs that exceed RunPod execution window ──────
    if duration > MAX_DURATION_SECONDS:
        logger.warning(
            f"[job:{job_id}] Requested duration {duration}s exceeds cap {MAX_DURATION_SECONDS}s — clamping"
        )
        duration = float(MAX_DURATION_SECONDS)

    # ── Apply quality preset ──────────────────────────────────────────────────
    preset = QUALITY_PRESETS.get(quality, QUALITY_PRESETS["standard"])
    steps = preset["steps"]
    guidance_scale = preset["guidance_scale"]

    # Creativity overrides guidance_scale
    if creativity is not None:
        guidance_scale = _creativity_to_guidance(int(creativity))

    # Append vocal timbre descriptor (or gender fallback) to prompt
    prompt = _build_voice_prompt(prompt, vocal_gender, vocal_timbre)

    # Turbo model requires shift=3.0 for correct denoising schedule
    is_turbo = "turbo" in DIT_MODEL.lower()
    model_shift = 3.0 if is_turbo else 1.0

    logger.info(
        f"[job:{job_id}] prompt={prompt[:80]!r} duration={duration}s "
        f"steps={steps} guidance={guidance_scale} lang={vocal_language} bpm={bpm} "
        f"shift={model_shift} timbre={vocal_timbre or vocal_gender} "
        f"voice_char={voice_character or 'none'} "
        f"ref_audio={'yes' if reference_audio_url else 'no'}"
    )

    # ── Generate ──────────────────────────────────────────────────────────────
    from acestep.inference import GenerationParams, GenerationConfig, generate_music

    gen_start = time.time()
    # Initialize output vars so the outer except block never hits UnboundLocalError
    audio_url = None
    result_s3_key = None

    try:
        config = GenerationConfig(
            batch_size=1,
            use_random_seed=(seed is None or seed == -1),
            seeds=[int(seed)] if (seed is not None and seed != -1) else None,
            audio_format=AUDIO_FORMAT,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            # Download reference audio for TimbreEncoder (if provided).
            # Must be inside the tmpdir block so the file path stays valid during generation.
            local_ref_audio = None
            if reference_audio_url:
                try:
                    import urllib.request
                    ref_path = os.path.join(tmpdir, "reference.wav")
                    urllib.request.urlretrieve(reference_audio_url, ref_path)
                    ref_size = os.path.getsize(ref_path)
                    logger.info(f"[job:{job_id}] Voice ref downloaded ({ref_size//1024}KB)")
                    # Validate: must be >= 10KB and start with RIFF header
                    if ref_size < 10_000:
                        logger.warning(f"[job:{job_id}] Voice ref too small ({ref_size}B) — URL may have returned an error page, skipping")
                    else:
                        with open(ref_path, "rb") as _f:
                            _header = _f.read(4)
                        if _header != b"RIFF":
                            logger.warning(f"[job:{job_id}] Voice ref not a valid WAV (header={_header!r}), skipping")
                        else:
                            local_ref_audio = ref_path
                except Exception as ref_err:
                    logger.warning(f"[job:{job_id}] Voice ref download failed: {ref_err}")

            # Download source audio for cover mode (if provided)
            local_src_audio = None
            if src_audio_url and task_type == "cover":
                try:
                    import urllib.request
                    src_path = os.path.join(tmpdir, "source_audio.mp3")
                    urllib.request.urlretrieve(src_audio_url, src_path)
                    src_size = os.path.getsize(src_path)
                    logger.info(f"[job:{job_id}] Source audio downloaded ({src_size//1024}KB) for cover mode")
                    if src_size >= 10_000:
                        local_src_audio = src_path
                    else:
                        logger.warning(f"[job:{job_id}] Source audio too small ({src_size}B), skipping cover mode")
                        task_type = "text2music"
                except Exception as src_err:
                    logger.warning(f"[job:{job_id}] Source audio download failed: {src_err} — falling back to text2music")
                    task_type = "text2music"

            _lm_enabled = (LM_MODEL != "")
            params = GenerationParams(
                task_type=task_type,
                caption=prompt,
                lyrics=lyrics,
                vocal_language=vocal_language,
                instrumental=(not lyrics or not lyrics.strip() or lyrics.strip() == "[Instrumental]"),
                bpm=int(bpm) if bpm is not None else None,
                keyscale=key_scale or "",
                timesignature=str(time_signature) if time_signature else "",
                duration=duration,
                inference_steps=steps,
                guidance_scale=guidance_scale,
                seed=int(seed) if seed and seed != -1 else -1,
                shift=model_shift,
                thinking=_lm_enabled,
                use_cot_metas=_lm_enabled,
                # V3: Don't let LM rewrite our vocal-centric caption or override language
                use_cot_caption=False,
                use_cot_language=False,
                # V3: Windowed attention tuning for better lyric alignment
                use_windowed_attention=use_windowed_attention,
                guidance_steps_pct=0.75,      # Apply windowed attention for 75% of steps (was 50%)
                soft_mask_value=3.5,           # Stronger attention bias toward in-window positions (was 2.0)
                window_margin_ratio=0.2,       # Tighter windows, less bleed between sections (was 0.3)
                # V3: Tighter LM planner — more deterministic audio code generation
                lm_temperature=0.6,            # Was 0.85 — less random, more predictable lyric placement
                lm_cfg_scale=3.0,              # Was 2.0 — LM follows lyrics more strictly
                lm_top_p=0.8,                  # Was 0.9 — narrower sampling
                reference_audio=local_ref_audio,
                src_audio=local_src_audio,
                audio_cover_strength=audio_cover_strength,
            )

            # Load per-job LoRA if requested
            if job_lora_path and job_lora_path != LORA_PATH:
                try:
                    _dit_handler.load_lora(job_lora_path)
                    logger.info(f"[job:{job_id}] Loaded per-job LoRA: {job_lora_path}")
                except Exception as e:
                    logger.warning(f"[job:{job_id}] Failed to load per-job LoRA: {e}")

            # ── V3: Batch generation + best-of-N scoring ─────────────────────
            # Generate multiple candidates, score each with DiT Lyrics Alignment
            # Score, keep the best. Only activates for vocal songs.
            has_lyrics_for_batch = lyrics and lyrics.strip() and lyrics.strip() != "[Instrumental]"
            V3_NUM_CANDIDATES = int(os.environ.get("V3_NUM_CANDIDATES", "3"))
            num_candidates = V3_NUM_CANDIDATES if has_lyrics_for_batch else 1

            if num_candidates > 1:
                logger.info(f"[job:{job_id}] V3 batch scoring: generating {num_candidates} candidates")

            best_result = None
            best_score = -1.0
            best_audio_path = None
            best_audio_params = None

            for cand_idx in range(num_candidates):
                # Each candidate gets a unique random seed (except first if user specified)
                if cand_idx == 0:
                    cand_config = config
                else:
                    cand_config = GenerationConfig(
                        batch_size=1,
                        use_random_seed=True,
                        seeds=None,
                        audio_format=AUDIO_FORMAT,
                    )

                def _run_generation():
                    return generate_music(
                        dit_handler=_dit_handler,
                        llm_handler=_llm_handler,
                        params=params,
                        config=cand_config,
                        save_dir=tmpdir,
                    )

                # Run with timeout
                cand_result = None
                if GENERATION_TIMEOUT > 0:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(_run_generation)
                        try:
                            cand_result = future.result(timeout=GENERATION_TIMEOUT)
                        except concurrent.futures.TimeoutError:
                            logger.error(f"[job:{job_id}] Candidate {cand_idx+1}/{num_candidates} timed out")
                            continue
                else:
                    cand_result = _run_generation()

                if not cand_result or not cand_result.success or not cand_result.audios:
                    logger.warning(f"[job:{job_id}] Candidate {cand_idx+1}/{num_candidates} failed")
                    if best_result is None:
                        best_result = cand_result
                    continue

                # Score this candidate
                cand_score = 0.0
                if has_lyrics_for_batch and cand_result.extra_outputs and num_candidates > 1:
                    try:
                        extra = cand_result.extra_outputs
                        pl = extra.get("pred_latents")
                        ehs = extra.get("encoder_hidden_states")
                        eam = extra.get("encoder_attention_mask")
                        cl = extra.get("context_latents")
                        lti = extra.get("lyric_token_idss")
                        if all(v is not None for v in [pl, ehs, eam, cl, lti]):
                            sr = _dit_handler.get_lyric_score(
                                pred_latent=pl[0:1], encoder_hidden_states=ehs[0:1],
                                encoder_attention_mask=eam[0:1], context_latents=cl[0:1],
                                lyric_token_ids=lti[0:1], vocal_language=vocal_language,
                                inference_steps=int(steps),
                            )
                            if sr.get("success"):
                                dit_s = float(sr.get("dit_score", 0.0))
                                lm_s = float(sr.get("lm_score", 0.0))
                                cand_score = 0.7 * dit_s + 0.3 * lm_s
                                logger.info(
                                    f"[job:{job_id}] Candidate {cand_idx+1}/{num_candidates}: "
                                    f"dit={dit_s:.4f}, lm={lm_s:.4f}, combined={cand_score:.4f}"
                                )
                    except Exception as se:
                        logger.warning(f"[job:{job_id}] Scoring failed for candidate {cand_idx+1}: {se}")

                if cand_score > best_score or best_result is None:
                    best_score = cand_score
                    best_result = cand_result
                    ai = cand_result.audios[0]
                    best_audio_path = ai.get("path", "")
                    best_audio_params = ai.get("params", {})
                    if num_candidates > 1:
                        logger.info(f"[job:{job_id}] V3 new best: candidate {cand_idx+1} (score={cand_score:.4f})")

            if num_candidates > 1:
                logger.info(f"[job:{job_id}] V3 batch complete: best score={best_score:.4f}")

            # If all candidates timed out
            if best_result is None:
                return {
                    "error": f"All {num_candidates} candidates failed or timed out",
                    "generation_time": time.time() - gen_start,
                }

            result = best_result
            gen_time = time.time() - gen_start

            if not result.success:
                return {"error": result.error or "Generation failed", "generation_time": gen_time}

            if not result.audios:
                return {"error": "No audio produced", "generation_time": gen_time}

            audio_info = result.audios[0]
            audio_path = best_audio_path or audio_info.get("path", "")
            audio_params = best_audio_params or audio_info.get("params", {})
            actual_seed = audio_params.get("seed", seed)

            if not audio_path or not os.path.exists(audio_path):
                return {"error": f"Audio file missing at: {audio_path}", "generation_time": gen_time}

            # ── LRC + alignment scoring (cross-attention) ─────────────────────
            lrc_text = None
            alignment_scores = None
            lm_metadata = None

            # Capture lm_metadata (auto-detected BPM, key, time sig, language, genres)
            if result.extra_outputs:
                raw_lm_meta = result.extra_outputs.get("lm_metadata")
                if raw_lm_meta and isinstance(raw_lm_meta, dict):
                    lm_metadata = {
                        k: v for k, v in raw_lm_meta.items()
                        if k in ("bpm", "keyscale", "timesignature", "language", "caption", "genres", "duration")
                        and v is not None
                    } or None

            has_lyrics = lyrics and lyrics.strip() and lyrics.strip() != "[Instrumental]"
            if has_lyrics and result.extra_outputs:
                try:
                    extra = result.extra_outputs
                    pred_latents = extra.get("pred_latents")
                    encoder_hidden_states = extra.get("encoder_hidden_states")
                    encoder_attention_mask = extra.get("encoder_attention_mask")
                    context_latents = extra.get("context_latents")
                    lyric_token_idss = extra.get("lyric_token_idss")
                    if all(v is not None for v in [pred_latents, encoder_hidden_states,
                                                   encoder_attention_mask, context_latents,
                                                   lyric_token_idss]):
                        # LRC timestamps
                        lrc_result = _dit_handler.get_lyric_timestamp(
                            pred_latent=pred_latents[0:1],
                            encoder_hidden_states=encoder_hidden_states[0:1],
                            encoder_attention_mask=encoder_attention_mask[0:1],
                            context_latents=context_latents[0:1],
                            lyric_token_ids=lyric_token_idss[0:1],
                            total_duration_seconds=float(duration),
                            vocal_language=vocal_language,
                            inference_steps=int(steps),
                        )
                        if lrc_result.get("success") and lrc_result.get("lrc_text"):
                            lrc_text = lrc_result["lrc_text"]
                            logger.info(f"[job:{job_id}] LRC generated ({len(lrc_text)} chars)")
                        # DiT alignment scores
                        score_result = _dit_handler.get_lyric_score(
                            pred_latent=pred_latents[0:1],
                            encoder_hidden_states=encoder_hidden_states[0:1],
                            encoder_attention_mask=encoder_attention_mask[0:1],
                            context_latents=context_latents[0:1],
                            lyric_token_ids=lyric_token_idss[0:1],
                            vocal_language=vocal_language,
                            inference_steps=int(steps),
                        )
                        if score_result.get("success"):
                            alignment_scores = {
                                "lm_score": round(float(score_result.get("lm_score", 0.0)), 4),
                                "dit_score": round(float(score_result.get("dit_score", 0.0)), 4),
                            }
                            logger.info(f"[job:{job_id}] Alignment scores: {alignment_scores}")
                except Exception as _lrc_exc:
                    logger.warning(f"[job:{job_id}] LRC/scoring failed (non-fatal): {_lrc_exc}")

            logger.info(f"[job:{job_id}] Generated in {gen_time:.1f}s, uploading to S3...")

            # ── Upload to S3 ──────────────────────────────────────────────
            ext = os.path.splitext(audio_path)[1].lstrip(".") or AUDIO_FORMAT
            object_key = f"generated/{uuid.uuid4()}.{ext}"

            try:
                audio_url = upload_to_s3(audio_path, object_key)
                logger.info(f"[job:{job_id}] Uploaded: {object_key}")
            except Exception as e:
                logger.error(f"[job:{job_id}] S3 upload failed: {e}")
                return {"error": f"S3 upload failed: {e}", "generation_time": gen_time}

            # Store the object key so the Flask API can regenerate presigned URLs
            # on demand (avoids 7-day expiry of presigned URLs stored in the DB)
            result_s3_key = object_key

    except Exception as e:
        logger.error(f"[job:{job_id}] Generation error: {e}", exc_info=True)
        return {"error": str(e), "generation_time": time.time() - gen_start}

    total_time = time.time() - gen_start
    logger.info(f"[job:{job_id}] Complete in {total_time:.1f}s")

    return {
        "audio_url": audio_url,       # Presigned URL (valid 7 days) — use for immediate playback
        "s3_key": result_s3_key,       # Stable object key — use for permanent storage
        "lrc_text": lrc_text,          # Timestamped lyrics (LRC format) or None
        "alignment_scores": alignment_scores,  # DiT alignment scores {lm_score, dit_score}
        "lm_metadata": lm_metadata,    # LM auto-detected: bpm, keyscale, language, genres, etc.
        "duration": duration,
        "format": ext,
        "seed": actual_seed,
        "generation_time": round(total_time, 2),
        "metadata": {
            "prompt": prompt,
            "quality": quality,
            "steps": steps,
            "guidance_scale": guidance_scale,
            "vocal_language": vocal_language,
            "bpm": bpm,
        },
    }


# ── Entrypoint ────────────────────────────────────────────────────────────────
# Pre-warm model on worker start (reduces first-job latency)
logger.info("[startup] Pre-warming ACE-Step model...")
try:
    initialize_model()
except Exception as e:
    logger.error(f"[startup] Pre-warm failed (will retry on first job): {e}")

runpod.serverless.start({"handler": handler})
