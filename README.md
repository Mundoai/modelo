# LaMusica — XL-Turbo Worker

ACE-Step v1.5 XL-Turbo (4B DiT) RunPod serverless worker for music generation.

## Staging endpoint — NOT production

This is the XL-Turbo staging worker. Production uses the 2B turbo model
at a separate endpoint (`pl3pv6kxsy9jf2`). Do NOT point production traffic here.

## Environment Variables

| Var | Default | Description |
|-----|---------|-------------|
| `ACESTEP_DIT_MODEL` | `acestep-v15-xl-turbo` | DiT model variant |
| `ACESTEP_QUANTIZATION` | (empty) | `int8_weight_only` for INT8, empty for bf16 |
| `ACESTEP_LM_MODEL` | `acestep-5Hz-lm-4B` | LM planner model |
| `ACESTEP_CPU_OFFLOAD` | `false` | CPU offload (auto/true/false) |
| `ACESTEP_AUDIO_FORMAT` | `mp3` | Output format |
| `CHECKPOINTS_DIR` | `/runpod-volume/checkpoints` | Persistent volume path |
| `S3_*` | — | S3/R2 credentials (same as production) |

## RunPod Endpoint Config

- FlashBoot: ON
- Workers: min=0, max=1, idle=10s
- Timeout: 600s
- GPUs: 48GB only (A40, L40, L40S, RTX A6000, RTX 6000 Ada)

## Testing

Use RunPod dashboard or curl:

```bash
curl -X POST "https://api.runpod.ai/v2/ENDPOINT_ID/run" \
  -H "Authorization: Bearer RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d @test_job.json
```

## Switching bf16 / INT8

| Mode | Env Var | LoRA | Use Case |
|------|---------|------|----------|
| bf16 (default) | `ACESTEP_QUANTIZATION=` | Yes | LoRA testing, quality |
| INT8 | `ACESTEP_QUANTIZATION=int8_weight_only` | Patched peft | Speed testing |
