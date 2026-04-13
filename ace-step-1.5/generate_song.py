#!/usr/bin/env python3
"""
MUSICA Unified Song Generator — Generate songs for ANY genre using the lyrics engine.

Usage:
    python generate_song.py --genre "Reggaeton"
    python generate_song.py --genre "Cha-Cha-Cha" --duration 120 --description "una fiesta en La Habana"
    python generate_song.py --genre "Corrido" --duration 180 --device mps
    python generate_song.py --list-genres          # Show all available genres
    python generate_song.py --list-frameworks      # Show genre→framework mappings

Requires:
    - GROQ_API_KEY env var (or --api-key flag) for lyrics generation
    - ACE-Step 1.5 model downloaded in checkpoints/

The script:
1. Calls our lyrics engine (Groq LLM + genre-conditional prompts from lyrics_prompts.py)
2. Gets the genre-specific caption from genre_specs.py
3. Feeds both to ACE-Step for audio generation
"""

import os
import sys
import json
import time
import types
import argparse
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Step 0: Compatibility patches (must run before any ACE-Step imports)
# ---------------------------------------------------------------------------

def _patch_for_compatibility():
    """Mock numba and fix PyTorch weight_norm for Intel Mac / older PyTorch."""
    # Mock numba (only used for LRC timestamps, not generation)
    numba_mock = types.ModuleType("numba")
    def _jit_decorator(*args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return lambda func: func
    numba_mock.jit = _jit_decorator
    sys.modules["numba"] = numba_mock

    # Fix weight_norm for PyTorch < 2.4 + diffusers meta tensors
    try:
        import torch
        from torch.nn.utils.weight_norm import WeightNorm
        _original = WeightNorm.compute_weight
        def _patched(self, module):
            g = getattr(module, self.name + '_g')
            v = getattr(module, self.name + '_v')
            if v.device.type == 'meta' or g.device.type == 'meta':
                return torch.empty_like(v)
            return torch._weight_norm(v, g, self.dim)
        WeightNorm.compute_weight = _patched
    except Exception:
        pass

_patch_for_compatibility()

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
API_DIR = os.path.join(PROJECT_ROOT, "api")

# Add both ACE-Step and api/ to path so we can import lyrics engine + genre specs
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
if API_DIR not in sys.path:
    sys.path.insert(0, API_DIR)


# ---------------------------------------------------------------------------
# Lyrics engine (calls Groq LLM with our genre-conditional prompts)
# ---------------------------------------------------------------------------

def generate_lyrics(genre: str, description: str, duration: int, api_key: str) -> dict:
    """Call Groq to generate lyrics using our lyrics engine.

    Returns dict with keys: lyrics, genre, prompt (caption), title, cover_query
    """
    import requests
    from lyrics_prompts import build_lyrics_prompt

    system_prompt = build_lyrics_prompt(genre=genre)

    # Build user message
    if description:
        user_msg = f"{description}\n\nGénero seleccionado: {genre}\nDuración: {duration} segundos"
    else:
        user_msg = f"Escríbeme una canción de {genre}.\n\nGénero seleccionado: {genre}\nDuración: {duration} segundos"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_msg},
    ]

    print(f"  -> Calling Groq (llama-3.3-70b-versatile) for {genre} lyrics...")
    llm_output = None
    for attempt in range(3):
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": messages,
                "temperature": 0.9,
                "max_tokens": 2048,
                "frequency_penalty": 0.8,
                "presence_penalty": 0.5,
            },
            timeout=60,
        )
        if resp.status_code == 429:
            wait = (attempt + 1) * 30
            print(f"  -> Rate limited, waiting {wait}s (attempt {attempt + 1}/3)...")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        llm_output = resp.json()["choices"][0]["message"]["content"]
        break
    if llm_output is None:
        raise RuntimeError("Groq rate limit exceeded after 3 retries. Wait a few minutes and try again.")

    # Extract lyrics (everything before the JSON block)
    import re
    lyrics = ""
    json_config = {}

    # Find JSON block
    json_match = re.search(r'```json\s*\n(.*?)\n\s*```', llm_output, re.DOTALL)
    if not json_match:
        json_match = re.search(r'\{[^{}]*"ready_to_generate"[^{}]*\}', llm_output, re.DOTALL)

    if json_match:
        try:
            json_str = json_match.group(1) if '```' in json_match.group(0) else json_match.group(0)
            json_config = json.loads(json_str)
        except json.JSONDecodeError:
            print("  -> WARNING: Could not parse JSON config from LLM output")

        # Lyrics = everything before the JSON block
        text_before_json = llm_output[:json_match.start()]
        # Find first section tag
        tag_match = re.search(
            r'\[(?:Intro|Verse|Chorus|Pre-Chorus|Bridge|Outro|Hook|Instrumental|'
            r'Coro|Pregon|Mambo|Jaleo|Guitar Solo|Piano Interlude|Breakdown|'
            r'Build|Drop|Final Chorus|Grito)',
            text_before_json
        )
        if tag_match:
            lyrics = text_before_json[tag_match.start():].strip()

    if not lyrics:
        # Fallback: try to extract any section-tagged text
        tag_match = re.search(r'\[(?:Intro|Verse|Chorus|Coro)', llm_output)
        if tag_match:
            lyrics = llm_output[tag_match.start():].strip()
            # Remove JSON block if embedded
            if '```json' in lyrics:
                lyrics = lyrics[:lyrics.index('```json')].strip()

    return {
        "lyrics": lyrics,
        "genre": json_config.get("genre", genre),
        "prompt": json_config.get("prompt", ""),
        "title": json_config.get("title", f"Test {genre}"),
        "cover_query": json_config.get("cover_query", ""),
    }


def get_genre_caption(genre: str) -> str:
    """Get genre-specific caption from genre_specs.py as fallback."""
    from genre_specs import enhance_prompt_with_genre
    return enhance_prompt_with_genre(f"{genre} style, Spanish vocals", genre)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def list_genres():
    """Print all available genres from genre_specs.py."""
    from genre_specs import LATIN_GENRE_SPECS
    print(f"\n{'Genre':<25} {'BPM':>6}  {'Time Sig':>8}  Mood")
    print("-" * 80)
    for name, spec in sorted(LATIN_GENRE_SPECS.items()):
        bpm = spec.get("bpm_typical", "?")
        ts = spec.get("time_signature", "?")
        mood = spec.get("mood", "")[:40]
        print(f"  {name:<23} {bpm:>6}  {ts:>8}  {mood}")
    print(f"\nTotal: {len(LATIN_GENRE_SPECS)} genres/subgenres")


def list_frameworks():
    """Print genre→framework mappings from lyrics_prompts.py."""
    from lyrics_prompts import GENRE_TO_FRAMEWORK, FRAMEWORKS
    print(f"\n{'Genre':<25} {'Framework':>10}  Description")
    print("-" * 70)
    # Group by framework
    fw_genres = {}
    for genre, fw in sorted(GENRE_TO_FRAMEWORK.items()):
        fw_genres.setdefault(fw, []).append(genre)
    for fw_key in sorted(fw_genres.keys()):
        fw_text = FRAMEWORKS.get(fw_key, "")
        # Extract first line of framework description
        desc = ""
        for line in fw_text.strip().split("\n"):
            if line.startswith("## Song Structure"):
                desc = line.replace("## Song Structure — ", "")
                break
        genres_str = ", ".join(fw_genres[fw_key])
        print(f"  {fw_key:<10} {desc:<30}  → {genres_str}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate a song for any genre using the MUSICA lyrics engine + ACE-Step",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--genre", type=str, help="Genre name (e.g., 'Reggaeton', 'Cha-Cha-Cha', 'Corrido')")
    parser.add_argument("--description", "-d", type=str, default="", help="Song description/theme (optional — lyrics engine invents if empty)")
    parser.add_argument("--duration", type=int, default=60, help="Duration in seconds (default: 60)")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "mps", "cuda"], help="Device for generation")
    parser.add_argument("--steps", type=int, default=8, help="Inference steps (default: 8 for turbo)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory (default: audio_files/)")
    parser.add_argument("--api-key", type=str, default=None, help="Groq API key (default: GROQ_API_KEY env var)")
    parser.add_argument("--lyrics-only", action="store_true", help="Only generate lyrics (skip audio generation)")
    parser.add_argument("--list-genres", action="store_true", help="List all available genres")
    parser.add_argument("--list-frameworks", action="store_true", help="List genre→framework mappings")
    args = parser.parse_args()

    if args.list_genres:
        list_genres()
        return

    if args.list_frameworks:
        list_frameworks()
        return

    if not args.genre:
        parser.error("--genre is required (use --list-genres to see options)")

    api_key = args.api_key or os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        parser.error("GROQ_API_KEY env var or --api-key required for lyrics generation")

    output_dir = args.output_dir or os.path.join(PROJECT_ROOT, "audio_files")
    os.makedirs(output_dir, exist_ok=True)

    genre = args.genre
    safe_genre = genre.lower().replace(" ", "_").replace("-", "_")

    # ── Step 1: Generate lyrics ──────────────────────────────────────────
    print(f"\n[1/4] Generating lyrics for: {genre}")
    lyrics_result = generate_lyrics(genre, args.description, args.duration, api_key)

    lyrics = lyrics_result["lyrics"]
    title = lyrics_result["title"]
    # Use LLM-generated caption if available, otherwise fall back to genre_specs
    caption = lyrics_result["prompt"] or get_genre_caption(genre)

    line_count = len([l for l in lyrics.split("\n") if l.strip() and not l.strip().startswith("[")])
    print(f"  -> Title: {title}")
    print(f"  -> Lyrics: {line_count} lines")
    print(f"  -> Caption: {caption[:80]}...")

    if args.lyrics_only:
        print(f"\n{'='*60}")
        print(f"TITLE: {title}")
        print(f"GENRE: {genre}")
        print(f"CAPTION: {caption}")
        print(f"{'='*60}")
        print(lyrics)
        print(f"{'='*60}")
        # Save lyrics to file
        lyrics_path = os.path.join(output_dir, f"{safe_genre}_lyrics.txt")
        with open(lyrics_path, "w") as f:
            f.write(f"# {title}\n# Genre: {genre}\n# Caption: {caption}\n\n{lyrics}")
        print(f"  -> Saved to: {lyrics_path}")
        return

    # ── Step 2: Initialize ACE-Step ──────────────────────────────────────
    print(f"\n[2/4] Initializing ACE-Step on {args.device}...")

    from acestep.handler import AceStepHandler
    from acestep.llm_inference import LLMHandler
    from acestep.inference import GenerationParams, GenerationConfig, generate_music

    dit_handler = AceStepHandler()
    llm_handler = LLMHandler()

    use_flash = args.device == "cuda"
    init_start = time.time()
    status, success = dit_handler.initialize_service(
        project_root=SCRIPT_DIR,
        config_path="acestep-v15-turbo",
        device=args.device,
        use_flash_attention=use_flash,
        compile_model=False,
        offload_to_cpu=(args.device != "cpu"),
        offload_dit_to_cpu=False,
    )
    init_time = time.time() - init_start

    if not success:
        print(f"  ERROR: Failed to initialize model: {status}")
        sys.exit(1)
    print(f"  -> Initialized in {init_time:.1f}s")

    # ── Step 3: Generate audio ───────────────────────────────────────────
    print(f"\n[3/4] Generating {args.duration}s {genre} song...")
    if args.device == "cpu":
        print("  -> WARNING: CPU generation is SLOW (expect 30-60+ minutes for long songs)")

    # Detect BPM from genre_specs if available
    bpm = None
    try:
        from genre_specs import LATIN_GENRE_SPECS
        genre_lower = genre.lower().strip()
        for key, spec in LATIN_GENRE_SPECS.items():
            if key == genre_lower or genre_lower in key or key in genre_lower:
                bpm = int(spec.get("bpm_typical", 0)) or None
                break
    except Exception:
        pass

    is_turbo = True  # We always use turbo model
    params = GenerationParams(
        task_type="text2music",
        caption=caption,
        lyrics=lyrics,
        vocal_language="es",
        bpm=bpm,
        duration=float(args.duration),
        inference_steps=args.steps,
        guidance_scale=7.0,
        seed=args.seed,
        shift=3.0 if is_turbo else 1.0,
        thinking=False,
        use_cot_metas=False,
        use_cot_caption=False,
        use_cot_language=False,
        use_windowed_attention=True,
    )

    config = GenerationConfig(
        batch_size=1,
        use_random_seed=False,
        seeds=[args.seed],
        audio_format="wav",
    )

    print(f"  -> Caption: {caption[:60]}...")
    print(f"  -> BPM: {bpm or 'auto'}")
    print(f"  -> Duration: {args.duration}s | Steps: {args.steps} | Seed: {args.seed}")

    gen_start = time.time()
    result = generate_music(
        dit_handler=dit_handler,
        llm_handler=llm_handler,
        params=params,
        config=config,
        save_dir=output_dir,
    )
    gen_time = time.time() - gen_start

    # ── Step 4: Save results ─────────────────────────────────────────────
    print(f"\n[4/4] Generation complete in {gen_time:.1f}s ({gen_time/60:.1f} minutes)")

    if result.success:
        import shutil
        target_name = f"{safe_genre}_{args.duration}s_seed{args.seed}"
        for i, audio in enumerate(result.audios):
            audio_path = audio.get("path", "")
            if audio_path:
                ext = os.path.splitext(audio_path)[1] or ".wav"
                target_path = os.path.join(output_dir, f"{target_name}{ext}")
                if audio_path != target_path:
                    shutil.copy2(audio_path, target_path)
                file_size = os.path.getsize(target_path)
                print(f"  -> Audio: {target_path} ({file_size / (1024*1024):.1f} MB)")
            else:
                tensor = audio.get("tensor")
                if tensor is not None:
                    import torchaudio
                    target_path = os.path.join(output_dir, f"{target_name}.wav")
                    sample_rate = audio.get("sample_rate", 48000)
                    torchaudio.save(target_path, tensor.cpu(), sample_rate)
                    print(f"  -> Audio: {target_path}")

        # Save metadata
        meta_path = os.path.join(output_dir, f"{target_name}_meta.json")
        with open(meta_path, "w") as f:
            json.dump({
                "title": title,
                "genre": genre,
                "caption": caption,
                "lyrics": lyrics,
                "duration": args.duration,
                "bpm": bpm,
                "seed": args.seed,
                "steps": args.steps,
                "device": args.device,
                "gen_time_seconds": round(gen_time, 1),
            }, f, indent=2, ensure_ascii=False)
        print(f"  -> Metadata: {meta_path}")
        print(f"\n  SUCCESS: {title}")
    else:
        print(f"  FAILED: {result.error}")
        print(f"  Status: {result.status_message}")
        sys.exit(1)


if __name__ == "__main__":
    main()
