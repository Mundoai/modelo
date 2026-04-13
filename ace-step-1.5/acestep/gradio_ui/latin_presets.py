"""
LaMusica Latin Genre Presets for ACE-Step v1.5

Ported from v1.0 customizations. Contains:
- 33 Latin genre presets with detailed instrument tags, BPM, and mood descriptors
- Voice timbre system for appending vocal descriptors to captions
- Per-genre optimized inference settings for Latin genres
- Local LoRA model auto-detection
"""

import os

# ============================================================================
# GENRE PRESETS - Detailed instrument tags, BPM, and mood descriptors
# ============================================================================

GENRE_PRESETS = {
    # === CARIBBEAN GENRES ===
    "Salsa": "salsa, piano montuno, congas, timbales, bongos, brass section, trumpet, trombone, tropical rhythm, Latin percussion, 180 bpm, energetic, danceable, passionate, Spanish vocals, Cuban style",
    "Bachata": "bachata, acoustic guitar, bongos, guira, bass, requinto, 130 bpm, romantic, sensual, Latin, Spanish vocals, smooth male vocals, Dominican style",
    "Merengue": "merengue, tambora, guira, saxophone, bass, 160 bpm, fast, energetic, danceable, Spanish vocals, Dominican style, Caribbean, tropical",
    "Son Cubano": "son cubano, tres guitar, bongos, maracas, trumpet, bass, 120 bpm, traditional, Afro-Cuban, Spanish vocals, call and response",
    "Mambo": "mambo, brass section, congas, timbales, piano, bass, 180 bpm, big band, energetic, danceable, Spanish vocals, Cuban style",
    "Cha-Cha-Cha": "cha-cha-cha, guiro, timbales, piano, flute, violins, 120 bpm, elegant, danceable, Cuban, Spanish vocals, charanga style",
    "Bolero": "bolero, acoustic guitar, piano, strings, soft percussion, 80 bpm, romantic, melancholic, passionate, Spanish vocals, smooth male vocals, classic Latin",
    "Reggaeton": "reggaeton, dembow beat, 808 bass, hi-hats, synth, 95 bpm, urban, Latin trap, perreo, Spanish vocals, male vocals, sexy, club music",

    # === REGGAETON SUBGENRES ===
    "Dembow": "dembow, fast dancehall rhythm, boom-ch-boom-chick, 808 bass, hi-hats, 100 bpm, Puerto Rican, energetic, perreo, Spanish vocals, raw urban",
    "Pop-Reggaeton": "pop reggaeton, melodic, catchy hooks, smooth vocals, synth, 95 bpm, mainstream, radio-friendly, Spanish vocals, polished production, commercial",
    "Romantiqueo": "reggaeton romantico, soft dembow, love ballad, sensual, 90 bpm, romantic, slow grind, Spanish vocals, smooth male vocals, intimate",
    "Malianteo": "malianteo, gritty reggaeton, street, raw 808 bass, aggressive hi-hats, 92 bpm, hustling, callejero, Spanish vocals, raspy male vocals, underground",
    "Neoperreo": "neoperreo, experimental reggaeton, underground, distorted bass, glitchy synths, 98 bpm, avant-garde, Spanish vocals, innovative, bold production",
    "Trapeton": "trapeton, Latin trap fusion, dark 808 bass, triplet hi-hats, 80 bpm, heavy bass, auto-tune, Spanish vocals, trap flow, moody",
    "Dominican Dembow": "Dominican dembow, fast rhythm, simple beat, heavy bass, 120 bpm, Santo Domingo, party, Spanish vocals, energetic, Caribbean",
    "Cubaton": "cubaton, timba, piano montuno, timbales, congas, brass section, trumpet, dembow rhythm, 108 bpm, Havana, tropical party, Spanish vocals, Cuban flavor, energetic, salsa urbana",
    "RKT": "RKT, cachengue, cumbiaton, Argentine reggaeton, cumbia villera fusion, 100 bpm, Buenos Aires, perreo intenso, Spanish vocals, villero style",

    # === MEXICAN GENRES ===
    "Mariachi": "mariachi, trumpets, violins, vihuela, guitarron, acoustic guitar, 110 bpm, traditional Mexican, passionate, Spanish male vocals, fiesta, celebration",
    "Ranchera": "ranchera, mariachi, trumpets, violins, guitar, vihuela, 100 bpm, passionate, dramatic, Mexican, Spanish male vocals, emotional, heartbreak",
    "Corrido": "corrido, accordion, bajo sexto, drums, bass, 120 bpm, Mexican ballad, storytelling, narrative, Spanish male vocals, regional Mexican",
    "Norteno": "norteno, accordion, bajo sexto, drums, bass, 120 bpm, polka influence, Mexican, Spanish male vocals, upbeat, Tejano",
    "Banda": "banda, brass section, clarinets, drums, tuba, 130 bpm, festive, Mexican, Spanish vocals, loud, Sinaloa style, party music",
    "Cumbia Mexicana": "cumbia sonidera, synth, bass, drums, guiro, 100 bpm, Mexican cumbia, tropical, danceable, Spanish vocals, party",
    "Bolero Ranchero": "bolero ranchero, mariachi, violins, guitar, trumpets, 80 bpm, romantic Mexican, passionate, Spanish male vocals, sentimental",

    # === SOUTH AMERICAN GENRES ===
    "Cumbia": "cumbia, accordion, drums, bass, guira, gaita, 100 bpm, festive, tropical, danceable, Spanish vocals, joyful, Colombian style",
    "Vallenato": "vallenato, accordion, caja vallenata, guacharaca, bass, 110 bpm, romantic, storytelling, Colombian, Spanish male vocals, folk",
    "Tango": "tango, bandoneon, violin, piano, double bass, 130 bpm, dramatic, passionate, Argentine, Spanish vocals, melancholic",

    # === MODERN LATIN ===
    "Latin Pop": "latin pop, synth, drums, guitar, piano, 110 bpm, catchy, romantic, Spanish vocals, polished production, radio-friendly",
    "Latin Rock": "latin rock, electric guitar, drums, bass, 130 bpm, energetic, Spanish vocals, rock en espanol, powerful",

    # === OTHER GENRES ===
    "Modern Pop": "pop, synth, drums, guitar, 120 bpm, upbeat, catchy, vibrant, female vocals, polished vocals",
    "Rock": "rock, electric guitar, drums, bass, 130 bpm, energetic, rebellious, gritty, male vocals, raw vocals",
    "Hip Hop": "hip hop, 808 bass, hi-hats, synth, 90 bpm, bold, urban, intense, male vocals, rhythmic vocals",
    "Country": "country, acoustic guitar, steel guitar, fiddle, 100 bpm, heartfelt, rustic, warm, male vocals, twangy vocals",
    "EDM": "edm, synth, bass, kick drum, 128 bpm, euphoric, pulsating, energetic, instrumental",
    "Reggae": "reggae, guitar, bass, drums, 80 bpm, chill, soulful, positive, male vocals, smooth vocals",
    "Classical": "classical, orchestral, strings, piano, 60 bpm, elegant, emotive, timeless, instrumental",
    "Jazz": "jazz, saxophone, piano, double bass, 110 bpm, smooth, improvisational, soulful, male vocals, crooning vocals",
    "Metal": "metal, electric guitar, double kick drum, bass, 160 bpm, aggressive, intense, heavy, male vocals, screamed vocals",
    "R&B": "r&b, synth, bass, drums, 85 bpm, sultry, groovy, romantic, female vocals, silky vocals",
}

# ============================================================================
# VOICE TIMBRES - Append descriptive vocal tags to caption
# ============================================================================

VOICE_TIMBRES = {
    "Default": "",  # Use genre default
    "Male Smooth": "male vocals, smooth tenor, warm voice, clear pronunciation",
    "Male Deep": "male vocals, deep baritone, rich voice, powerful",
    "Male Raspy": "male vocals, raspy voice, soulful, gritty texture",
    "Female Smooth": "female vocals, smooth soprano, silky voice, elegant",
    "Female Powerful": "female vocals, powerful belting, strong voice, diva",
    "Female Soft": "female vocals, soft whisper, breathy, intimate",
    "Duet": "male and female vocals, duet, harmonies, call and response",
}

# ============================================================================
# LATIN GENRES LIST - Used to identify which genres get Latin-optimized settings
# ============================================================================

LATIN_GENRE_NAMES = [
    # Caribbean
    "Salsa", "Bachata", "Merengue", "Son Cubano", "Mambo", "Cha-Cha-Cha", "Bolero", "Reggaeton",
    # Reggaeton Subgenres
    "Dembow", "Pop-Reggaeton", "Romantiqueo", "Malianteo", "Neoperreo", "Trapeton",
    "Dominican Dembow", "Cubaton", "RKT",
    # Mexican
    "Mariachi", "Ranchera", "Corrido", "Norteno", "Banda", "Cumbia Mexicana", "Bolero Ranchero",
    # South American
    "Cumbia", "Vallenato", "Tango",
    # Modern Latin
    "Latin Pop", "Latin Rock",
]

# ============================================================================
# LOCAL LORA AUTO-DETECTION
# ============================================================================

LORA_BASE_DIR = os.path.expanduser("~/Desktop/PROJECTS/MUSICA/models/lora/")


def get_available_loras():
    """Auto-detect LoRA models from the local lora directory.

    Returns:
        list: List of tuples (display_name, path) for available LoRAs.
    """
    lora_choices = [("None (Base Model)", "")]

    if not os.path.isdir(LORA_BASE_DIR):
        return lora_choices

    for entry in sorted(os.listdir(LORA_BASE_DIR)):
        full_path = os.path.join(LORA_BASE_DIR, entry)
        if os.path.isdir(full_path):
            # Check if it looks like a LoRA directory (has adapter_config.json or similar)
            has_adapter = any(
                os.path.exists(os.path.join(full_path, f))
                for f in ["adapter_config.json", "adapter_model.safetensors", "adapter_model.bin"]
            )
            if has_adapter:
                display_name = entry.replace("_", " ").replace("-", " ").title()
                lora_choices.append((display_name, full_path))
            else:
                # Still list directories that might be LoRA models
                display_name = entry.replace("_", " ").replace("-", " ").title()
                lora_choices.append((f"{display_name} (unverified)", full_path))

    return lora_choices


# ============================================================================
# HANDLER FUNCTIONS for Gradio events
# ============================================================================

def update_caption_from_preset(preset_name, voice_timbre):
    """Update caption text when genre preset or voice timbre changes.

    Args:
        preset_name: Selected genre preset name, or "Custom"
        voice_timbre: Selected voice timbre name

    Returns:
        str: Updated caption text
    """
    if preset_name == "Custom":
        return ""  # Let user write their own

    caption = GENRE_PRESETS.get(preset_name, "")

    # Append voice timbre tags if not default
    if voice_timbre and voice_timbre != "Default":
        timbre_tags = VOICE_TIMBRES.get(voice_timbre, "")
        if timbre_tags:
            caption = f"{caption}, {timbre_tags}"

    return caption


def update_caption_from_voice(preset_name, voice_timbre):
    """Update only caption when voice timbre changes (keeps current genre).

    Args:
        preset_name: Current genre preset name
        voice_timbre: Newly selected voice timbre name

    Returns:
        str: Updated caption text
    """
    return update_caption_from_preset(preset_name, voice_timbre)


def is_latin_genre(genre_name):
    """Check if a genre is a Latin genre that gets optimized settings."""
    return genre_name in LATIN_GENRE_NAMES
