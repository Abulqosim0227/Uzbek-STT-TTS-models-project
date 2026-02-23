#!/usr/bin/env python3
"""
V3 FERRARI UNLEASHED
Fixing the "held back" feeling with parameter tuning + noise reduction
"""

import torch
from TTS.utils.synthesizer import Synthesizer
import soundfile as sf
import noisereduce as nr
import numpy as np
import os

# === CONFIG ===
checkpoint_path = "/mnt/c/Users/Admin/Desktop/voice_dataset/ISSAI_USC/uzbek_tts_project/training_output_ipa_v3/uzbek_tts_ipa_v3-December-12-2025_09+25AM-0000000/checkpoint_60000.pth"
config_path = "/mnt/c/Users/Admin/Desktop/voice_dataset/ISSAI_USC/uzbek_tts_project/training_output_ipa_v3/uzbek_tts_ipa_v3-December-12-2025_09+25AM-0000000/config.json"

# === OUTPUT FOLDER ===
AUDIO_DIR = "/mnt/c/Users/Admin/Desktop/voice_dataset/ISSAI_USC/uzbek_tts_project/audio"
os.makedirs(AUDIO_DIR, exist_ok=True)

# === IPA CONVERTER (FIXED VERSION) ===
def uzbek_to_ipa(text):
    """Convert Uzbek to IPA - matches training data format"""
    text = text.lower()

    # Normalize apostrophes
    for apo in ["'", "'", "`", "ʻ", "ʼ"]:
        text = text.replace(apo, "'")

    # Multi-char first (order matters!)
    text = text.replace("o'", "ø")
    text = text.replace("g'", "ʁ")
    text = text.replace("sh", "ʃ")
    text = text.replace("ch", "tʃ")
    text = text.replace("ng", "ŋ")

    # Single chars
    text = text.replace("x", "χ")
    text = text.replace("j", "dʒ")  # CORRECT: j -> dʒ
    text = text.replace("y", "j")   # y -> j
    text = text.replace("a", "ɑ")
    text = text.replace("o", "ɔ")

    # Clean apostrophes
    text = text.replace("'", "")

    return text

print("=" * 60)
print("V3 FERRARI UNLEASHED")
print("=" * 60)
print("\nLoading checkpoint_60000...")

synthesizer = Synthesizer(
    tts_checkpoint=checkpoint_path,
    tts_config_path=config_path,
    use_cuda=True
)
print("Model loaded!")

def generate(text, filename, breath, speed, noise_clean=0.7):
    """Generate with unleash parameters"""
    ipa = uzbek_to_ipa(text)
    print(f"\n{'='*50}")
    print(f"File: {filename}")
    print(f"Uzbek: {text}")
    print(f"IPA:   {ipa}")
    print(f"Breath (noise_scale): {breath}")
    print(f"Speed (length_scale): {speed}")

    # === THE MAGIC KNOBS ===
    wav = synthesizer.tts(
        ipa,
        length_scale=speed,      # 0.9 = Faster (Less hesitation)
        noise_scale=breath,      # 0.8+ = More "Air" (Fixes "holding back")
        noise_scale_w=0.9        # Natural Flow
    )

    # === STUDIO CLEANING ===
    wav_np = np.array(wav, dtype=np.float32)

    # Noise reduction - kills the static
    clean_wav = nr.reduce_noise(
        y=wav_np,
        sr=22050,
        prop_decrease=noise_clean,  # 0.7 = Strong cleaning
        stationary=True
    )

    # Volume normalization
    max_val = np.max(np.abs(clean_wav))
    if max_val > 0:
        clean_wav = clean_wav * (0.95 / max_val)

    filepath = os.path.join(AUDIO_DIR, filename)
    sf.write(filepath, clean_wav, 22050)
    duration = len(clean_wav) / 22050
    print(f"Saved: {filename} ({duration:.1f}s)")

# === TEST SENTENCES ===
# Contains G' (ʁ), O' (ø), X (χ) - the special Uzbek sounds
test1 = "G'ozlar g'ovur-g'uvur qilib, o'tloqda yurishibdi."
test2 = "Xorazm viloyatida g'o'za yetishtirish juda rivojlangan."
test3 = "Men o'zbek tilida gaplashaman va bu tilni yaxshi ko'raman."

print("\n" + "=" * 60)
print("GENERATING TEST SAMPLES")
print("=" * 60)

# === TEST 1: SAFE (Default - might feel held back) ===
generate(test1, "v3_safe.wav", breath=0.667, speed=1.0)

# === TEST 2: RELEASED (Target naturalness) ===
generate(test1, "v3_released.wav", breath=0.8, speed=0.9)

# === TEST 3: MAX ENERGY (Very free, might be too airy) ===
generate(test1, "v3_energy.wav", breath=0.9, speed=0.95)

# === TEST 4: BEST GUESS (Sweet spot) ===
generate(test1, "v3_sweet.wav", breath=0.85, speed=0.92)

# === BONUS: Test other sentences with "released" settings ===
print("\n" + "=" * 60)
print("BONUS TESTS (Released settings)")
print("=" * 60)

generate(test2, "v3_xorazm_released.wav", breath=0.8, speed=0.9)
generate(test3, "v3_uzbek_released.wav", breath=0.8, speed=0.9)

print("\n" + "=" * 60)
print("UNLEASH COMPLETE!")
print("=" * 60)
print("\nListen and compare:")
print("  v3_safe.wav      - Default (might feel held back)")
print("  v3_released.wav  - Released (target naturalness)")
print("  v3_energy.wav    - Max energy (very free)")
print("  v3_sweet.wav     - Sweet spot guess")
print("\nFocus on:")
print("  - Does 'released' feel more natural?")
print("  - Is the G' (ʁ) sound correct?")
print("  - Is there still noise after cleaning?")
