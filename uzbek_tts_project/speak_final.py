#!/usr/bin/env python3
"""
UZBEK TTS - FINAL OPTIMIZED VERSION
Best settings found: Good punctuation + Max release

Usage:
    python speak_final.py "Salom! Bugun ob-havo juda yaxshi."
    python speak_final.py  # Interactive mode
"""

import sys
import torch
from TTS.utils.synthesizer import Synthesizer
import soundfile as sf
import noisereduce as nr
import numpy as np
import os
import re

# === PATHS ===
BASE = "/mnt/c/Users/Admin/Desktop/voice_dataset/ISSAI_USC/uzbek_tts_project"
MODEL = f"{BASE}/training_output/uzbek_tts_single_speaker-December-04-2025_08+50AM-0000000/checkpoint_540000.pth"
CONFIG = f"{BASE}/training_output/uzbek_tts_single_speaker-December-04-2025_08+50AM-0000000/config.json"
AUDIO_DIR = f"{BASE}/audio"

# === BEST SETTINGS (Slower, Steady, Natural) ===
NOISE_SCALE = 0.82      # Balanced = natural without too much air
NOISE_SCALE_W = 0.88    # Controlled = steady flow
LENGTH_SCALE = 1.08     # Higher = slower, clearer speech

# === TEXT CLEANER ===
def clean_text(text):
    """Add punctuation hints for natural pauses"""
    # Normalize apostrophes
    text = text.replace("'", "'").replace("'", "'").replace("`", "'")

    # Ensure sentence endings have punctuation
    if text and text[-1] not in '.!?':
        text += '.'

    # Clean multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# === MAIN TTS FUNCTION ===
class UzbekTTS:
    def __init__(self):
        print("Loading Uzbek TTS (V1 Optimized)...")
        self.synth = Synthesizer(
            tts_checkpoint=MODEL,
            tts_config_path=CONFIG,
            use_cuda=True
        )
        os.makedirs(AUDIO_DIR, exist_ok=True)
        print("Ready!")

    def speak(self, text, filename="output.wav"):
        """Generate speech with optimized settings"""
        text = clean_text(text)
        print(f"\nText: {text}")

        # Generate with max release settings
        wav = self.synth.tts(
            text,
            noise_scale=NOISE_SCALE,
            noise_scale_w=NOISE_SCALE_W,
            length_scale=LENGTH_SCALE
        )

        wav_np = np.array(wav, dtype=np.float32)

        # Noise reduction
        clean = nr.reduce_noise(
            y=wav_np,
            sr=22050,
            prop_decrease=0.5,
            stationary=True
        )

        # Volume normalization
        mx = np.max(np.abs(clean))
        if mx > 0:
            clean = clean * (0.95 / mx)

        # Save
        filepath = os.path.join(AUDIO_DIR, filename)
        sf.write(filepath, clean, 22050)

        duration = len(clean) / 22050
        print(f"Saved: {filepath} ({duration:.1f}s)")

        return filepath

# === MAIN ===
if __name__ == "__main__":
    tts = UzbekTTS()

    if len(sys.argv) > 1:
        # Command line mode
        text = " ".join(sys.argv[1:])
        tts.speak(text, "output.wav")
    else:
        # Interactive mode
        print("\n" + "="*50)
        print("INTERACTIVE MODE")
        print("Type text and press Enter. Type 'exit' to quit.")
        print("="*50)

        count = 1
        while True:
            try:
                text = input("\nText: ").strip()
                if text.lower() == 'exit':
                    break
                if text:
                    tts.speak(text, f"output_{count}.wav")
                    count += 1
            except KeyboardInterrupt:
                break

        print("\nGoodbye!")
