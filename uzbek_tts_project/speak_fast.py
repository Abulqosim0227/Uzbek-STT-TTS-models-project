#!/usr/bin/env python3
"""Generate TTS with adjustable speed (length_scale)"""

from TTS.utils.synthesizer import Synthesizer
import sys

# Paths
MODEL_PATH = "training_output/uzbek_tts_single_speaker-December-02-2025_09+01AM-0000000/best_model.pth"
CONFIG_PATH = "training_output/uzbek_tts_single_speaker-December-02-2025_09+01AM-0000000/config.json"

# Speed: 0.9 = faster (less hesitation), 1.0 = normal, 1.1 = slower
LENGTH_SCALE = 0.9

# Text to speak
text = "Salom, men o'zbek tilida gaplashaman. Bugun havo juda yaxshi."

# Output file
output_file = "test_sample_fast.wav"

print(f"Loading model...")
synthesizer = Synthesizer(
    tts_checkpoint=MODEL_PATH,
    tts_config_path=CONFIG_PATH,
    use_cuda=True
)

print(f"Generating speech with length_scale={LENGTH_SCALE}...")
wav = synthesizer.tts(text, length_scale=LENGTH_SCALE)

print(f"Saving to {output_file}...")
synthesizer.save_wav(wav, output_file)

print(f"Done! File saved: {output_file}")
