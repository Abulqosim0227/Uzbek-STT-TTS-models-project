#!/usr/bin/env python3
"""
V3 FERRARI ENHANCED - With Full Text Preprocessing
By Abulqosim Rafiqov - December 2025

Features:
- Auto English word detection and transliteration
- Individual letter pronunciation (name or sound mode)
- Proper punctuation handling
- Number to Uzbek conversion
- Noise reduction and volume normalization
"""

import torch
from TTS.utils.synthesizer import Synthesizer
import soundfile as sf
import noisereduce as nr
import numpy as np
import os

# Import the new preprocessor
from text_preprocessor import UzbekTextPreprocessor, is_english_word

# === CONFIG ===
CHECKPOINT_PATH = "/mnt/c/Users/Admin/Desktop/voice_dataset/ISSAI_USC/uzbek_tts_project/training_output_ipa_v3/uzbek_tts_ipa_v3-December-12-2025_09+25AM-0000000/checkpoint_60000.pth"
CONFIG_PATH = "/mnt/c/Users/Admin/Desktop/voice_dataset/ISSAI_USC/uzbek_tts_project/training_output_ipa_v3/uzbek_tts_ipa_v3-December-12-2025_09+25AM-0000000/config.json"

# === OUTPUT FOLDER ===
AUDIO_DIR = "/mnt/c/Users/Admin/Desktop/voice_dataset/ISSAI_USC/uzbek_tts_project/audio"
os.makedirs(AUDIO_DIR, exist_ok=True)


class UzbekTTSv3:
    """
    Enhanced Uzbek TTS with full text preprocessing
    """

    def __init__(self, use_cuda: bool = True, letter_mode: str = "name"):
        """
        Initialize TTS engine

        Args:
            use_cuda: Use GPU acceleration
            letter_mode: "name" for letter names (G' -> g'e), "sound" for just sounds
        """
        print("=" * 60)
        print("V3 FERRARI ENHANCED - Loading...")
        print("=" * 60)

        self.synthesizer = Synthesizer(
            tts_checkpoint=CHECKPOINT_PATH,
            tts_config_path=CONFIG_PATH,
            use_cuda=use_cuda
        )

        self.preprocessor = UzbekTextPreprocessor(
            detect_english=True,
            transliterate_english=True,
            convert_numbers=True,
            letter_mode=letter_mode,
            convert_to_ipa=True,
            preserve_punctuation=True
        )

        self.sample_rate = 22050
        print("Model loaded successfully!")

    def speak(
        self,
        text: str,
        output_file: str = "output.wav",
        # Voice parameters - Slower, Steady, Natural
        breath: float = 0.82,       # noise_scale (0.6-1.0, higher = more natural)
        speed: float = 1.08,        # length_scale (0.8-1.2, higher = slower)
        flow: float = 0.88,         # noise_scale_w (steady flow)
        # Post-processing
        noise_reduction: float = 0.7,
        normalize_volume: bool = True,
        # Letter mode override
        letter_mode: str = None,
        # Debug
        verbose: bool = True
    ) -> np.ndarray:
        """
        Generate speech from Uzbek text

        Args:
            text: Input text (Uzbek or mixed with English)
            output_file: Output WAV file path
            breath: Naturalness (higher = more breath/air in voice)
            speed: Speech speed (lower = faster)
            flow: Voice flow naturalness
            noise_reduction: Noise reduction strength (0-1)
            normalize_volume: Normalize output volume
            letter_mode: Override letter pronunciation mode ("name" or "sound")
            verbose: Print debug info

        Returns:
            Audio waveform as numpy array
        """
        if letter_mode:
            self.preprocessor.letter_mode = letter_mode

        # Preprocess text
        if verbose:
            print(f"\n{'='*50}")
            print(f"Original:  {text}")

        processed = self.preprocessor.preprocess(text)

        if verbose:
            print(f"Processed: {processed}")
            print(f"Params: breath={breath}, speed={speed}, flow={flow}")

        # Generate audio
        wav = self.synthesizer.tts(
            processed,
            length_scale=speed,
            noise_scale=breath,
            noise_scale_w=flow
        )

        wav_np = np.array(wav, dtype=np.float32)

        # Post-processing
        if noise_reduction > 0:
            wav_np = nr.reduce_noise(
                y=wav_np,
                sr=self.sample_rate,
                prop_decrease=noise_reduction,
                stationary=True
            )

        if normalize_volume:
            max_val = np.max(np.abs(wav_np))
            if max_val > 0:
                wav_np = wav_np * (0.95 / max_val)

        # Save
        filepath = os.path.join(AUDIO_DIR, output_file) if not os.path.isabs(output_file) else output_file
        sf.write(filepath, wav_np, self.sample_rate)

        duration = len(wav_np) / self.sample_rate
        if verbose:
            print(f"Saved: {output_file} ({duration:.1f}s)")

        return wav_np

    def speak_batch(
        self,
        texts: list,
        output_prefix: str = "batch",
        **kwargs
    ) -> list:
        """Generate speech for multiple texts"""
        results = []
        for i, text in enumerate(texts):
            output_file = f"{output_prefix}_{i+1:03d}.wav"
            wav = self.speak(text, output_file, **kwargs)
            results.append(wav)
        return results


# === QUICK FUNCTIONS ===

def quick_speak(text: str, output: str = "quick_output.wav", **kwargs):
    """Quick one-liner TTS"""
    tts = UzbekTTSv3(use_cuda=True)
    return tts.speak(text, output, **kwargs)


# === DEMO ===

if __name__ == "__main__":
    tts = UzbekTTSv3(use_cuda=True, letter_mode="name")

    print("\n" + "=" * 70)
    print("TEST 1: ENGLISH WORDS AUTO-DETECTION")
    print("=" * 70)

    english_tests = [
        "Men Python va JavaScript dasturlash tillarini o'rganmoqdaman.",
        "Facebook, Instagram va WhatsApp ilovalarini har kuni ishlataman.",
        "Machine learning va artificial intelligence juda qiziqarli.",
        "Bu website juda yaxshi design qilingan.",
    ]

    for i, text in enumerate(english_tests, 1):
        tts.speak(text, f"test_english_{i}.wav")

    print("\n" + "=" * 70)
    print("TEST 2: INDIVIDUAL LETTERS (NAME MODE)")
    print("=" * 70)

    letter_tests = [
        "G', X, L, P harflarini aytib bering.",
        "O'zbek alifbosida: A, B, D, E, F harflari bor.",
        "Maxsus harflar: G', O', Sh, Ch.",
    ]

    for i, text in enumerate(letter_tests, 1):
        tts.speak(text, f"test_letters_name_{i}.wav", letter_mode="name")

    print("\n" + "=" * 70)
    print("TEST 3: INDIVIDUAL LETTERS (SOUND MODE)")
    print("=" * 70)

    for i, text in enumerate(letter_tests, 1):
        tts.speak(text, f"test_letters_sound_{i}.wav", letter_mode="sound")

    print("\n" + "=" * 70)
    print("TEST 4: NUMBERS AND DATES")
    print("=" * 70)

    number_tests = [
        "Bugun 2024 yil 19-dekabr.",
        "Men 25 yoshdaman va 100% ishonchim bor.",
        "2023 yilda 50 ming dollar ishladim.",
    ]

    for i, text in enumerate(number_tests, 1):
        tts.speak(text, f"test_numbers_{i}.wav")

    print("\n" + "=" * 70)
    print("TEST 5: PUNCTUATION AND INTONATION")
    print("=" * 70)

    punctuation_tests = [
        "Salom! Qalaysiz? Yaxshimisiz.",
        "Ertaga, albatta, kelaman.",
        "Bu juda muhim: e'tibor bering!",
    ]

    for i, text in enumerate(punctuation_tests, 1):
        tts.speak(text, f"test_punctuation_{i}.wav")

    print("\n" + "=" * 70)
    print("TEST 6: COMPLEX MIXED TEXT")
    print("=" * 70)

    complex_tests = [
        "Men 2023 yilda YouTube kanalimda Python va JavaScript darslarini boshladim. G', X, L harflarini to'g'ri talaffuz qilish muhim!",
        "WhatsApp orqali 100% bepul xabar yuboring. Machine learning 2024 yilda juda mashhur.",
    ]

    for i, text in enumerate(complex_tests, 1):
        tts.speak(text, f"test_complex_{i}.wav")

    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETE!")
    print("=" * 70)
    print(f"\nAudio files saved to: {AUDIO_DIR}")
    print("\nKey improvements:")
    print("  1. English words auto-detected and transliterated")
    print("  2. Individual letters pronounced correctly")
    print("  3. Numbers converted to Uzbek text")
    print("  4. Punctuation preserved for natural pauses")
