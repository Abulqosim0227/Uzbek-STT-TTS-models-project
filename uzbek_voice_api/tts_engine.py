#!/usr/bin/env python3
"""
TTS Engine - Uzbek Text-to-Speech
Supports V1 (plain text) and V3 (IPA) models
"""

import torch
from TTS.utils.synthesizer import Synthesizer
import soundfile as sf
import noisereduce as nr
import numpy as np
import re
import glob

class TTSEngine:
    """Uzbek TTS Engine - supports V1 and V3 models"""

    # V1 Model (540k steps) - Plain text
    V1_MODEL = "/mnt/c/Users/Admin/Desktop/voice_dataset/ISSAI_USC/uzbek_tts_project/training_output/uzbek_tts_single_speaker-December-04-2025_08+50AM-0000000/checkpoint_540000.pth"
    V1_CONFIG = "/mnt/c/Users/Admin/Desktop/voice_dataset/ISSAI_USC/uzbek_tts_project/training_output/uzbek_tts_single_speaker-December-04-2025_08+50AM-0000000/config.json"

    # V3 Model (IPA) - Get latest checkpoint
    V3_DIR = "/mnt/c/Users/Admin/Desktop/voice_dataset/ISSAI_USC/uzbek_tts_project/training_output_ipa_v3/uzbek_tts_ipa_v3-December-12-2025_09+25AM-0000000"

    # Optimized parameters - NATURAL PACE
    NOISE_SCALE = 0.75       # Balanced natural voice
    NOISE_SCALE_W = 0.5      # Lower = more consistent pace throughout

    # Separate speed settings for each model
    V1_LENGTH_SCALE = 1.0    # V1 normal speed
    V3_LENGTH_SCALE = 3.0    # V3 much slower (IPA model speaks very fast)

    def __init__(self, model_version="v1"):
        self.model_version = model_version
        self.synthesizer = None
        self.load_model(model_version)

    def load_model(self, version):
        """Load V1 or V3 model"""
        self.model_version = version

        if version == "v3":
            # Get latest V3 checkpoint
            checkpoints = glob.glob(f"{self.V3_DIR}/checkpoint_*.pth")
            if checkpoints:
                model_path = sorted(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))[-1]
                config_path = f"{self.V3_DIR}/config.json"
                print(f"[TTS] Loading V3 (IPA): {model_path.split('/')[-1]}")
            else:
                print("[TTS] V3 not found, falling back to V1")
                version = "v1"

        if version == "v1":
            model_path = self.V1_MODEL
            config_path = self.V1_CONFIG
            print("[TTS] Loading V1 (plain text): checkpoint_540000.pth")

        self.synthesizer = Synthesizer(
            tts_checkpoint=model_path,
            tts_config_path=config_path,
            use_cuda=torch.cuda.is_available()
        )
        self.model_version = version
        print(f"[TTS] {version.upper()} loaded! CUDA: {torch.cuda.is_available()}")

    # Number to Uzbek words
    ONES = ['', 'bir', 'ikki', 'uch', 'to\'rt', 'besh', 'olti', 'yetti', 'sakkiz', 'to\'qqiz']
    TENS = ['', 'o\'n', 'yigirma', 'o\'ttiz', 'qirq', 'ellik', 'oltmish', 'yetmish', 'sakson', 'to\'qson']

    # English tech words → Uzbek pronunciation
    TECH_WORDS = {
        'sql': 'es kyu el',
        'html': 'eych ti em el',
        'css': 'si es es',
        'js': 'jey es',
        'api': 'ey pi ay',
        'url': 'yu ar el',
        'http': 'eych ti ti pi',
        'https': 'eych ti ti pi es',
        'json': 'jeyson',
        'xml': 'eks em el',
        'php': 'pi eych pi',
        'node': 'nod',
        'redis': 'redis',
        'mysql': 'may es kyu el',
        'mongodb': 'mongo di bi',
        'python': 'payton',
        'java': 'java',
        'react': 'riekt',
        'vue': 'vyu',
        'angular': 'angyular',
        'docker': 'doker',
        'linux': 'linuks',
        'windows': 'vindovs',
        'github': 'git hab',
        'git': 'git',
        'npm': 'en pi em',
        'ai': 'ey ay',
        'ml': 'em el',
        'gpt': 'ji pi ti',
        'llm': 'el el em',
        'cpu': 'si pi yu',
        'gpu': 'ji pi yu',
        'ram': 'ram',
        'ssd': 'es es di',
        'usb': 'yu es bi',
        'wifi': 'vay fay',
        'ios': 'ay o es',
        'android': 'android',
    }

    def number_to_uzbek(self, n: int) -> str:
        """Convert number to Uzbek words"""
        if n == 0:
            return 'nol'
        if n < 0:
            return 'minus ' + self.number_to_uzbek(-n)

        result = []

        # Billions
        if n >= 1000000000:
            result.append(self.number_to_uzbek(n // 1000000000) + ' milliard')
            n %= 1000000000

        # Millions
        if n >= 1000000:
            result.append(self.number_to_uzbek(n // 1000000) + ' million')
            n %= 1000000

        # Thousands
        if n >= 1000:
            thousands = n // 1000
            if thousands == 1:
                result.append('ming')
            else:
                result.append(self.number_to_uzbek(thousands) + ' ming')
            n %= 1000

        # Hundreds
        if n >= 100:
            hundreds = n // 100
            if hundreds == 1:
                result.append('yuz')
            else:
                result.append(self.ONES[hundreds] + ' yuz')
            n %= 100

        # Tens and ones
        if n >= 10:
            result.append(self.TENS[n // 10])
            n %= 10

        if n > 0:
            result.append(self.ONES[n])

        return ' '.join(result)

    def preprocess_text(self, text: str) -> str:
        """Preprocess text: convert numbers and tech words"""
        # Convert tech words (case insensitive)
        words = text.split()
        processed = []

        for word in words:
            # Check for punctuation at end
            punct = ''
            clean_word = word.lower()
            if clean_word and clean_word[-1] in '.,!?;:':
                punct = clean_word[-1]
                clean_word = clean_word[:-1]

            # Check if it's a tech word
            if clean_word in self.TECH_WORDS:
                processed.append(self.TECH_WORDS[clean_word] + punct)
            # Check if it's a number
            elif clean_word.isdigit():
                processed.append(self.number_to_uzbek(int(clean_word)) + punct)
            # Check for year-like numbers (e.g., 1970, 2024)
            elif re.match(r'^\d{4}$', clean_word):
                processed.append(self.number_to_uzbek(int(clean_word)) + punct)
            else:
                processed.append(word)

        return ' '.join(processed)

    def clean_text(self, text: str) -> str:
        """Clean and prepare text for synthesis"""
        # Preprocess numbers and tech words
        text = self.preprocess_text(text)

        # Normalize apostrophes
        text = text.replace("'", "'").replace("'", "'").replace("`", "'")

        # Ensure sentence endings have punctuation
        if text and text[-1] not in '.!?':
            text += '.'

        # Clean multiple spaces
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def to_ipa(self, text: str) -> str:
        """Convert Uzbek text to IPA for V3 model"""
        text = text.lower()

        # Normalize apostrophes
        for apo in ["'", "'", "`", "ʻ", "ʼ"]:
            text = text.replace(apo, "'")

        # Multi-char replacements first (order matters!)
        text = text.replace("o'", "ø")
        text = text.replace("g'", "ʁ")
        text = text.replace("sh", "ʃ")
        text = text.replace("ch", "tʃ")
        text = text.replace("ng", "ŋ")

        # Single char replacements
        text = text.replace("x", "χ")
        text = text.replace("j", "dʒ")
        text = text.replace("y", "j")
        text = text.replace("a", "ɑ")
        text = text.replace("o", "ɔ")

        # Clean remaining apostrophes
        text = text.replace("'", "")

        return text

    def synthesize(self, text: str, output_path: str, speed: float = 1.0) -> float:
        """
        Synthesize speech from text

        Args:
            text: Uzbek text to synthesize
            output_path: Path to save WAV file
            speed: Speed multiplier (1.0 = normal)

        Returns:
            Duration in seconds
        """
        # Clean text
        text = self.clean_text(text)

        # Convert to IPA if using V3
        if self.model_version == "v3":
            text = self.to_ipa(text)

        # Adjust length scale for speed (V3 needs slower speed)
        base_scale = self.V3_LENGTH_SCALE if self.model_version == "v3" else self.V1_LENGTH_SCALE
        length_scale = base_scale / speed

        print(f"[TTS DEBUG] Model: {self.model_version}, base_scale: {base_scale}, length_scale: {length_scale}")

        # Generate audio
        wav = self.synthesizer.tts(
            text,
            noise_scale=self.NOISE_SCALE,
            noise_scale_w=self.NOISE_SCALE_W,
            length_scale=length_scale
        )

        wav_np = np.array(wav, dtype=np.float32)

        # Add small silence padding at the end to prevent cutoff
        padding = np.zeros(int(22050 * 0.2))  # 200ms silence
        wav_np = np.concatenate([wav_np, padding])

        # Noise reduction - reduced strength to preserve quiet endings
        clean_wav = nr.reduce_noise(
            y=wav_np,
            sr=22050,
            prop_decrease=0.3,  # Reduced from 0.5 to preserve endings
            stationary=True
        )

        # Apply fade-out to last 100ms to smooth the ending
        fade_samples = int(22050 * 0.1)  # 100ms
        fade_out = np.linspace(1, 0, fade_samples)
        clean_wav[-fade_samples:] = clean_wav[-fade_samples:] * fade_out

        # Volume normalization
        max_val = np.max(np.abs(clean_wav))
        if max_val > 0:
            clean_wav = clean_wav * (0.95 / max_val)

        # Save
        sf.write(output_path, clean_wav, 22050)

        duration = len(clean_wav) / 22050
        return round(duration, 2)

    def get_version(self) -> str:
        """Get current model version"""
        return self.model_version


# Test
if __name__ == "__main__":
    engine = TTSEngine()
    duration = engine.synthesize(
        "Salom! Men O'zbek tilida gaplashaman.",
        "test_tts.wav"
    )
    print(f"Generated: test_tts.wav ({duration}s)")
