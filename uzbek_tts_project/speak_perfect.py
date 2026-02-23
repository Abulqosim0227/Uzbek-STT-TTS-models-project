#!/usr/bin/env python3
"""
Uzbek TTS - Professional Production Script
By Abulqosim Rafiqov

This script includes:
- Text cleaning and normalization
- Phonetic tricks for foreign words
- Number to text conversion
- Punctuation optimization
- Noise reduction and volume normalization
"""

import torch
from TTS.utils.synthesizer import Synthesizer
import soundfile as sf
import noisereduce as nr
import numpy as np
import re

# === CONFIG ===
MODEL_PATH = "training_output/uzbek_tts_single_speaker-December-04-2025_08+50AM-0000000/checkpoint_540000.pth"
CONFIG_PATH = "training_output/uzbek_tts_single_speaker-December-04-2025_08+50AM-0000000/config.json"

# === NUMBER TO UZBEK TEXT ===
UZBEK_NUMBERS = {
    0: "nol", 1: "bir", 2: "ikki", 3: "uch", 4: "to'rt",
    5: "besh", 6: "olti", 7: "yetti", 8: "sakkiz", 9: "to'qqiz",
    10: "o'n", 11: "o'n bir", 12: "o'n ikki", 13: "o'n uch",
    14: "o'n to'rt", 15: "o'n besh", 16: "o'n olti", 17: "o'n yetti",
    18: "o'n sakkiz", 19: "o'n to'qqiz", 20: "yigirma",
    30: "o'ttiz", 40: "qirq", 50: "ellik", 60: "oltmish",
    70: "yetmish", 80: "sakson", 90: "to'qson",
    100: "yuz", 1000: "ming", 1000000: "million", 1000000000: "milliard"
}

def number_to_uzbek(n, ordinal=False):
    """Convert a number to Uzbek text

    Args:
        n: The number to convert
        ordinal: If True, add "inchi/nchi" suffix (for years, dates)
    """
    if n < 0:
        return "minus " + number_to_uzbek(-n)

    result = ""

    if n < 20:
        result = UZBEK_NUMBERS.get(n, str(n))
    elif n < 100:
        tens, ones = divmod(n, 10)
        if ones == 0:
            result = UZBEK_NUMBERS[tens * 10]
        else:
            result = UZBEK_NUMBERS[tens * 10] + " " + UZBEK_NUMBERS[ones]
    elif n < 1000:
        hundreds, remainder = divmod(n, 100)
        if remainder == 0:
            result = (UZBEK_NUMBERS[hundreds] + " " if hundreds > 1 else "") + "yuz"
        else:
            result = (UZBEK_NUMBERS[hundreds] + " " if hundreds > 1 else "") + "yuz " + number_to_uzbek(remainder)
    elif n < 1000000:
        thousands, remainder = divmod(n, 1000)
        if remainder == 0:
            result = number_to_uzbek(thousands) + " ming"
        else:
            result = number_to_uzbek(thousands) + " ming " + number_to_uzbek(remainder)
    elif n < 1000000000:
        millions, remainder = divmod(n, 1000000)
        if remainder == 0:
            result = number_to_uzbek(millions) + " million"
        else:
            result = number_to_uzbek(millions) + " million " + number_to_uzbek(remainder)
    else:
        result = str(n)

    # Add ordinal suffix if requested
    if ordinal and result:
        # Uzbek ordinal rules:
        # - ends in vowel (a, i, o, u, e) -> "nchi"
        # - ends in consonant -> "inchi"
        last_char = result[-1].lower()
        vowels = "aeiouoʻ'"
        if last_char in vowels:
            result += "nchi"
        else:
            result += "inchi"

    return result

def convert_numbers_in_text(text):
    """Find and convert all numbers in text to Uzbek words"""

    # 1. First handle YEARS with ORDINAL: "2002 yil" or "2002 yilda" -> ordinal
    # Only yil, yilda, yili are ordinal (2002-chi yilda)
    # yildan, yilga, yilgacha are NOT ordinal (2750 yildan = from 2750 years)
    def replace_year_ordinal(match):
        num = int(match.group(1))
        suffix = match.group(2)  # "yil" or "yilda" or "yili"
        return number_to_uzbek(num, ordinal=True) + " " + suffix

    text = re.sub(r'\b(\d{4})\s*(yil|yilda|yili)\b', replace_year_ordinal, text)

    # 2. Handle years WITHOUT ordinal: "2750 yildan" = from 2750 years (NOT ordinal)
    def replace_year_regular(match):
        num = int(match.group(1))
        suffix = match.group(2)
        return number_to_uzbek(num, ordinal=False) + " " + suffix

    text = re.sub(r'\b(\d+)\s*(yildan|yilga|yilgacha|yillik)\b', replace_year_regular, text)

    # 2. Handle dates: "6-dekabr" -> "oltinchi dekabr"
    def replace_date(match):
        num = int(match.group(1))
        month = match.group(2)
        return number_to_uzbek(num, ordinal=True) + " " + month

    text = re.sub(r'\b(\d{1,2})-(yanvar|fevral|mart|aprel|may|iyun|iyul|avgust|sentabr|oktabr|noyabr|dekabr)\b',
                  replace_date, text, flags=re.IGNORECASE)

    # 3. Handle ordinals with -chi suffix: "1-chi" -> "birinchi"
    def replace_ordinal_chi(match):
        num = int(match.group(1))
        return number_to_uzbek(num, ordinal=True)

    text = re.sub(r'\b(\d+)-?(chi|nchi|inchi)\b', replace_ordinal_chi, text, flags=re.IGNORECASE)

    # 4. Regular numbers (non-ordinal)
    def replace_number(match):
        num = int(match.group())
        return number_to_uzbek(num, ordinal=False)

    text = re.sub(r'\b\d+\b', replace_number, text)

    return text

# === PHONETIC DICTIONARY (Foreign words) ===
PHONETIC_MAP = {
    # English Tech Words
    "Python": "Payton",
    "python": "payton",
    "JavaScript": "Javaskript",
    "javascript": "javaskript",
    "Facebook": "Feysbuk",
    "facebook": "feysbuk",
    "Instagram": "Instagramm",
    "YouTube": "Yutub",
    "youtube": "yutub",
    "Google": "Gugl",
    "google": "gugl",
    "iPhone": "Ayfon",
    "iphone": "ayfon",
    "Samsung": "Samsang",
    "WhatsApp": "Votsap",
    "Telegram": "Telegramm",
    "Twitter": "Tvitter",
    "TikTok": "Tiktok",
    "Android": "Android",
    "Windows": "Vindovs",
    "Linux": "Linuks",
    "Microsoft": "Maykrosoft",
    "Apple": "Eppl",
    "Amazon": "Amazon",
    "Netflix": "Netfliks",
    "Spotify": "Spotifay",
    "Uber": "Uber",
    "Tesla": "Tesla",
    "Bitcoin": "Bitkoin",
    "AI": "ey-ay",
    "OK": "okey",
    "ok": "okey",
    "Hello": "Xello",
    "hello": "xello",
    "Hi": "Xay",
    "WiFi": "Vay-fay",
    "wifi": "vay-fay",
    "USB": "yu-es-bi",
    "PDF": "pi-di-ef",
    "SMS": "es-em-es",
    "GPS": "ji-pi-es",
    "ATM": "ey-ti-em",
    "CEO": "si-i-ou",
    "IT": "ay-ti",

    # Common English words
    "computer": "kompyuter",
    "laptop": "laptop",
    "internet": "internet",
    "online": "onlayn",
    "offline": "offlayn",
    "email": "imeil",
    "website": "vebsayt",
    "software": "softver",
    "hardware": "xardver",
    "download": "download",
    "upload": "upload",
    "startup": "startap",
    "business": "biznes",
    "marketing": "marketing",
    "manager": "menejer",
    "project": "proyekt",
    "design": "dizayn",
    "developer": "developer",
    "programmer": "programmist",
}

def apply_phonetic_map(text):
    """Replace foreign words with phonetic Uzbek equivalents"""
    for eng, uzb in PHONETIC_MAP.items():
        text = text.replace(eng, uzb)
    return text

# === TEXT CLEANER ===
def clean_uzbek_text(text):
    """
    Professional text cleaner for Uzbek TTS
    """
    # 1. Standardize apostrophes (crucial for Uzbek o', g', etc.)
    text = text.replace("'", "'").replace("'", "'").replace("`", "'").replace("ʻ", "'").replace("ʼ", "'")

    # 2. Apply phonetic map for foreign words
    text = apply_phonetic_map(text)

    # 3. Convert numbers to Uzbek words
    text = convert_numbers_in_text(text)

    # 4. Expand common abbreviations
    text = text.replace(" v.h.", " va hokazo")
    text = text.replace(" h.k.", " hamda boshqalar")
    text = text.replace(" va b.", " va boshqalar")
    text = text.replace(" t.b.", " to'g'ri bo'lsa")
    text = text.replace("masalan,", "masalan")
    text = text.replace("ya'ni,", "ya'ni")

    # 5. Handle percentages (BEFORE number conversion!)
    text = re.sub(r'(\d+)\s*%', lambda m: number_to_uzbek(int(m.group(1))) + " foiz", text)

    # 6. Remove any remaining % symbols
    text = text.replace("%", " foiz")

    # 6. Clean up multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# === MAIN TTS CLASS ===
class UzbekTTS:
    def __init__(self, use_cuda=True):
        print("Loading Uzbek TTS model...")
        self.synthesizer = Synthesizer(
            tts_checkpoint=MODEL_PATH,
            tts_config_path=CONFIG_PATH,
            use_cuda=use_cuda
        )
        self.sample_rate = 22050
        print("Model loaded!")

    def speak(
        self,
        text,
        output_file="output.wav",
        # Voice style presets
        style="natural",  # "natural", "clear", "expressive", "slow"
        # Or manual control
        noise_scale=None,
        noise_scale_w=None,
        length_scale=None,
        # Post-processing
        noise_reduction=True,
        normalize_volume=True
    ):
        """
        Generate speech from Uzbek text

        Styles:
        - natural: Balanced, human-like (default)
        - clear: Slower, clearer pronunciation
        - expressive: More emotional variation
        - slow: Very clear, for announcements
        """

        # Style presets
        STYLES = {
            "natural": {"noise_scale": 0.8, "noise_scale_w": 0.9, "length_scale": 1.0},
            "clear": {"noise_scale": 0.75, "noise_scale_w": 0.85, "length_scale": 1.05},
            "expressive": {"noise_scale": 0.85, "noise_scale_w": 0.95, "length_scale": 0.95},
            "slow": {"noise_scale": 0.7, "noise_scale_w": 0.8, "length_scale": 1.15},
            "fast": {"noise_scale": 0.8, "noise_scale_w": 0.9, "length_scale": 0.9},
        }

        # Get style settings or use manual values
        style_settings = STYLES.get(style, STYLES["natural"])
        ns = noise_scale if noise_scale is not None else style_settings["noise_scale"]
        nsw = noise_scale_w if noise_scale_w is not None else style_settings["noise_scale_w"]
        ls = length_scale if length_scale is not None else style_settings["length_scale"]

        # Clean text
        print(f"Original: {text}")
        clean_text = clean_uzbek_text(text)
        print(f"Cleaned:  {clean_text}")
        print(f"Style: {style} (ns={ns}, nsw={nsw}, ls={ls})")

        # Generate
        wav = self.synthesizer.tts(
            clean_text,
            noise_scale=ns,
            noise_scale_w=nsw,
            length_scale=ls
        )

        wav_np = np.array(wav, dtype=np.float32)

        # Noise reduction
        if noise_reduction:
            wav_np = nr.reduce_noise(
                y=wav_np,
                sr=self.sample_rate,
                prop_decrease=0.6,
                stationary=True
            )

        # Volume normalization to -3dB
        if normalize_volume:
            rms = np.sqrt(np.mean(wav_np**2))
            if rms > 0:
                target_rms = 0.316
                max_scale = 0.99 / np.max(np.abs(wav_np))
                scale = min(target_rms / rms, max_scale)
                wav_np = wav_np * scale

        # Save
        sf.write(output_file, wav_np, self.sample_rate)
        duration = len(wav_np) / self.sample_rate
        print(f"Saved: {output_file} ({duration:.1f}s)")

        return wav_np


# === DEMO / TEST ===
if __name__ == "__main__":
    tts = UzbekTTS(use_cuda=True)

    print("\n" + "="*60)
    print("TEST 1: Tone/Punctuation")
    print("="*60)
    tts.speak(
        "Assalomu alaykum! Ishlaringiz yaxshimi? Bugun havo juda ajoyib.",
        "test_tone.wav",
        style="natural"
    )

    print("\n" + "="*60)
    print("TEST 2: Foreign Words (Auto-Phonetic)")
    print("="*60)
    tts.speak(
        "Men kecha Facebook tarmog'ida YouTube video ko'rdim. Python juda zo'r til!",
        "test_foreign.wav",
        style="natural"
    )

    print("\n" + "="*60)
    print("TEST 3: Numbers (Auto-Conversion)")
    print("="*60)
    tts.speak(
        "Mening yoshim 25 da. Men 2022 yilda tug'ilganman. Bu 100% to'g'ri.",
        "test_numbers.wav",
        style="clear"
    )

    print("\n" + "="*60)
    print("TEST 4: Full Demo with Creator Credit")
    print("="*60)
    tts.speak(
        """O'zbekiston juda go'zal mamlakat.
        Samarqand, Buxoro va Xiva shaharlari butun dunyoga mashhur.
        Bugun men sizlarga sun'iy intellekt haqida gapirib bermoqchiman.
        Bu texnologiya kundan kunga rivojlanmoqda.
        Men, Abulqosim Rafiqov tomonidan, yaratilganman.
        Rahmat e'tiboringiz uchun!""",
        "test_full_demo.wav",
        style="natural"
    )

    print("\n" + "="*60)
    print("TEST 5: Different Styles Comparison")
    print("="*60)

    test_sentence = "Salom, men sun'iy intellekt yordamida gaplashyapman."

    for style in ["natural", "clear", "expressive", "slow", "fast"]:
        tts.speak(test_sentence, f"test_style_{style}.wav", style=style)

    print("\n" + "="*60)
    print("ALL TESTS COMPLETE!")
    print("="*60)
