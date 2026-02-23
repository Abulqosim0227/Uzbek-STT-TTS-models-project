#!/usr/bin/env python3
"""
Uzbek Text to IPA Converter
By Abulqosim Rafiqov - December 2025

Converts Uzbek text (Latin script) to IPA phonemes for better TTS.
"""

import re
from typing import Dict, List, Tuple

# Uzbek to IPA mapping
UZBEK_IPA_MAP: Dict[str, str] = {
    # Vowels (Unlilar)
    'a': 'ɑ',
    'e': 'e',
    'i': 'i',
    'o': 'ɔ',
    'u': 'u',
    "o'": 'ø',  # O with modifier

    # Consonants (Undoshlar)
    'b': 'b',
    'd': 'd',
    'f': 'f',
    'g': 'g',
    "g'": 'ʁ',  # G with modifier (voiced uvular fricative)
    'h': 'h',
    'j': 'dʒ',
    'k': 'k',
    'l': 'l',
    'm': 'm',
    'n': 'n',
    'p': 'p',
    'q': 'q',
    'r': 'r',
    's': 's',
    't': 't',
    'v': 'v',
    'x': 'χ',  # Voiceless uvular fricative
    'y': 'j',
    'z': 'z',

    # Digraphs (Ikki harfli)
    'sh': 'ʃ',
    'ch': 'tʃ',
    'ng': 'ŋ',
}

# Special rules for context-dependent pronunciation
SPECIAL_RULES = [
    # Rule: 'n' before 'g' or 'k' becomes velar nasal
    (r'n(?=[gk])', 'ŋ'),
]

# Word stress patterns (O'zbek tilida urg'u odatda oxirgi bo'g'inda)
# For now, we mark primary stress with 'ˈ' before stressed syllable


class UzbekIPAConverter:
    """Convert Uzbek text to IPA phonemes"""

    def __init__(self):
        self.ipa_map = UZBEK_IPA_MAP.copy()
        # Sort by length (longest first) for proper matching
        self.sorted_keys = sorted(self.ipa_map.keys(), key=len, reverse=True)

    def preprocess(self, text: str) -> str:
        """Normalize text before conversion"""
        text = text.lower().strip()
        # Replace special characters
        text = text.replace("\u2018", "'")  # Left single quote
        text = text.replace("\u2019", "'")  # Right single quote
        text = text.replace("\u02BB", "'")  # Modifier letter
        text = text.replace("`", "'")
        return text

    def apply_special_rules(self, text: str) -> str:
        """Apply context-dependent pronunciation rules"""
        for pattern, replacement in SPECIAL_RULES:
            text = re.sub(pattern, replacement, text)
        return text

    def text_to_ipa(self, text: str, add_stress: bool = False) -> str:
        """
        Convert Uzbek text to IPA

        Args:
            text: Uzbek text in Latin script
            add_stress: Add stress markers (experimental)

        Returns:
            IPA transcription
        """
        text = self.preprocess(text)

        # Process word by word
        words = text.split()
        ipa_words = []

        for word in words:
            ipa_word = self._convert_word(word)
            if add_stress and len(ipa_word) > 2:
                # Simple stress: mark last syllable
                ipa_word = self._add_stress(ipa_word)
            ipa_words.append(ipa_word)

        return ' '.join(ipa_words)

    def _convert_word(self, word: str) -> str:
        """Convert a single word to IPA"""
        result = []
        i = 0

        while i < len(word):
            matched = False

            # Try to match longest sequences first
            for key in self.sorted_keys:
                if word[i:].startswith(key):
                    result.append(self.ipa_map[key])
                    i += len(key)
                    matched = True
                    break

            if not matched:
                char = word[i]
                if char.isalpha():
                    # Unknown letter - keep as is
                    result.append(char)
                elif char in ".,!?;:":
                    # Punctuation - add pause marker
                    result.append('|')
                # Skip other characters
                i += 1

        return ''.join(result)

    def _add_stress(self, ipa: str) -> str:
        """Add stress marker to IPA (simplified)"""
        # Find vowels
        vowels = 'ɑeiɔuø'
        vowel_positions = [i for i, c in enumerate(ipa) if c in vowels]

        if len(vowel_positions) >= 2:
            # Stress on last syllable (common in Uzbek)
            last_vowel = vowel_positions[-1]
            # Find start of syllable
            start = last_vowel
            while start > 0 and ipa[start-1] not in vowels:
                start -= 1
            ipa = ipa[:start] + 'ˈ' + ipa[start:]

        return ipa

    def batch_convert(self, texts: List[str]) -> List[str]:
        """Convert multiple texts to IPA"""
        return [self.text_to_ipa(text) for text in texts]


def create_ipa_lexicon(vocab_file: str, output_file: str):
    """
    Create IPA lexicon from vocabulary file

    Args:
        vocab_file: File with words (one per line)
        output_file: Output file for word|IPA pairs
    """
    converter = UzbekIPAConverter()

    with open(vocab_file, 'r', encoding='utf-8') as f:
        words = [line.strip() for line in f if line.strip()]

    with open(output_file, 'w', encoding='utf-8') as f:
        for word in words:
            ipa = converter.text_to_ipa(word)
            f.write(f"{word}|{ipa}\n")

    print(f"Created lexicon with {len(words)} entries: {output_file}")


# === Demo and Testing ===

def demo():
    """Demonstrate IPA conversion"""
    converter = UzbekIPAConverter()

    test_sentences = [
        "Salom, dunyo!",
        "Men o'zbek tilida gaplashaman.",
        "Kitobni o'qiyapman.",
        "Bugun ob-havo yaxshi.",
        "Toshkent O'zbekistonning poytaxti.",
        "Assalomu alaykum!",
        "Xayrli tong!",
        "Rahmat, sog' bo'ling.",
        "Bog'da gullar ochildi.",
        "Quyosh chiqdi.",
    ]

    print("=" * 60)
    print("Uzbek to IPA Conversion Demo")
    print("=" * 60)

    for sentence in test_sentences:
        ipa = converter.text_to_ipa(sentence)
        print(f"\nText: {sentence}")
        print(f"IPA:  {ipa}")

    print("\n" + "=" * 60)
    print("Character mapping:")
    print("=" * 60)

    for char, ipa in sorted(UZBEK_IPA_MAP.items()):
        print(f"  {char:4} → {ipa}")


if __name__ == "__main__":
    demo()
