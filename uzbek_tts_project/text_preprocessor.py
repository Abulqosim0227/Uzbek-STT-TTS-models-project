#!/usr/bin/env python3
"""
Enhanced Text Preprocessor for Uzbek TTS
By Abulqosim Rafiqov - December 2025

Features:
- Auto English word detection and transliteration
- Individual letter pronunciation (name or sound)
- Proper punctuation handling
- Number to Uzbek conversion
- Foreign word phonetic mapping
"""

import re
from typing import Dict, List, Tuple, Optional


# =============================================================================
# ENGLISH TO UZBEK PHONETIC TRANSLITERATION
# =============================================================================

# English phoneme patterns -> Uzbek approximation
ENGLISH_PHONETIC_RULES = [
    # Digraphs and common patterns (order matters - longest first)
    (r'tion\b', 'shn'),          # nation -> neyshn
    (r'sion\b', 'zhn'),          # vision -> vizhn
    (r'ous\b', 'as'),            # famous -> feymas
    (r'ture\b', 'cher'),         # culture -> kalcher
    (r'ight\b', 'ayt'),          # right -> rayt
    (r'ough\b', 'o'),            # though -> tho
    (r'ough', 'af'),             # enough -> inaf
    (r'tion', 'shn'),
    (r'th', 't'),                # think -> tink (no 'th' in Uzbek)
    (r'ph', 'f'),                # phone -> fon
    (r'gh', 'g'),                # ghost -> gost
    (r'wh', 'v'),                # what -> vat
    (r'wr', 'r'),                # write -> rayt
    (r'kn', 'n'),                # know -> no
    (r'ck', 'k'),                # back -> bak
    (r'qu', 'kv'),               # queen -> kvin
    (r'x', 'ks'),                # box -> boks

    # Vowel patterns
    (r'ee', 'i'),                # see -> si
    (r'ea', 'i'),                # read -> rid
    (r'oo', 'u'),                # food -> fud
    (r'ou', 'av'),               # house -> havs
    (r'ow\b', 'o'),              # show -> sho
    (r'ow', 'ov'),               # power -> pover
    (r'ey\b', 'ey'),             # key -> ki
    (r'ay\b', 'ey'),             # day -> dey
    (r'ai', 'ey'),               # rain -> reyn
    (r'au', 'o'),                # auto -> oto
    (r'aw', 'o'),                # law -> lo
    (r'oy', 'oy'),               # boy -> boy
    (r'oi', 'oy'),               # coin -> koyn

    # Silent e pattern
    (r'([bcdfghjklmnpqrstvwxyz])e\b', r'\1'),  # like -> layk

    # Consonant sounds
    (r'w', 'v'),                 # web -> veb
    (r'j', 'j'),                 # job -> job (Uzbek j = dʒ)
    (r'y(?=[aeiou])', 'y'),      # yes -> yes
    (r'c(?=[eiy])', 's'),        # city -> siti
    (r'c', 'k'),                 # cat -> kat
    (r'g(?=[eiy])', 'j'),        # gem -> jem

    # Double letters -> single
    (r'([bcdfghjklmnpqrstvwxyz])\1', r'\1'),  # apple -> apl
]

# Known English words with correct Uzbek pronunciation
PHONETIC_MAP = {
    # Tech brands
    "python": "payton",
    "javascript": "javaskript",
    "facebook": "feysbuk",
    "instagram": "instagram",
    "youtube": "yutub",
    "google": "gugl",
    "iphone": "ayfon",
    "samsung": "samsung",
    "whatsapp": "votsap",
    "telegram": "telegram",
    "twitter": "tvitter",
    "tiktok": "tiktok",
    "android": "android",
    "windows": "vindovs",
    "linux": "linuks",
    "microsoft": "maykrosoft",
    "apple": "epl",
    "amazon": "amazon",
    "netflix": "netfliks",
    "spotify": "spotifay",
    "uber": "uber",
    "tesla": "tesla",
    "bitcoin": "bitkoin",
    "chatgpt": "chat-ji-pi-ti",
    "openai": "open-ey-ay",

    # Abbreviations
    "ai": "ey-ay",
    "ok": "okey",
    "wifi": "vay-fay",
    "usb": "yu-es-bi",
    "pdf": "pi-di-ef",
    "sms": "es-em-es",
    "gps": "ji-pi-es",
    "atm": "ey-ti-em",
    "ceo": "si-i-o",
    "it": "ay-ti",
    "api": "ey-pi-ay",
    "url": "yu-ar-el",
    "html": "eych-ti-em-el",
    "css": "si-es-es",
    "sql": "es-kyu-el",

    # Common English words
    "hello": "xello",
    "hi": "xay",
    "bye": "bay",
    "thanks": "tenks",
    "please": "pliz",
    "sorry": "sori",
    "computer": "kompyuter",
    "laptop": "laptop",
    "internet": "internet",
    "online": "onlayn",
    "offline": "oflayn",
    "email": "imeil",
    "website": "vebsayt",
    "software": "softver",
    "hardware": "xardver",
    "download": "download",
    "upload": "aploud",
    "startup": "startap",
    "business": "biznes",
    "marketing": "marketing",
    "manager": "menejer",
    "project": "proyekt",
    "design": "dizayn",
    "developer": "developer",
    "programmer": "programmist",
    "machine": "mashin",
    "learning": "lerning",
    "network": "netvork",
    "server": "server",
    "database": "databeys",
    "cloud": "klaud",
    "mobile": "mobayl",
    "app": "ep",
    "application": "aplikeyshn",
    "update": "apdeyt",
    "version": "vershn",
    "feature": "ficher",
    "bug": "bag",
    "fix": "fiks",
    "test": "test",
    "code": "kod",
    "file": "fayl",
    "folder": "folder",
    "user": "yuzer",
    "password": "parol",
    "login": "login",
    "logout": "logaut",
    "account": "akkaunt",
    "profile": "profayl",
    "settings": "settings",
    "search": "serch",
    "filter": "filtr",
    "sort": "sort",
    "save": "seyv",
    "delete": "dilit",
    "edit": "edit",
    "copy": "kopi",
    "paste": "peyst",
    "share": "sher",
    "like": "layk",
    "comment": "komment",
    "post": "post",
    "story": "stori",
    "video": "video",
    "audio": "audio",
    "image": "imij",
    "photo": "foto",
    "camera": "kamera",
    "screen": "skrin",
    "display": "displey",
    "keyboard": "klaviatura",
    "mouse": "maus",
    "click": "klik",
    "scroll": "skrol",
    "zoom": "zum",
    "print": "print",
    "scan": "skan",
    "bluetooth": "blutus",
    "speaker": "spiker",
    "microphone": "mikrofon",
    "headphone": "xedfon",
    "charger": "zaryadka",
    "battery": "batareya",
    "power": "paver",
    "button": "tugma",
    "icon": "ikon",
    "menu": "menyu",
    "window": "oyna",
    "tab": "tab",
    "browser": "brauzer",
    "chrome": "xrom",
    "firefox": "fayerfoks",
    "safari": "safari",
    "game": "geym",
    "play": "pley",
    "stop": "stop",
    "pause": "pauza",
    "next": "nekst",
    "previous": "previus",
    "volume": "volume",
    "mute": "myut",
    "record": "rekord",
    "stream": "strim",
    "live": "layv",
    "channel": "kanal",
    "subscribe": "sabskrayb",
    "notification": "notifikatsiya",
    "message": "xabar",
    "chat": "chat",
    "call": "qo'ng'iroq",
    "meeting": "miting",
    "conference": "konferensiya",
    "presentation": "prezentatsiya",
    "document": "dokument",
    "spreadsheet": "elektron jadval",
    "format": "format",
    "style": "stayl",
    "template": "shablon",
    "draft": "qoralama",
    "final": "final",
    "submit": "yuborish",
    "confirm": "tasdiqlash",
    "cancel": "bekor qilish",
    "error": "xato",
    "warning": "ogohlantirish",
    "success": "muvaffaqiyat",
    "loading": "yuklanmoqda",
    "processing": "qayta ishlanmoqda",
    "complete": "tugallangan",
    "pending": "kutilmoqda",
    "active": "faol",
    "inactive": "nofaol",
    "enable": "yoqish",
    "disable": "o'chirish",
}

# Uzbek alphabet letter names
UZBEK_LETTER_NAMES = {
    "a": "a",
    "b": "be",
    "c": "se",      # rarely used
    "d": "de",
    "e": "e",
    "f": "ef",
    "g": "ge",
    "g'": "g'e",    # voiced uvular
    "h": "ha",
    "i": "i",
    "j": "je",
    "k": "ka",
    "l": "el",
    "m": "em",
    "n": "en",
    "o": "o",
    "o'": "o'",     # rounded
    "p": "pe",
    "q": "qe",
    "r": "er",
    "s": "es",
    "t": "te",
    "u": "u",
    "v": "ve",
    "x": "xa",
    "y": "ye",
    "z": "ze",
    "sh": "sha",
    "ch": "che",
    "ng": "eng",
}


# =============================================================================
# ENGLISH WORD DETECTION
# =============================================================================

def is_english_word(word: str) -> bool:
    """
    Detect if a word is likely English (not Uzbek)

    Rules:
    - Contains 'w' (not in Uzbek alphabet)
    - Contains 'c' alone (not as 'ch')
    - Contains patterns like 'th', 'ph', 'ght'
    - Multiple consecutive consonants uncommon in Uzbek
    - Ends with typical English suffixes
    """
    word_lower = word.lower().strip()

    # Skip if too short
    if len(word_lower) < 2:
        return False

    # Skip numbers and punctuation
    if not word_lower.isalpha():
        return False

    # Characters NOT in Uzbek alphabet (clear indicators)
    non_uzbek_chars = ['w']
    for char in non_uzbek_chars:
        if char in word_lower:
            return True

    # 'c' without 'h' following (Uzbek uses 'ch' but not standalone 'c')
    if re.search(r'c(?!h)', word_lower):
        return True

    # English-specific patterns
    english_patterns = [
        r'th',          # the, this, think
        r'ph',          # phone, photo
        r'ght',         # right, night
        r'tion\b',      # nation, action
        r'sion\b',      # vision, mission
        r'ous\b',       # famous, serious
        r'ness\b',      # happiness
        r'ment\b',      # development
        r'ing\b',       # running, coding
        r'ed\b',        # worked, played
        r'ly\b',        # quickly, slowly
        r'^un',         # unusual, under
        r'^re',         # return, review (but careful - Uzbek has 're')
        r'ck',          # back, click
        r'wh',          # what, where
        r'wr',          # write, wrong
        r'kn',          # know, knife
        r'[aeiou]{3}',  # three+ vowels together
    ]

    for pattern in english_patterns:
        if re.search(pattern, word_lower):
            return True

    return False


def transliterate_english_to_uzbek(word: str) -> str:
    """
    Convert English word to Uzbek phonetic spelling
    """
    result = word.lower()

    # First check known words
    if result in PHONETIC_MAP:
        return PHONETIC_MAP[result]

    # Apply phonetic rules
    for pattern, replacement in ENGLISH_PHONETIC_RULES:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)

    return result


# =============================================================================
# LETTER PRONUNCIATION
# =============================================================================

def get_letter_pronunciation(letter: str, mode: str = "name") -> str:
    """
    Get pronunciation for individual letter

    Args:
        letter: Single letter or digraph (e.g., 'g'', 'sh', 'ch')
        mode: "name" for letter name, "sound" for just the sound

    Returns:
        Pronunciation string
    """
    letter_lower = letter.lower().strip()

    # Normalize apostrophes
    for apo in ["'", "'", "`", "ʻ", "ʼ"]:
        letter_lower = letter_lower.replace(apo, "'")

    if mode == "name":
        # Return letter name (like reading alphabet)
        return UZBEK_LETTER_NAMES.get(letter_lower, letter_lower)
    else:
        # Return just the sound (for IPA conversion later)
        return letter_lower


def expand_letter_sequences(text: str, mode: str = "name") -> str:
    """
    Detect and expand sequences of individual letters

    Example: "G', X, L, P" -> "g'e, xa, el, pe" (if mode="name")
    """
    # Pattern: letter followed by comma/space and another letter
    # This catches "A, B, C" style sequences

    # First, find letter sequences
    def replace_letter(match):
        letter = match.group(1)
        return get_letter_pronunciation(letter, mode)

    # Match individual Uzbek letters (including digraphs) when isolated
    # Pattern: word boundary, letter/digraph, word boundary
    digraph_pattern = r"\b(g'|o'|sh|ch|ng)\b"
    single_pattern = r"\b([a-zA-Z])\b"

    # Replace digraphs first
    text = re.sub(digraph_pattern, replace_letter, text, flags=re.IGNORECASE)
    # Then single letters (but not if already part of a word)
    # Only replace if surrounded by punctuation or spaces
    text = re.sub(r'(?<=[,\s])([a-zA-Z])(?=[,\s\.]|$)', replace_letter, text)

    return text


# =============================================================================
# NUMBER TO UZBEK
# =============================================================================

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


def number_to_uzbek(n: int, ordinal: bool = False) -> str:
    """Convert number to Uzbek text"""
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

    if ordinal and result:
        last_char = result[-1].lower()
        vowels = "aeiouoʻ'"
        if last_char in vowels:
            result += "nchi"
        else:
            result += "inchi"

    return result


def convert_numbers_in_text(text: str) -> str:
    """Find and convert all numbers in text"""

    # Years with ordinal
    def replace_year_ordinal(match):
        num = int(match.group(1))
        suffix = match.group(2)
        return number_to_uzbek(num, ordinal=True) + " " + suffix

    text = re.sub(r'\b(\d{4})\s*(yil|yilda|yili)\b', replace_year_ordinal, text)

    # Years without ordinal
    def replace_year_regular(match):
        num = int(match.group(1))
        suffix = match.group(2)
        return number_to_uzbek(num, ordinal=False) + " " + suffix

    text = re.sub(r'\b(\d+)\s*(yildan|yilga|yilgacha|yillik)\b', replace_year_regular, text)

    # Dates
    def replace_date(match):
        num = int(match.group(1))
        month = match.group(2)
        return number_to_uzbek(num, ordinal=True) + " " + month

    text = re.sub(
        r'\b(\d{1,2})-(yanvar|fevral|mart|aprel|may|iyun|iyul|avgust|sentabr|oktabr|noyabr|dekabr)\b',
        replace_date, text, flags=re.IGNORECASE
    )

    # Ordinals with -chi suffix
    def replace_ordinal_chi(match):
        num = int(match.group(1))
        return number_to_uzbek(num, ordinal=True)

    text = re.sub(r'\b(\d+)-?(chi|nchi|inchi)\b', replace_ordinal_chi, text, flags=re.IGNORECASE)

    # Percentages
    text = re.sub(r'(\d+)\s*%', lambda m: number_to_uzbek(int(m.group(1))) + " foiz", text)

    # Regular numbers
    def replace_number(match):
        num = int(match.group())
        return number_to_uzbek(num, ordinal=False)

    text = re.sub(r'\b\d+\b', replace_number, text)

    return text


# =============================================================================
# UZBEK IPA CONVERTER (FIXED)
# =============================================================================

UZBEK_IPA_MAP = {
    # Vowels
    'a': 'ɑ',
    'e': 'e',
    'i': 'i',
    'o': 'ɔ',
    'u': 'u',
    "o'": 'ø',

    # Consonants
    'b': 'b',
    'd': 'd',
    'f': 'f',
    'g': 'g',
    "g'": 'ʁ',
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
    'x': 'χ',
    'y': 'j',
    'z': 'z',

    # Digraphs
    'sh': 'ʃ',
    'ch': 'tʃ',
    'ng': 'ŋ',
}


def uzbek_to_ipa(text: str, preserve_punctuation: bool = True) -> str:
    """
    Convert Uzbek text to IPA with proper punctuation handling

    Args:
        text: Uzbek text to convert
        preserve_punctuation: If True, keep punctuation for pause markers
    """
    text = text.lower()

    # Normalize apostrophes
    for apo in ["'", "'", "`", "ʻ", "ʼ"]:
        text = text.replace(apo, "'")

    result = []
    i = 0

    # Sort by length (longest first)
    sorted_keys = sorted(UZBEK_IPA_MAP.keys(), key=len, reverse=True)

    while i < len(text):
        matched = False

        # Try to match longest sequences first
        for key in sorted_keys:
            if text[i:].startswith(key):
                result.append(UZBEK_IPA_MAP[key])
                i += len(key)
                matched = True
                break

        if not matched:
            char = text[i]
            if char.isalpha():
                # Unknown letter - keep as is
                result.append(char)
            elif preserve_punctuation and char in ".,!?;:-":
                # Keep punctuation (model should handle it)
                result.append(char)
            elif char == ' ':
                result.append(' ')
            # Skip other characters
            i += 1

    return ''.join(result)


# =============================================================================
# MAIN PREPROCESSOR
# =============================================================================

class UzbekTextPreprocessor:
    """
    Complete text preprocessing for Uzbek TTS
    """

    def __init__(
        self,
        detect_english: bool = True,
        transliterate_english: bool = True,
        convert_numbers: bool = True,
        letter_mode: str = "name",  # "name" or "sound"
        convert_to_ipa: bool = True,
        preserve_punctuation: bool = True
    ):
        self.detect_english = detect_english
        self.transliterate_english = transliterate_english
        self.convert_numbers = convert_numbers
        self.letter_mode = letter_mode
        self.convert_to_ipa = convert_to_ipa
        self.preserve_punctuation = preserve_punctuation

    def preprocess(self, text: str) -> str:
        """
        Full preprocessing pipeline
        """
        original = text

        # 1. Normalize apostrophes
        for apo in ["'", "'", "`", "ʻ", "ʼ"]:
            text = text.replace(apo, "'")

        # 2. Convert numbers
        if self.convert_numbers:
            text = convert_numbers_in_text(text)

        # 3. Expand abbreviations
        text = text.replace(" v.h.", " va hokazo")
        text = text.replace(" h.k.", " hamda boshqalar")
        text = text.replace(" va b.", " va boshqalar")

        # 4. Handle individual letters
        text = expand_letter_sequences(text, self.letter_mode)

        # 5. Detect and transliterate English words
        if self.detect_english and self.transliterate_english:
            words = text.split()
            processed_words = []
            for word in words:
                # Extract punctuation
                prefix = ""
                suffix = ""
                core = word

                while core and not core[0].isalnum():
                    prefix += core[0]
                    core = core[1:]
                while core and not core[-1].isalnum():
                    suffix = core[-1] + suffix
                    core = core[:-1]

                if core and is_english_word(core):
                    core = transliterate_english_to_uzbek(core)

                processed_words.append(prefix + core + suffix)

            text = ' '.join(processed_words)

        # 6. Clean up multiple spaces
        text = re.sub(r'\s+', ' ', text).strip()

        # 7. Convert to IPA (if enabled)
        if self.convert_to_ipa:
            text = uzbek_to_ipa(text, self.preserve_punctuation)

        return text

    def preprocess_for_v1(self, text: str) -> str:
        """Preprocess for V1 model (text-based, no IPA)"""
        self.convert_to_ipa = False
        return self.preprocess(text)

    def preprocess_for_v3(self, text: str) -> str:
        """Preprocess for V3 model (IPA-based)"""
        self.convert_to_ipa = True
        return self.preprocess(text)


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    preprocessor = UzbekTextPreprocessor()

    print("=" * 70)
    print("UZBEK TEXT PREPROCESSOR DEMO")
    print("=" * 70)

    test_cases = [
        # English words
        ("Men Python va JavaScript o'rganmoqdaman.", "English words in sentence"),
        ("Facebook, Instagram va WhatsApp ilovalarini ishlataman.", "Social media brands"),
        ("Machine learning juda qiziqarli soha.", "Unknown English word"),

        # Individual letters
        ("G', X, L, P harflarini aytib bering.", "Individual letters"),
        ("A, B, C, D harflari.", "Latin letters"),

        # Numbers
        ("2024 yilda 25 yoshda bo'laman.", "Year and age"),
        ("100% to'g'ri.", "Percentage"),
        ("6-dekabr kuni.", "Date"),

        # Mixed
        ("Men 2023 yilda YouTube kanalimda Python darslarini boshladim.", "Complex mixed"),

        # Punctuation
        ("Salom, dunyo! Qalaysiz? Yaxshi.", "Punctuation test"),
    ]

    for text, description in test_cases:
        print(f"\n--- {description} ---")
        print(f"Original:    {text}")

        # For V1 (no IPA)
        result_v1 = preprocessor.preprocess_for_v1(text)
        print(f"V1 (text):   {result_v1}")

        # For V3 (with IPA)
        result_v3 = preprocessor.preprocess_for_v3(text)
        print(f"V3 (IPA):    {result_v3}")

    print("\n" + "=" * 70)
    print("ENGLISH DETECTION TEST")
    print("=" * 70)

    english_tests = [
        "machine", "learning", "python", "javascript",
        "kitob", "o'qish", "yozish", "gapirish",
        "computer", "website", "database", "network",
        "salom", "rahmat", "xayr", "kechirasiz",
    ]

    for word in english_tests:
        is_eng = is_english_word(word)
        if is_eng:
            transliterated = transliterate_english_to_uzbek(word)
            print(f"{word:15} -> English -> {transliterated}")
        else:
            print(f"{word:15} -> Uzbek")
