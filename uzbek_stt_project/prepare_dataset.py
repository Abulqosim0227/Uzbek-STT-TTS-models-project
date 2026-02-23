#!/usr/bin/env python3
"""
Uzbek STT Dataset Preparation Script
By Abulqosim Rafiqov - December 2025

This script prepares the ISSAI Uzbek Speech Corpus for Whisper fine-tuning:
1. Loads audio files and transcripts
2. Applies strict text normalization (apostrophe standardization)
3. Resamples audio to 16kHz (Whisper requirement)
4. Creates HuggingFace Dataset format
5. Saves processed dataset for training
"""

import os
import re
import json
import wave
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datasets import Dataset, DatasetDict, Audio
import librosa
import soundfile as sf

# === CONFIGURATION ===
BASE_DIR = Path("/mnt/c/Users/Admin/Desktop/voice_dataset/ISSAI_USC")
OUTPUT_DIR = BASE_DIR / "uzbek_stt_project" / "data"
SAMPLE_RATE = 16000  # Whisper requirement

# === TEXT NORMALIZATION ===
def normalize_uzbek_text(text: str) -> str:
    """
    Strict Uzbek text normalization for STT training.

    Critical: Standardize ALL apostrophe variants to single type '
    This prevents the model from learning multiple representations
    of the same phoneme (o', g', etc.)
    """
    # 1. Standardize apostrophes - CRITICAL for Uzbek
    # All these variants represent the same sound
    apostrophe_variants = ["'", "'", "`", "ʻ", "ʼ", "'", "´", "ˈ"]
    for variant in apostrophe_variants:
        text = text.replace(variant, "'")

    # 2. Convert any Cyrillic to Latin (if present)
    cyrillic_to_latin = {
        'а': 'a', 'б': 'b', 'в': 'v', 'г': 'g', 'д': 'd',
        'е': 'e', 'ё': 'yo', 'ж': 'j', 'з': 'z', 'и': 'i',
        'й': 'y', 'к': 'k', 'л': 'l', 'м': 'm', 'н': 'n',
        'о': 'o', 'п': 'p', 'р': 'r', 'с': 's', 'т': 't',
        'у': 'u', 'ф': 'f', 'х': 'x', 'ц': 'ts', 'ч': 'ch',
        'ш': 'sh', 'щ': 'sh', 'ъ': "'", 'ы': 'i', 'ь': '',
        'э': 'e', 'ю': 'yu', 'я': 'ya',
        # Uzbek-specific Cyrillic
        'ў': "o'", 'қ': 'q', 'ғ': "g'", 'ҳ': 'h',
    }
    for cyr, lat in cyrillic_to_latin.items():
        text = text.replace(cyr, lat)
        text = text.replace(cyr.upper(), lat.capitalize() if lat else '')

    # 3. Lowercase
    text = text.lower()

    # 4. Remove unwanted characters (keep letters, apostrophe, space)
    # Uzbek Latin alphabet: a-z + ' (for o', g', sh, ch are just digraphs)
    text = re.sub(r"[^a-z'\s]", " ", text)

    # 5. Clean up multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def load_split(split_dir: Path, max_samples: int = None) -> dict:
    """
    Load audio files and transcripts from a split directory.

    Args:
        split_dir: Path to train/dev/test directory
        max_samples: Optional limit for testing

    Returns:
        Dictionary with audio_paths and texts lists
    """
    audio_paths = []
    texts = []
    skipped = 0

    # Get all WAV files
    wav_files = sorted(split_dir.glob("*.wav"))
    if max_samples:
        wav_files = wav_files[:max_samples]

    print(f"Processing {len(wav_files)} files from {split_dir.name}...")

    for wav_path in tqdm(wav_files, desc=f"Loading {split_dir.name}"):
        # Find corresponding transcript
        txt_path = wav_path.with_suffix('.txt')

        if not txt_path.exists():
            skipped += 1
            continue

        # Read and normalize transcript
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                raw_text = f.read().strip()

            normalized_text = normalize_uzbek_text(raw_text)

            # Skip empty transcripts
            if not normalized_text:
                skipped += 1
                continue

            # Check audio duration (skip very short or very long)
            try:
                with wave.open(str(wav_path), 'rb') as wf:
                    duration = wf.getnframes() / wf.getframerate()

                # Whisper works best with 0.5s - 30s audio
                if duration < 0.5 or duration > 30:
                    skipped += 1
                    continue

            except Exception as e:
                skipped += 1
                continue

            audio_paths.append(str(wav_path))
            texts.append(normalized_text)

        except Exception as e:
            skipped += 1
            continue

    print(f"  Loaded: {len(audio_paths)}, Skipped: {skipped}")

    return {
        "audio": audio_paths,
        "text": texts
    }


def create_dataset(data_dict: dict) -> Dataset:
    """
    Create HuggingFace Dataset with audio resampling.
    """
    dataset = Dataset.from_dict(data_dict)

    # Cast audio column to Audio feature with resampling
    dataset = dataset.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))

    return dataset


def analyze_dataset(dataset: Dataset, name: str):
    """
    Print dataset statistics.
    """
    print(f"\n=== {name} Dataset Statistics ===")
    print(f"Total samples: {len(dataset)}")

    # Text statistics
    texts = dataset["text"]
    avg_words = sum(len(t.split()) for t in texts) / len(texts)
    avg_chars = sum(len(t) for t in texts) / len(texts)

    print(f"Avg words per sample: {avg_words:.1f}")
    print(f"Avg chars per sample: {avg_chars:.1f}")

    # Character set
    all_chars = set()
    for t in texts:
        all_chars.update(t)
    print(f"Character set ({len(all_chars)} chars): {''.join(sorted(all_chars))}")


def main():
    """
    Main function to prepare the dataset.
    """
    print("=" * 60)
    print("Uzbek STT Dataset Preparation")
    print("=" * 60)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load each split
    splits = {}

    for split_name in ["train", "dev", "test"]:
        split_dir = BASE_DIR / split_name
        if split_dir.exists():
            data = load_split(split_dir)
            splits[split_name] = create_dataset(data)
            analyze_dataset(splits[split_name], split_name.upper())

    # Create DatasetDict
    dataset_dict = DatasetDict(splits)

    # Save to disk
    output_path = OUTPUT_DIR / "uzbek_stt_dataset"
    print(f"\nSaving dataset to {output_path}...")
    dataset_dict.save_to_disk(str(output_path))

    # Also save a small sample for quick testing
    sample_path = OUTPUT_DIR / "uzbek_stt_sample"
    sample_dict = DatasetDict({
        "train": splits["train"].select(range(min(100, len(splits["train"])))),
        "dev": splits["dev"].select(range(min(50, len(splits["dev"])))),
        "test": splits["test"].select(range(min(50, len(splits["test"])))),
    })
    sample_dict.save_to_disk(str(sample_path))
    print(f"Saved sample dataset to {sample_path}")

    print("\n" + "=" * 60)
    print("Dataset preparation complete!")
    print("=" * 60)

    # Print summary
    print(f"\nTotal samples prepared:")
    print(f"  Train: {len(splits['train']):,}")
    print(f"  Dev:   {len(splits['dev']):,}")
    print(f"  Test:  {len(splits['test']):,}")
    print(f"\nDataset saved to: {output_path}")


if __name__ == "__main__":
    main()
