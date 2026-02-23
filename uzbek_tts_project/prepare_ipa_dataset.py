#!/usr/bin/env python3
"""
Prepare IPA Dataset for Uzbek TTS Training
By Abulqosim Rafiqov - December 2025

Converts existing TTS dataset transcripts to IPA format.
"""

import os
import json
from pathlib import Path
from tqdm import tqdm
from uzbek_ipa import UzbekIPAConverter

# Paths
BASE_DIR = Path("/mnt/c/Users/Admin/Desktop/voice_dataset/ISSAI_USC")
TTS_PROJECT = BASE_DIR / "uzbek_tts_project"
OUTPUT_DIR = TTS_PROJECT / "ipa_dataset"

# Original data directories
DATA_DIRS = [
    BASE_DIR / "train",
    BASE_DIR / "dev",
    BASE_DIR / "test",
]


def convert_dataset_to_ipa():
    """Convert all transcripts to IPA format"""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    converter = UzbekIPAConverter()

    all_entries = []
    stats = {"total": 0, "converted": 0, "errors": 0}

    print("=" * 60)
    print("Converting Uzbek TTS Dataset to IPA")
    print("=" * 60)

    for data_dir in DATA_DIRS:
        if not data_dir.exists():
            print(f"Skipping {data_dir} (not found)")
            continue

        txt_files = list(data_dir.glob("*.txt"))
        print(f"\nProcessing {data_dir.name}: {len(txt_files)} files")

        for txt_file in tqdm(txt_files, desc=data_dir.name):
            wav_file = txt_file.with_suffix(".wav")

            if not wav_file.exists():
                continue

            stats["total"] += 1

            try:
                # Read original text
                with open(txt_file, 'r', encoding='utf-8') as f:
                    text = f.read().strip()

                # Convert to IPA
                ipa = converter.text_to_ipa(text)

                # Store entry
                all_entries.append({
                    "audio_path": str(wav_file),
                    "text": text,
                    "ipa": ipa,
                    "split": data_dir.name,
                })

                stats["converted"] += 1

            except Exception as e:
                stats["errors"] += 1
                continue

    # Split into train/val/test
    train_entries = [e for e in all_entries if e["split"] == "train"]
    val_entries = [e for e in all_entries if e["split"] == "dev"]
    test_entries = [e for e in all_entries if e["split"] == "test"]

    # Save datasets
    datasets = {
        "train_ipa.json": train_entries,
        "val_ipa.json": val_entries,
        "test_ipa.json": test_entries,
        "all_ipa.json": all_entries,
    }

    for filename, data in datasets.items():
        output_path = OUTPUT_DIR / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Saved: {output_path} ({len(data)} entries)")

    # Create LJSpeech-style metadata files
    print("\nCreating LJSpeech-style metadata...")

    for split_name, entries in [("train", train_entries), ("val", val_entries)]:
        # Text version (original)
        meta_text = OUTPUT_DIR / f"metadata_{split_name}.txt"
        with open(meta_text, 'w', encoding='utf-8') as f:
            for entry in entries:
                basename = Path(entry["audio_path"]).stem
                f.write(f"{basename}|{entry['text']}|{entry['text']}\n")

        # IPA version
        meta_ipa = OUTPUT_DIR / f"metadata_{split_name}_ipa.txt"
        with open(meta_ipa, 'w', encoding='utf-8') as f:
            for entry in entries:
                basename = Path(entry["audio_path"]).stem
                f.write(f"{basename}|{entry['ipa']}|{entry['ipa']}\n")

    # Print statistics
    print("\n" + "=" * 60)
    print("Conversion Statistics")
    print("=" * 60)
    print(f"Total files: {stats['total']}")
    print(f"Converted: {stats['converted']}")
    print(f"Errors: {stats['errors']}")
    print(f"\nTrain: {len(train_entries)}")
    print(f"Val: {len(val_entries)}")
    print(f"Test: {len(test_entries)}")

    # Show some examples
    print("\n" + "=" * 60)
    print("Sample Conversions")
    print("=" * 60)

    for entry in all_entries[:5]:
        print(f"\nText: {entry['text'][:60]}...")
        print(f"IPA:  {entry['ipa'][:60]}...")

    print("\n" + "=" * 60)
    print(f"Output saved to: {OUTPUT_DIR}")
    print("=" * 60)

    return OUTPUT_DIR


if __name__ == "__main__":
    convert_dataset_to_ipa()
