#!/usr/bin/env python3
"""
Evaluate STT Models - Compare Original vs Augmented Training
By Abulqosim Rafiqov - December 2025

Tests both models on:
1. Clean real audio (test set)
2. TTS-generated audio
3. Noisy/augmented audio
"""

import os
import json
import random
import numpy as np
import torch
import librosa
from pathlib import Path
from tqdm import tqdm
import evaluate

from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel

# === CONFIGURATION ===
BASE_DIR = Path("/mnt/c/Users/Admin/Desktop/voice_dataset/ISSAI_USC")
STT_PROJECT = BASE_DIR / "uzbek_stt_project"

# Models to compare
ORIGINAL_MODEL = STT_PROJECT / "output" / "final_model"
NEW_MODEL = STT_PROJECT / "whisper_finetuned_v2"
BASE_MODEL = "openai/whisper-large-v3"

SAMPLE_RATE = 16000
MAX_SAMPLES = 50  # Per category for quick evaluation


def load_model(model_path, base_model_name):
    """Load fine-tuned model"""
    print(f"Loading model from {model_path}...")

    processor = WhisperProcessor.from_pretrained(model_path)

    # Load base model
    model = WhisperForConditionalGeneration.from_pretrained(
        base_model_name,
        torch_dtype=torch.float32,
    )

    # Load LoRA weights
    model = PeftModel.from_pretrained(model, model_path)
    model = model.merge_and_unload()

    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language="uz", task="transcribe"
    )

    return model, processor


def transcribe(model, processor, audio_path):
    """Transcribe single audio file"""
    try:
        audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE)

        inputs = processor(
            audio,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt"
        )

        with torch.no_grad():
            predicted_ids = model.generate(
                inputs.input_features,
                max_length=225,
            )

        transcription = processor.batch_decode(
            predicted_ids,
            skip_special_tokens=True
        )[0]

        return transcription
    except Exception as e:
        return ""


def evaluate_on_dataset(model, processor, test_data, desc="Evaluating"):
    """Evaluate model on test dataset"""
    metric = evaluate.load("wer")

    predictions = []
    references = []

    for item in tqdm(test_data, desc=desc):
        audio_path = item["audio_path"]
        reference = item["text"]

        prediction = transcribe(model, processor, audio_path)

        predictions.append(prediction)
        references.append(reference)

    wer = 100 * metric.compute(predictions=predictions, references=references)
    return wer, predictions, references


def prepare_test_sets():
    """Prepare different test sets"""
    test_sets = {}

    # 1. Clean real audio (from test folder)
    test_dir = BASE_DIR / "test"
    clean_samples = []

    for wav_file in list(test_dir.glob("*.wav"))[:MAX_SAMPLES]:
        txt_file = wav_file.with_suffix(".txt")
        if txt_file.exists():
            with open(txt_file, 'r', encoding='utf-8') as f:
                text = f.read().strip()
            clean_samples.append({
                "audio_path": str(wav_file),
                "text": text,
                "type": "clean_real"
            })

    test_sets["clean_real"] = clean_samples
    print(f"Clean real samples: {len(clean_samples)}")

    # 2. TTS-generated audio
    augmented_dir = STT_PROJECT / "data" / "augmented_dataset"
    tts_samples = []

    for wav_file in list(augmented_dir.glob("tts_clean_*.wav"))[:MAX_SAMPLES]:
        # Find corresponding text from metadata
        stem = wav_file.stem
        idx = int(stem.split("_")[-1])

        # Read the augmented json to get text
        val_json = augmented_dir / "val_augmented.json"
        if val_json.exists():
            with open(val_json, 'r', encoding='utf-8') as f:
                val_data = json.load(f)

            # Find TTS samples
            for item in val_data:
                if item["type"] == "tts_clean" and str(wav_file) == item["audio_path"]:
                    tts_samples.append(item)
                    break

    # Also try training json
    train_json = augmented_dir / "train_augmented.json"
    if train_json.exists() and len(tts_samples) < MAX_SAMPLES:
        with open(train_json, 'r', encoding='utf-8') as f:
            train_data = json.load(f)

        for item in train_data:
            if item["type"] == "tts_clean" and len(tts_samples) < MAX_SAMPLES:
                if Path(item["audio_path"]).exists():
                    tts_samples.append(item)

    test_sets["tts_audio"] = tts_samples[:MAX_SAMPLES]
    print(f"TTS audio samples: {len(test_sets['tts_audio'])}")

    # 3. Noisy/augmented audio
    noisy_samples = []

    if train_json.exists():
        with open(train_json, 'r', encoding='utf-8') as f:
            train_data = json.load(f)

        for item in train_data:
            if "noise" in item["type"] or "random" in item["type"]:
                if Path(item["audio_path"]).exists():
                    noisy_samples.append(item)
                    if len(noisy_samples) >= MAX_SAMPLES:
                        break

    test_sets["noisy_audio"] = noisy_samples
    print(f"Noisy audio samples: {len(noisy_samples)}")

    return test_sets


def main():
    print("=" * 60)
    print("STT Model Evaluation - Original vs Augmented Training")
    print("=" * 60)

    # Prepare test sets
    print("\nPreparing test sets...")
    test_sets = prepare_test_sets()

    results = {}

    # Evaluate original model
    if ORIGINAL_MODEL.exists():
        print("\n" + "=" * 60)
        print("Evaluating ORIGINAL model")
        print("=" * 60)

        model_orig, processor_orig = load_model(ORIGINAL_MODEL, BASE_MODEL)

        for test_name, test_data in test_sets.items():
            if len(test_data) > 0:
                wer, _, _ = evaluate_on_dataset(
                    model_orig, processor_orig, test_data,
                    desc=f"Original - {test_name}"
                )
                results[f"original_{test_name}"] = wer
                print(f"  {test_name}: WER = {wer:.2f}%")

        del model_orig
    else:
        print(f"Original model not found at {ORIGINAL_MODEL}")

    # Evaluate new model
    if NEW_MODEL.exists():
        print("\n" + "=" * 60)
        print("Evaluating NEW (augmented) model")
        print("=" * 60)

        model_new, processor_new = load_model(NEW_MODEL, BASE_MODEL)

        for test_name, test_data in test_sets.items():
            if len(test_data) > 0:
                wer, _, _ = evaluate_on_dataset(
                    model_new, processor_new, test_data,
                    desc=f"New - {test_name}"
                )
                results[f"new_{test_name}"] = wer
                print(f"  {test_name}: WER = {wer:.2f}%")

        del model_new
    else:
        print(f"New model not found at {NEW_MODEL}")

    # Print comparison
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)

    print(f"\n{'Test Set':<20} {'Original':<15} {'New':<15} {'Change':<15}")
    print("-" * 65)

    for test_name in test_sets.keys():
        orig_key = f"original_{test_name}"
        new_key = f"new_{test_name}"

        orig_wer = results.get(orig_key, "N/A")
        new_wer = results.get(new_key, "N/A")

        if isinstance(orig_wer, float) and isinstance(new_wer, float):
            change = new_wer - orig_wer
            change_str = f"{change:+.2f}%"
            if change < 0:
                change_str += " (better)"
            elif change > 0:
                change_str += " (worse)"
        else:
            change_str = "N/A"

        orig_str = f"{orig_wer:.2f}%" if isinstance(orig_wer, float) else orig_wer
        new_str = f"{new_wer:.2f}%" if isinstance(new_wer, float) else new_wer

        print(f"{test_name:<20} {orig_str:<15} {new_str:<15} {change_str:<15}")

    # Save results
    results_path = STT_PROJECT / "evaluation_results.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
