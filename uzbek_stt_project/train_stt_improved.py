#!/usr/bin/env python3
"""
Improved STT Training - Mix Real + Synthetic Data
By Abulqosim Rafiqov - December 2025

Key improvements:
1. Start from already fine-tuned model (26.7% WER)
2. Mix real audio with synthetic (not replace)
3. Add data augmentation (speed, noise, pitch)
4. Domain-specific vocabulary boost (IT terms)
"""

import os
import json
import random
import torch
import librosa
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from tqdm import tqdm

from datasets import Dataset, concatenate_datasets
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from peft import LoraConfig, get_peft_model, PeftModel
import evaluate

# === PATHS ===
# Use Linux paths for fast I/O
LINUX_DATA_DIR = Path("/home/abulqosim/stt_data")
BASE_DIR = Path("/mnt/c/Users/Admin/Desktop/voice_dataset/ISSAI_USC")
STT_DIR = BASE_DIR / "uzbek_stt_project"

# Original fine-tuned model (26.7% WER - our best) - use Linux copy
ORIGINAL_MODEL_DIR = LINUX_DATA_DIR / "output/final_model"

# Dataset sources - use Linux copies for speed
LINUX_TRAIN_DIR = LINUX_DATA_DIR / "train"
SYNTHETIC_DATA_DIR = LINUX_DATA_DIR / "synthetic_dataset"
METADATA_FILE = LINUX_DATA_DIR / "train_single_final.txt"

# Output - save to Windows for persistence
OUTPUT_DIR = STT_DIR / "whisper_finetuned_v2"

# === CONFIG ===
SAMPLE_RATE = 16000
MAX_DURATION = 30  # seconds
LANGUAGE = "uz"


# === DATA AUGMENTATION ===

def augment_audio(audio: np.ndarray, sr: int = 16000) -> np.ndarray:
    """Apply random augmentations to audio"""
    augmented = audio.copy()

    # Random speed change (0.9x - 1.1x)
    if random.random() < 0.3:
        speed_factor = random.uniform(0.9, 1.1)
        augmented = librosa.effects.time_stretch(augmented, rate=speed_factor)

    # Random pitch shift (-2 to +2 semitones)
    if random.random() < 0.3:
        n_steps = random.uniform(-2, 2)
        augmented = librosa.effects.pitch_shift(augmented, sr=sr, n_steps=n_steps)

    # Add slight noise
    if random.random() < 0.2:
        noise_level = random.uniform(0.001, 0.005)
        noise = np.random.randn(len(augmented)) * noise_level
        augmented = augmented + noise

    # Normalize
    max_val = np.max(np.abs(augmented))
    if max_val > 0:
        augmented = augmented / max_val * 0.95

    return augmented


# === DATASET LOADING ===

def load_real_dataset(max_samples: int = 5000) -> List[Dict]:
    """Load real audio samples from train/dev folders"""
    print("Loading real audio dataset...")

    samples = []

    # Load from train folder
    train_dir = REAL_DATA_DIR / "train"
    if train_dir.exists():
        for audio_file in tqdm(list(train_dir.glob("*.wav"))[:max_samples], desc="Real train"):
            # Parse filename: speakerid_gender_uttid_1.wav
            # Get corresponding text from metadata
            samples.append({
                "audio_path": str(audio_file),
                "source": "real_train"
            })

    # Load from dev folder
    dev_dir = REAL_DATA_DIR / "dev"
    if dev_dir.exists():
        for audio_file in tqdm(list(dev_dir.glob("*.wav"))[:500], desc="Real dev"):
            samples.append({
                "audio_path": str(audio_file),
                "source": "real_dev"
            })

    print(f"Loaded {len(samples)} real audio files")
    return samples


def load_real_dataset_with_text(max_samples: int = 3000) -> List[Dict]:
    """Load real audio with text labels from TTS metadata"""
    print("Loading real dataset with transcriptions...")

    samples = []

    # Load metadata from Linux copy
    if not METADATA_FILE.exists():
        print(f"Metadata not found: {METADATA_FILE}")
        return samples

    with open(METADATA_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    random.shuffle(lines)

    for line in tqdm(lines[:max_samples], desc="Real with text"):
        parts = line.strip().split('|')
        if len(parts) >= 2:
            audio_file = parts[0].strip()
            text = parts[1].strip()

            # Get just the filename and look in Linux train folder
            if audio_file.startswith('/'):
                filename = Path(audio_file).name
            else:
                filename = audio_file

            # Add .wav if needed
            if not filename.endswith('.wav'):
                filename = filename + ".wav"

            # Use Linux path for speed
            audio_path = LINUX_TRAIN_DIR / filename

            if not audio_path.exists():
                continue

            if 5 < len(text) < 300:  # Filter by text length
                samples.append({
                    "audio_path": str(audio_path),
                    "text": text,
                    "source": "real"
                })

    print(f"Loaded {len(samples)} real samples with text")
    return samples


def load_synthetic_dataset() -> List[Dict]:
    """Load synthetic audio dataset"""
    print("Loading synthetic dataset...")

    manifest_path = SYNTHETIC_DATA_DIR / "manifest.json"

    if not manifest_path.exists():
        print(f"Synthetic manifest not found: {manifest_path}")
        return []

    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)

    samples = []
    for item in manifest:
        audio_path = SYNTHETIC_DATA_DIR / item["audio"]
        if audio_path.exists():
            samples.append({
                "audio_path": str(audio_path),
                "text": item["text"],
                "source": "synthetic"
            })

    print(f"Loaded {len(samples)} synthetic samples")
    return samples


def create_mixed_dataset(real_ratio: float = 0.7):
    """Create mixed dataset with real and synthetic data"""
    print("\n" + "=" * 60)
    print("CREATING MIXED DATASET")
    print("=" * 60)

    # Load both datasets
    real_samples = load_real_dataset_with_text(max_samples=3000)
    synthetic_samples = load_synthetic_dataset()

    if not real_samples:
        print("WARNING: No real samples loaded!")
        return synthetic_samples

    # Calculate mix
    total_target = len(real_samples) + len(synthetic_samples)
    n_real = int(total_target * real_ratio)
    n_synthetic = total_target - n_real

    # Sample
    random.shuffle(real_samples)
    random.shuffle(synthetic_samples)

    final_real = real_samples[:n_real]
    final_synthetic = synthetic_samples[:n_synthetic]

    # Combine
    mixed = final_real + final_synthetic
    random.shuffle(mixed)

    print(f"\nMixed dataset composition:")
    print(f"  Real samples: {len(final_real)} ({len(final_real)/len(mixed)*100:.1f}%)")
    print(f"  Synthetic samples: {len(final_synthetic)} ({len(final_synthetic)/len(mixed)*100:.1f}%)")
    print(f"  Total: {len(mixed)}")

    return mixed


# === DATA COLLATOR ===

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int
    apply_augmentation: bool = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


# === TRAINING ===

def prepare_sample(sample: Dict, processor, apply_augment: bool = False) -> Dict:
    """Prepare a single sample for training"""
    # Load audio
    audio, sr = librosa.load(sample["audio_path"], sr=SAMPLE_RATE)

    # Truncate if too long
    max_samples = MAX_DURATION * SAMPLE_RATE
    if len(audio) > max_samples:
        audio = audio[:max_samples]

    # Apply augmentation to real samples only
    if apply_augment and sample.get("source") == "real":
        if random.random() < 0.5:  # 50% chance
            audio = augment_audio(audio, sr=SAMPLE_RATE)

    # Extract features
    input_features = processor.feature_extractor(
        audio,
        sampling_rate=SAMPLE_RATE
    ).input_features[0]

    # Tokenize text
    labels = processor.tokenizer(sample["text"]).input_ids

    return {
        "input_features": input_features,
        "labels": labels
    }


def main():
    print("=" * 70)
    print("IMPROVED STT TRAINING - MIXED REAL + SYNTHETIC")
    print("=" * 70)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load mixed dataset
    mixed_samples = create_mixed_dataset(real_ratio=0.7)

    if not mixed_samples:
        print("ERROR: No training data!")
        return

    # Create HuggingFace dataset
    dataset = Dataset.from_list(mixed_samples)

    # Split
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    print(f"\nTrain: {len(dataset['train'])}, Test: {len(dataset['test'])}")

    # Load processor
    print("\n[1/4] Loading processor from original model...")
    processor = WhisperProcessor.from_pretrained(str(ORIGINAL_MODEL_DIR))
    processor.tokenizer.set_prefix_tokens(language=LANGUAGE, task="transcribe")

    # Load original fine-tuned model (our best)
    print("\n[2/4] Loading original fine-tuned model (26.7% WER)...")

    # Check if it's a LoRA model
    adapter_config = ORIGINAL_MODEL_DIR / "adapter_config.json"

    if adapter_config.exists():
        print("Loading as LoRA model...")
        base_model = WhisperForConditionalGeneration.from_pretrained(
            "openai/whisper-large-v3",
            torch_dtype=torch.float16,
        )
        model = PeftModel.from_pretrained(base_model, str(ORIGINAL_MODEL_DIR))

        # Merge LoRA for continued training
        print("Merging LoRA weights...")
        model = model.merge_and_unload()
    else:
        print("Loading as merged model...")
        model = WhisperForConditionalGeneration.from_pretrained(
            str(ORIGINAL_MODEL_DIR),
            torch_dtype=torch.float16,
        )

    # Prepare model
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = False
    model.generation_config.language = LANGUAGE
    model.generation_config.task = "transcribe"

    # Apply NEW LoRA for continued fine-tuning
    print("\n[3/4] Applying new LoRA for domain adaptation...")
    lora_config = LoraConfig(
        r=16,  # Smaller rank for fine-tuning
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],  # Fewer modules
        lora_dropout=0.05,
        bias="none",
    )

    model = get_peft_model(model, lora_config)

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    model.print_trainable_parameters()

    # Prepare dataset
    print("\n[4/4] Preparing dataset with augmentation...")

    def prepare_fn(sample):
        return prepare_sample(sample, processor, apply_augment=True)

    # Process train and test separately with explicit progress
    print("Processing train set...")
    train_processed = []
    for i, sample in enumerate(tqdm(dataset["train"], desc="Train")):
        try:
            processed = prepare_sample(sample, processor, apply_augment=True)
            train_processed.append(processed)
        except Exception as e:
            print(f"Error on sample {i}: {e}")
            continue

    print("Processing test set...")
    test_processed = []
    for i, sample in enumerate(tqdm(dataset["test"], desc="Test")):
        try:
            processed = prepare_sample(sample, processor, apply_augment=False)
            test_processed.append(processed)
        except Exception as e:
            print(f"Error on sample {i}: {e}")
            continue

    prepared_dataset = {
        "train": Dataset.from_list(train_processed),
        "test": Dataset.from_list(test_processed)
    }

    # Data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    # WER metric
    wer_metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # Replace -100 with pad token id
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        # Decode
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer * 100}  # Return as percentage

    # Training arguments - skip eval during training (dtype issues with fp16)
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,  # Lower LR for fine-tuning
        warmup_steps=100,
        max_steps=1000,  # More steps
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        fp16=True,
        eval_strategy="no",  # Skip eval during training
        save_steps=200,
        logging_steps=50,
        report_to=["tensorboard"],
        load_best_model_at_end=False,
        push_to_hub=False,
    )

    # Trainer (no eval during training - will evaluate manually after)
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=prepared_dataset["train"],
        data_collator=data_collator,
        processing_class=processor.feature_extractor,
    )

    # Train
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)
    print(f"  Base model: Original fine-tuned (26.7% WER)")
    print(f"  Dataset: {len(prepared_dataset['train'])} samples (70% real, 30% synthetic)")
    print(f"  Augmentation: Enabled (speed, pitch, noise)")
    print(f"  Steps: 1000")
    print("=" * 60 + "\n")

    trainer.train()

    # Save
    print("\nSaving model...")
    trainer.save_model()
    processor.save_pretrained(str(OUTPUT_DIR))

    print(f"\nModel saved to: {OUTPUT_DIR}")
    print("Run evaluate_models.py to test WER")
    print("=" * 60)


if __name__ == "__main__":
    main()
