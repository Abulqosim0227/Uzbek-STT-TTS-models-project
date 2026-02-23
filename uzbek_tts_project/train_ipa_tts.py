#!/usr/bin/env python3
"""
Train Uzbek TTS with IPA Phonemes
By Abulqosim Rafiqov - December 2025

This trains a VITS model using IPA transcriptions for better pronunciation.
"""

import os
import json
from pathlib import Path

# Set environment
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.models.vits import Vits
from trainer import Trainer, TrainerArgs
from TTS.tts.datasets import load_tts_samples
from TTS.utils.audio import AudioProcessor

# Paths
BASE_DIR = Path("/mnt/c/Users/Admin/Desktop/voice_dataset/ISSAI_USC")
TTS_PROJECT = BASE_DIR / "uzbek_tts_project"
IPA_DATASET = TTS_PROJECT / "ipa_dataset"
OUTPUT_DIR = TTS_PROJECT / "training_output_ipa"

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# IPA characters used in Uzbek
IPA_CHARACTERS = "ɑeiɔuøbdfgʁhjdʒklmnpqrstvχzʃtʃŋ |.!?,"

def main():
    print("=" * 60)
    print("Uzbek TTS Training with IPA Phonemes")
    print("=" * 60)

    # Load config
    config = VitsConfig(
        run_name="uzbek_tts_ipa",
        run_description="Uzbek TTS with IPA - Better Pronunciation",

        # Audio settings
        audio={
            "sample_rate": 22050,
            "fft_size": 1024,
            "win_length": 1024,
            "hop_length": 256,
            "num_mels": 80,
            "mel_fmin": 0,
            "mel_fmax": None,
        },

        # Training settings
        batch_size=32,
        eval_batch_size=16,
        num_loader_workers=4,
        num_eval_loader_workers=2,
        run_eval=True,
        test_delay_epochs=5,
        epochs=1000,

        # Text settings - use IPA directly
        text_cleaner="basic_cleaners",
        use_phonemes=False,
        characters={
            "characters_class": "TTS.tts.models.vits.VitsCharacters",
            "pad": "_",
            "eos": "~",
            "bos": "^",
            "characters": IPA_CHARACTERS,
            "punctuations": " |.!?,",
            "phonemes": None,
        },

        # Model settings
        print_step=50,
        print_eval=False,
        mixed_precision=True,
        save_step=5000,
        save_n_checkpoints=3,
        save_best_after=10000,

        # Output
        output_path=str(OUTPUT_DIR),

        # Dataset
        datasets=[
            {
                "dataset_name": "uzbek_ipa",
                "formatter": "ljspeech",
                "meta_file_train": str(IPA_DATASET / "metadata_train_ipa_fixed.txt"),
                "meta_file_val": str(IPA_DATASET / "metadata_val_ipa_fixed.txt"),
                "path": str(BASE_DIR),
                "language": "uz",
            }
        ],

        # Test sentences in IPA
        test_sentences=[
            "sɑlɔm dunjɔ",
            "men øzbek tilidɑ gɑplɑʃɑmɑn",
            "bugun ɔb hɑvɔ jɑχʃi",
            "rɑhmɑt sɔʁ bøliŋ",
        ],
    )

    # Initialize audio processor
    ap = AudioProcessor.init_from_config(config)

    # Load training samples
    print("\nLoading training data...")
    train_samples, eval_samples = load_tts_samples(
        config.datasets,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )

    print(f"Training samples: {len(train_samples)}")
    print(f"Evaluation samples: {len(eval_samples)}")

    # Initialize model
    print("\nInitializing VITS model...")
    model = Vits(config, ap, tokenizer=None, speaker_manager=None)

    # Initialize trainer
    trainer = Trainer(
        TrainerArgs(
            restore_path=None,
            skip_train_epoch=False,
        ),
        config,
        output_path=str(OUTPUT_DIR),
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )

    # Start training
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)

    trainer.fit()

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
