#!/usr/bin/env python3
"""
Test the Fine-tuned Whisper Model for Uzbek STT
This script loads the LoRA-adapted model and tests it on audio files
"""

import os
import sys
import torch
import time
from pathlib import Path
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel
import librosa

# Paths
BASE_DIR = Path("/mnt/c/Users/Admin/Desktop/voice_dataset/ISSAI_USC/uzbek_stt_project")
MODEL_DIR = BASE_DIR / "whisper_finetuned_synthetic"

# Test audio files
TEST_AUDIOS = [
    "/mnt/c/Users/Admin/Desktop/voice_dataset/ISSAI_USC/uzbek_tts_project/audio/sound_ch1_v3.wav",
    "/mnt/c/Users/Admin/Desktop/voice_dataset/ISSAI_USC/uzbek_tts_project/audio/showcase_v3_1.wav",
    "/mnt/c/Users/Admin/Desktop/voice_dataset/ISSAI_USC/uzbek_tts_project/audio/showcase_v3_2.wav",
]


def load_model():
    """Load the fine-tuned Whisper model with LoRA"""
    print("=" * 60)
    print("LOADING FINE-TUNED WHISPER MODEL")
    print("=" * 60)

    start = time.time()

    # Load processor
    processor = WhisperProcessor.from_pretrained(str(MODEL_DIR))

    # Load base model
    print("Loading base Whisper Large V3...")
    base_model = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-large-v3",
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # Load LoRA adapter
    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, str(MODEL_DIR))

    # Merge LoRA weights for faster inference
    print("Merging LoRA weights...")
    model = model.merge_and_unload()

    load_time = time.time() - start
    print(f"Model loaded in {load_time:.1f}s")

    return model, processor


def transcribe(model, processor, audio_path):
    """Transcribe an audio file"""
    # Load audio
    audio, sr = librosa.load(audio_path, sr=16000)

    # Process
    input_features = processor.feature_extractor(
        audio,
        sampling_rate=16000,
        return_tensors="pt"
    ).input_features.to(model.device, dtype=torch.float16)

    # Generate
    start = time.time()

    with torch.no_grad():
        predicted_ids = model.generate(
            input_features,
            language="uz",
            task="transcribe",
            max_length=225,
        )

    inference_time = time.time() - start

    # Decode
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

    # Get audio duration
    duration = len(audio) / sr

    return {
        "text": transcription.strip(),
        "duration": duration,
        "inference_time": inference_time,
        "realtime_factor": inference_time / duration
    }


def main():
    # Check if model exists
    if not MODEL_DIR.exists():
        print(f"ERROR: Model not found at {MODEL_DIR}")
        print("Please run the training first!")
        sys.exit(1)

    # Load model
    model, processor = load_model()

    print("\n" + "=" * 60)
    print("TRANSCRIPTION TEST")
    print("=" * 60)

    # Test each audio file
    for audio_path in TEST_AUDIOS:
        if not Path(audio_path).exists():
            print(f"\n[SKIP] Audio not found: {audio_path}")
            continue

        print(f"\n[TEST] {Path(audio_path).name}")

        result = transcribe(model, processor, audio_path)

        print(f"  Text: {result['text']}")
        print(f"  Duration: {result['duration']:.2f}s")
        print(f"  Inference: {result['inference_time']:.3f}s")
        print(f"  Realtime Factor: {result['realtime_factor']:.2f}x")

    # Interactive mode
    print("\n" + "=" * 60)
    print("INTERACTIVE MODE")
    print("=" * 60)
    print("Enter audio file path (or 'quit' to exit):\n")

    while True:
        try:
            user_input = input("> ").strip()

            if user_input.lower() in ['quit', 'exit', 'q']:
                break

            if not user_input:
                continue

            if not Path(user_input).exists():
                print(f"File not found: {user_input}")
                continue

            result = transcribe(model, processor, user_input)
            print(f"\n  Text: {result['text']}")
            print(f"  Time: {result['inference_time']:.3f}s ({result['realtime_factor']:.2f}x realtime)\n")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

    print("\nGoodbye!")


if __name__ == "__main__":
    main()
