#!/usr/bin/env python3
"""
Generate Synthetic STT Training Dataset
Uses TTS V3 to create (audio, text) pairs for Whisper fine-tuning

This creates the "Golden Dataset" for training Jarvis-level STT
"""

import os
import sys
import json
import random
import torch
from pathlib import Path
from TTS.utils.synthesizer import Synthesizer
import soundfile as sf
import noisereduce as nr
import numpy as np
from tqdm import tqdm

# Paths
BASE_DIR = Path("/mnt/c/Users/Admin/Desktop/voice_dataset/ISSAI_USC")
OUTPUT_DIR = BASE_DIR / "uzbek_stt_project" / "synthetic_dataset"
AUDIO_DIR = OUTPUT_DIR / "audio"

# TTS V3 Model (best pronunciation)
V3_MODEL = BASE_DIR / "uzbek_tts_project/training_output_ipa_v3/uzbek_tts_ipa_v3-December-12-2025_09+25AM-0000000/checkpoint_714395.pth"
V3_CONFIG = BASE_DIR / "uzbek_tts_project/training_output_ipa_v3/uzbek_tts_ipa_v3-December-12-2025_09+25AM-0000000/config.json"

# IT/Nihol Domain Sentences (Priority Training)
IT_SENTENCES = [
    # Company & Services
    "Nihol kompaniyasi axborot texnologiyalari sohasida xizmat ko'rsatadi.",
    "Biz Cloud Computing yechimlarini taqdim etamiz.",
    "Data Center xizmatlari bizning asosiy yo'nalishimiz.",
    "Server infratuzilmasini boshqarish xizmatlarini taklif qilamiz.",
    "Kiberxavfsizlik bo'yicha mutaxassislarimiz bor.",
    "Axborot xavfsizligi muhim masala hisoblanadi.",
    "Backup va disaster recovery yechimlarini taqdim etamiz.",
    "Virtual serverlar orqali resurslarni tejash mumkin.",

    # Technical Terms
    "Oracle database bilan ishlaymiz.",
    "IBM serverlarini sozlash xizmatini ko'rsatamiz.",
    "Linux operatsion tizimini o'rnatamiz.",
    "Windows Server konfiguratsiyasi.",
    "VMware virtualizatsiya platformasi.",
    "Cisco tarmoq uskunalari.",
    "Huawei serverlar va storage tizimlari.",
    "Lenovo kompyuterlari va serverlar.",

    # Cloud Services
    "Cloud migration xizmatlari mavjud.",
    "Hybrid cloud yechimlarini joriy qilamiz.",
    "Private cloud infratuzilmasini quramiz.",
    "Public cloud xizmatlaridan foydalanamiz.",
    "Multi-cloud strategiyasini ishlab chiqamiz.",

    # Security
    "Firewall sozlamalari muhim.",
    "VPN ulanishini ta'minlaymiz.",
    "SSL sertifikatlarini o'rnatamiz.",
    "Penetration testing xizmati.",
    "Security audit o'tkazamiz.",

    # Support
    "Yigirma to'rt soat texnik yordam ko'rsatamiz.",
    "Monitoring xizmatlari mavjud.",
    "Help desk tizimini joriy qildik.",
    "Ticketing sistema orqali muammolarni hal qilamiz.",

    # General IT
    "API integratsiyasi bo'yicha yordam beramiz.",
    "Frontend va backend ishlab chiqish.",
    "Mobile ilova yaratish xizmatlari.",
    "Web sayt ishlab chiqish.",
    "DevOps amaliyotlarini joriy qilamiz.",
    "CI CD pipeline quramiz.",
    "Docker konteynerlaridan foydalanamiz.",
    "Kubernetes orkestratsiyasi.",
    "Microservices arxitekturasi.",
    "REST API ishlab chiqish.",

    # Common Phrases
    "Qanday yordam bera olaman?",
    "Sizga qanday xizmat kerak?",
    "Texnik muammo bormi?",
    "Server ishlamayaptimi?",
    "Internet ulanishi yo'qmi?",
    "Parolni unutdingizmi?",
    "Yangi hisob ochish kerakmi?",
    "Ma'lumotlar bazasi bilan muammo bormi?",
]

# TTS Settings
NOISE_SCALE = 0.5
NOISE_SCALE_W = 0.7
LENGTH_SCALE = 1.15  # Slightly slower for clarity

def to_ipa(text: str) -> str:
    """Convert Uzbek text to IPA"""
    text = text.lower()
    for apo in ["'", "'", "`", "ʻ", "ʼ"]:
        text = text.replace(apo, "'")
    text = text.replace("o'", "ø")
    text = text.replace("g'", "ʁ")
    text = text.replace("sh", "ʃ")
    text = text.replace("ch", "tʃ")
    text = text.replace("ng", "ŋ")
    text = text.replace("x", "χ")
    text = text.replace("j", "dʒ")
    text = text.replace("y", "j")
    text = text.replace("a", "ɑ")
    text = text.replace("o", "ɔ")
    text = text.replace("'", "")
    return text

def load_existing_sentences(max_sentences=500):
    """Load sentences from existing training data"""
    sentences = []
    metadata_file = BASE_DIR / "uzbek_tts_project/metadata/train_single_final.txt"

    if metadata_file.exists():
        with open(metadata_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            random.shuffle(lines)
            for line in lines[:max_sentences]:
                parts = line.strip().split('|')
                if len(parts) >= 2:
                    text = parts[1].strip()
                    if 10 < len(text) < 200:  # Filter by length
                        sentences.append(text)

    return sentences

def generate_audio(synthesizer, text, output_path):
    """Generate audio from text using TTS V3"""
    ipa_text = to_ipa(text)

    wav = synthesizer.tts(
        ipa_text,
        noise_scale=NOISE_SCALE,
        noise_scale_w=NOISE_SCALE_W,
        length_scale=LENGTH_SCALE
    )

    wav_np = np.array(wav, dtype=np.float32)

    # Light noise reduction
    clean_wav = nr.reduce_noise(y=wav_np, sr=22050, prop_decrease=0.3, stationary=True)

    # Normalize
    max_val = np.max(np.abs(clean_wav))
    if max_val > 0:
        clean_wav = clean_wav * (0.95 / max_val)

    # Save as 16kHz for Whisper
    import librosa
    wav_16k = librosa.resample(clean_wav, orig_sr=22050, target_sr=16000)
    sf.write(output_path, wav_16k, 16000)

    return len(wav_16k) / 16000  # Duration in seconds

def main():
    print("=" * 60)
    print("SYNTHETIC STT DATASET GENERATOR")
    print("=" * 60)

    # Create directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    # Load TTS model
    print("\n[1/4] Loading TTS V3 model...")
    synthesizer = Synthesizer(
        tts_checkpoint=str(V3_MODEL),
        tts_config_path=str(V3_CONFIG),
        use_cuda=torch.cuda.is_available()
    )
    print(f"Model loaded! CUDA: {torch.cuda.is_available()}")

    # Collect sentences
    print("\n[2/4] Collecting sentences...")

    # IT domain sentences (high priority - generate 3x)
    all_sentences = []
    for sent in IT_SENTENCES:
        all_sentences.extend([sent] * 3)  # Triple IT sentences

    # Add general Uzbek sentences
    general_sentences = load_existing_sentences(max_sentences=300)
    all_sentences.extend(general_sentences)

    # Shuffle
    random.shuffle(all_sentences)

    print(f"Total sentences: {len(all_sentences)}")
    print(f"  - IT domain: {len(IT_SENTENCES) * 3}")
    print(f"  - General: {len(general_sentences)}")

    # Generate audio
    print("\n[3/4] Generating synthetic audio...")

    manifest = []
    total_duration = 0

    for i, text in enumerate(tqdm(all_sentences, desc="Generating")):
        audio_filename = f"synthetic_{i:05d}.wav"
        audio_path = AUDIO_DIR / audio_filename

        try:
            duration = generate_audio(synthesizer, text, str(audio_path))
            total_duration += duration

            manifest.append({
                "audio": f"audio/{audio_filename}",
                "text": text,
                "duration": round(duration, 2)
            })
        except Exception as e:
            print(f"\nError on sentence {i}: {e}")
            continue

    # Save manifest
    print("\n[4/4] Saving manifest...")

    manifest_path = OUTPUT_DIR / "manifest.json"
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    # Also save in HuggingFace format
    hf_manifest = OUTPUT_DIR / "train.json"
    with open(hf_manifest, 'w', encoding='utf-8') as f:
        for item in manifest:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print("\n" + "=" * 60)
    print("DATASET GENERATION COMPLETE!")
    print("=" * 60)
    print(f"Total samples: {len(manifest)}")
    print(f"Total duration: {total_duration/60:.1f} minutes ({total_duration/3600:.2f} hours)")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Manifest: {manifest_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()
