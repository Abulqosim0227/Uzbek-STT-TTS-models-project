#!/usr/bin/env python3
"""
JARVIS PRO - Final Uzbek Voice Assistant
By Abulqosim Rafiqov - December 2025

2 weeks of engineering distilled into one file:
- STT: Original LoRA (26.7% WER) - The Winner
- TTS: V1 Single Speaker (540k) - The Natural One
- IT Vocabulary: Prompt Injection - The Cheat Code
"""

import torch
import sounddevice as sd
import numpy as np
import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel
from TTS.utils.synthesizer import Synthesizer
import noisereduce as nr

# ================= CONFIGURATION =================
# STT: Original LoRA Model (26.7% WER - Best)
STT_MODEL = "/mnt/c/Users/Admin/Desktop/voice_dataset/ISSAI_USC/uzbek_stt_project/output/final_model"
BASE_WHISPER = "openai/whisper-large-v3"

# TTS: V1 Single Speaker (540k steps - Natural)
TTS_MODEL = "/mnt/c/Users/Admin/Desktop/voice_dataset/ISSAI_USC/uzbek_tts_project/training_output/uzbek_tts_single_speaker-December-04-2025_08+50AM-0000000/checkpoint_540000.pth"
TTS_CONFIG = "/mnt/c/Users/Admin/Desktop/voice_dataset/ISSAI_USC/uzbek_tts_project/training_output/uzbek_tts_single_speaker-December-04-2025_08+50AM-0000000/config.json"

# IT Vocabulary Prompt (The Cheat Code - no training needed!)
IT_PROMPT = "Server, Kubernetes, Oracle, API, Cloud, Linux, Python, DevOps, Docker, Deploy, Backend, Frontend, Database, Cyber Security, GitHub, React, Node.js"

SAMPLE_RATE_STT = 16000
SAMPLE_RATE_TTS = 22050
# =================================================

print("=" * 60)
print("JARVIS PRO - Uzbek Voice Assistant")
print("=" * 60)

# Load STT (Ears)
print("\n[1/2] Loading STT (Ears)...")
processor = WhisperProcessor.from_pretrained(STT_MODEL)
base_model = WhisperForConditionalGeneration.from_pretrained(
    BASE_WHISPER,
    torch_dtype=torch.float16
)
stt_model = PeftModel.from_pretrained(base_model, STT_MODEL)
stt_model = stt_model.merge_and_unload().cuda().eval()
print("      STT loaded! (26.7% WER)")

# Load TTS (Mouth)
print("\n[2/2] Loading TTS (Mouth)...")
tts = Synthesizer(
    tts_checkpoint=TTS_MODEL,
    tts_config_path=TTS_CONFIG,
    use_cuda=True
)
print("      TTS loaded! (540k steps)")

print("\n" + "=" * 60)
print("System Ready!")
print("=" * 60)


def listen(duration=5):
    """Record and transcribe Uzbek speech with IT vocabulary boost"""
    print(f"\n🔴 Tinglayapman... ({duration} sekund)")

    # Record audio
    recording = sd.rec(
        int(duration * SAMPLE_RATE_STT),
        samplerate=SAMPLE_RATE_STT,
        channels=1,
        dtype='float32'
    )
    sd.wait()
    print("⏹️ Qayta ishlanmoqda...")

    audio = recording.flatten()

    # Normalize
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio)) * 0.95

    # Process with Whisper
    inputs = processor(
        audio,
        sampling_rate=SAMPLE_RATE_STT,
        return_tensors="pt"
    ).input_features.cuda().half()

    # Generate with IT vocabulary prompt injection
    # The prompt_ids help Whisper recognize IT terms
    prompt_ids = processor.get_prompt_ids(IT_PROMPT, return_tensors="pt").cuda()

    with torch.no_grad():
        generated_ids = stt_model.generate(
            inputs,
            prompt_ids=prompt_ids,
            language="uz",
            task="transcribe"
        )

    # Decode
    transcription = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0].strip()

    print(f"📝 Siz aytdingiz: '{transcription}'")
    return transcription.lower()


def speak(text):
    """Synthesize and play Uzbek speech"""
    print(f"🤖 Jarvis: {text}")

    # Generate audio
    wav = tts.tts(
        text,
        noise_scale=0.667,      # Clean voice
        noise_scale_w=0.8,      # Smooth flow
        length_scale=1.0        # Normal speed
    )

    wav_np = np.array(wav, dtype=np.float32)

    # Add padding to prevent cutoff
    padding = np.zeros(int(SAMPLE_RATE_TTS * 0.2))
    wav_np = np.concatenate([wav_np, padding])

    # Light noise reduction
    clean_wav = nr.reduce_noise(
        y=wav_np,
        sr=SAMPLE_RATE_TTS,
        prop_decrease=0.3,
        stationary=True
    )

    # Normalize
    if np.max(np.abs(clean_wav)) > 0:
        clean_wav = clean_wav / np.max(np.abs(clean_wav)) * 0.9

    # Play
    sd.play(clean_wav, SAMPLE_RATE_TTS)
    sd.wait()


def process_command(text):
    """Simple command processor - expand as needed"""

    # IT Commands
    if "server" in text:
        return "Serverlar stabil ishlamoqda."
    elif "kubernetes" in text or "kubernetis" in text:
        return "Kubernetes klasterlari tekshirilmoqda."
    elif "oracle" in text:
        return "Oracle bazasi bilan aloqa o'rnatildi."
    elif "docker" in text:
        return "Docker konteynerlar ishga tushirildi."
    elif "deploy" in text or "deploy" in text:
        return "Deployment jarayoni boshlandi."
    elif "api" in text:
        return "API endpointlar ishlayapti."
    elif "database" in text or "baza" in text:
        return "Ma'lumotlar bazasi tekshirilmoqda."

    # General
    elif "salom" in text or "assalom" in text:
        return "Assalomu alaykum! Sizga qanday yordam bera olaman?"
    elif "vaqt" in text or "soat" in text:
        from datetime import datetime
        now = datetime.now().strftime("%H:%M")
        return f"Hozir soat {now}."
    elif "rahmat" in text:
        return "Arzimaydi! Yana yordam kerak bo'lsa, ayting."
    elif "xayr" in text or "hayr" in text:
        return "Ko'rishguncha! Yaxshi kun tillayman."

    # Default
    else:
        return "Tushundim. Bajarilmoqda."


# === MAIN LOOP ===
if __name__ == "__main__":
    speak("Assalomu alaykum. Jarvis tizimi ishga tushdi.")

    while True:
        try:
            input("\n[Enter] bosing gapirish uchun...")
            text = listen(duration=5)

            if not text or len(text) < 2:
                speak("Eshitmadim. Iltimos, qaytadan ayting.")
                continue

            response = process_command(text)
            speak(response)

            if "xayr" in text or "hayr" in text:
                break

        except KeyboardInterrupt:
            print("\n\nTizim to'xtatildi.")
            break
        except Exception as e:
            print(f"Xatolik: {e}")
            continue
