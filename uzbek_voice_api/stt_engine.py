#!/usr/bin/env python3
"""
STT Engine - Uzbek Speech-to-Text
Uses Whisper + LoRA fine-tuned for Uzbek
"""

import torch
import numpy as np
import librosa
from pathlib import Path

class STTEngine:
    """Uzbek STT Engine using fine-tuned Whisper"""

    MODEL_DIR = Path("/mnt/c/Users/Admin/Desktop/voice_dataset/ISSAI_USC/uzbek_stt_project/output/final_model")
    SAMPLE_RATE = 16000

    # IT Vocabulary Prompt Injection (improves IT term recognition without retraining)
    IT_PROMPT = "Server, Kubernetes, Oracle, API, Cloud, Linux, Python, DevOps, Docker, Deploy, Backend, Frontend, Database, Cyber Security, GitHub, React, Node.js"

    def __init__(self):
        print("[STT] Loading Uzbek Whisper model...")

        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        from peft import PeftModel

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        # Load base Whisper + LoRA adapters
        print("[STT] Loading base Whisper model...")
        base_model = WhisperForConditionalGeneration.from_pretrained(
            "openai/whisper-large-v3",
            torch_dtype=self.dtype,
        )

        print("[STT] Loading LoRA adapters...")
        self.model = PeftModel.from_pretrained(base_model, str(self.MODEL_DIR))
        self.processor = WhisperProcessor.from_pretrained(str(self.MODEL_DIR))

        self.model = self.model.to(self.device)
        self.model.eval()

        print(f"[STT] Model loaded! Using {self.device}")

    def transcribe(self, audio_path: str) -> tuple:
        """
        Transcribe audio file to text

        Args:
            audio_path: Path to audio file

        Returns:
            Tuple of (text, confidence)
        """
        # Load and resample audio
        audio, sr = librosa.load(audio_path, sr=self.SAMPLE_RATE)

        # Extract features
        input_features = self.processor.feature_extractor(
            audio,
            sampling_rate=self.SAMPLE_RATE,
            return_tensors="pt"
        ).input_features.to(self.device, dtype=self.dtype)

        # Get IT vocabulary prompt IDs for better recognition
        prompt_ids = self.processor.get_prompt_ids(self.IT_PROMPT, return_tensors="pt")
        prompt_ids = prompt_ids.to(self.device)

        # Generate with prompt injection for IT terms
        with torch.no_grad():
            outputs = self.model.generate(
                input_features,
                prompt_ids=prompt_ids,
                language="uz",
                task="transcribe",
                max_length=225,
                return_dict_in_generate=True,
                output_scores=True
            )

        # Decode
        transcription = self.processor.batch_decode(
            outputs.sequences,
            skip_special_tokens=True
        )[0].strip()

        # Calculate rough confidence (simplified)
        # In production, you'd compute this from logits properly
        confidence = 0.85  # Placeholder - model generally performs well

        return transcription, confidence


# Test
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python stt_engine.py <audio_file>")
        sys.exit(1)

    engine = STTEngine()
    text, conf = engine.transcribe(sys.argv[1])
    print(f"Text: {text}")
    print(f"Confidence: {conf}")
