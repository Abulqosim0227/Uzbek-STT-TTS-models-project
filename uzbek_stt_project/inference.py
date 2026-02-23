#!/usr/bin/env python3
"""
Uzbek STT Inference Script
By Abulqosim Rafiqov - December 2025

Two inference modes:
1. HuggingFace Transformers (default) - for testing
2. Faster-Whisper (production) - 4x faster, requires model conversion

Usage:
    python inference.py audio.wav
    python inference.py audio.wav --fast  # Use Faster-Whisper if available
"""

import argparse
import time
from pathlib import Path
import torch
import numpy as np
import librosa

# === CONFIGURATION ===
MODEL_DIR = Path("/mnt/c/Users/Admin/Desktop/voice_dataset/ISSAI_USC/uzbek_stt_project/output/final_model")
FASTER_WHISPER_DIR = Path("/mnt/c/Users/Admin/Desktop/voice_dataset/ISSAI_USC/uzbek_stt_project/output/faster_whisper")
SAMPLE_RATE = 16000


class UzbekSTT:
    """
    Uzbek Speech-to-Text using fine-tuned Whisper with LoRA.
    """

    def __init__(self, model_path: str = None, use_fast: bool = False):
        """
        Initialize the STT model.

        Args:
            model_path: Path to the trained model (or None for default)
            use_fast: Use Faster-Whisper for faster inference
        """
        self.use_fast = use_fast
        model_path = Path(model_path) if model_path else MODEL_DIR

        if use_fast:
            self._load_faster_whisper(model_path)
        else:
            self._load_transformers(model_path)

    def _load_transformers(self, model_path: Path):
        """Load model using HuggingFace Transformers."""
        print(f"Loading model from {model_path}...")

        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        from peft import PeftModel

        # Set dtype based on device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        # Check if this is a LoRA model or merged model
        adapter_config = model_path / "adapter_config.json"

        if adapter_config.exists():
            # LoRA model - load base + adapters
            print("Loading LoRA model...")
            base_model = WhisperForConditionalGeneration.from_pretrained(
                "openai/whisper-large-v3",
                torch_dtype=self.dtype,
            )
            self.model = PeftModel.from_pretrained(base_model, str(model_path))
        else:
            # Merged model
            print("Loading merged model...")
            self.model = WhisperForConditionalGeneration.from_pretrained(
                str(model_path),
                torch_dtype=self.dtype,
            )

        self.processor = WhisperProcessor.from_pretrained(str(model_path))

        # Move to device
        self.model = self.model.to(self.device)
        self.model.eval()

        print(f"Model loaded! Using {self.device}")

    def _load_faster_whisper(self, model_path: Path):
        """Load model using Faster-Whisper (CTranslate2)."""
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise ImportError("faster-whisper not installed. Run: pip install faster-whisper")

        # Check for CTranslate2 converted model
        ct2_path = FASTER_WHISPER_DIR
        if not ct2_path.exists():
            raise FileNotFoundError(
                f"Faster-Whisper model not found at {ct2_path}. "
                "Run export_to_ctranslate2.py first."
            )

        print(f"Loading Faster-Whisper model from {ct2_path}...")
        self.model = WhisperModel(
            str(ct2_path),
            device="cuda" if torch.cuda.is_available() else "cpu",
            compute_type="float16" if torch.cuda.is_available() else "float32",
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Faster-Whisper loaded! Using {self.device}")

    def transcribe(self, audio_path: str, return_segments: bool = False) -> str:
        """
        Transcribe an audio file to text.

        Args:
            audio_path: Path to audio file (wav, mp3, etc.)
            return_segments: If True, return list of (start, end, text) tuples

        Returns:
            Transcribed text or list of segments
        """
        # Load and resample audio
        audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE)

        if self.use_fast:
            return self._transcribe_fast(audio, return_segments)
        else:
            return self._transcribe_transformers(audio, return_segments)

    def _transcribe_transformers(self, audio: np.ndarray, return_segments: bool) -> str:
        """Transcribe using HuggingFace Transformers."""
        # Extract features and cast to correct dtype
        input_features = self.processor.feature_extractor(
            audio,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt"
        ).input_features.to(self.device, dtype=self.dtype)

        # Set language to Uzbek
        forced_decoder_ids = self.processor.get_decoder_prompt_ids(
            language="uz",
            task="transcribe"
        )

        # Generate
        with torch.no_grad():
            predicted_ids = self.model.generate(
                input_features,
                forced_decoder_ids=forced_decoder_ids,
                max_length=225,
            )

        # Decode
        transcription = self.processor.batch_decode(
            predicted_ids,
            skip_special_tokens=True
        )[0]

        return transcription.strip()

    def _transcribe_fast(self, audio: np.ndarray, return_segments: bool) -> str:
        """Transcribe using Faster-Whisper."""
        segments, info = self.model.transcribe(
            audio,
            language="uz",
            beam_size=5,
            word_timestamps=return_segments,
        )

        if return_segments:
            return [(seg.start, seg.end, seg.text) for seg in segments]
        else:
            return " ".join([seg.text for seg in segments]).strip()

    def transcribe_realtime(self, audio_chunk: np.ndarray) -> str:
        """
        Transcribe a short audio chunk for real-time applications.
        Audio should be 16kHz mono float32.
        """
        return self._transcribe_transformers(audio_chunk, False)

    def transcribe_batch(self, audio_paths: list) -> list:
        """
        Transcribe multiple audio files.

        Args:
            audio_paths: List of paths to audio files

        Returns:
            List of transcriptions
        """
        results = []
        for path in audio_paths:
            results.append(self.transcribe(path))
        return results


def main():
    parser = argparse.ArgumentParser(description="Uzbek Speech-to-Text")
    parser.add_argument("audio", help="Path to audio file")
    parser.add_argument("--fast", action="store_true", help="Use Faster-Whisper")
    parser.add_argument("--model", type=str, help="Path to model directory")
    parser.add_argument("--segments", action="store_true", help="Return timestamped segments")

    args = parser.parse_args()

    # Check if audio file exists
    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"Error: Audio file not found: {audio_path}")
        return 1

    # Initialize STT
    try:
        stt = UzbekSTT(model_path=args.model, use_fast=args.fast)
    except Exception as e:
        print(f"Error loading model: {e}")
        return 1

    # Transcribe
    print(f"\nTranscribing: {audio_path}")
    start_time = time.time()

    try:
        result = stt.transcribe(str(audio_path), return_segments=args.segments)
    except Exception as e:
        print(f"Error during transcription: {e}")
        return 1

    elapsed = time.time() - start_time

    # Print result
    print("\n" + "=" * 60)
    if args.segments:
        print("Timestamped segments:")
        for start, end, text in result:
            print(f"  [{start:.2f}s - {end:.2f}s] {text}")
    else:
        print(f"Transcription: {result}")
    print("=" * 60)
    print(f"Time: {elapsed:.2f}s")

    return 0


if __name__ == "__main__":
    exit(main())
