#!/usr/bin/env python3
"""
JARVIS STT Engine - Ultra-Fast Uzbek Speech-to-Text
Uses Faster-Whisper (CTranslate2) with Context Prompting

Features:
- 4-5x faster than standard Whisper
- Context prompting for IT/Technical terms
- Optimized for Nihol company vocabulary
- Mixed Uzbek/English support
"""

import os
import time

# Set cuDNN library path for GPU support
cudnn_path = "/home/abulqosim/.local/lib/python3.10/site-packages/nvidia/cudnn/lib"
cublas_path = "/home/abulqosim/.local/lib/python3.10/site-packages/nvidia/cublas/lib"
if os.path.exists(cudnn_path):
    os.environ["LD_LIBRARY_PATH"] = f"{cudnn_path}:{cublas_path}:" + os.environ.get("LD_LIBRARY_PATH", "")

from faster_whisper import WhisperModel

class JarvisSTT:
    """Ultra-fast Uzbek STT with context awareness"""

    # Context prompt - teaches the model your vocabulary
    CONTEXT_PROMPT = """Bu suhbat Nihol IT kompaniyasi haqida.
    Bizda Cloud Computing, Server, Data Center, Oracle, IBM, Linux,
    Cisco, Huawei, Lenovo, Storage, Backup, Cybersecurity,
    kiberxavfsizlik, axborot xavfsizligi, infratuzilma,
    virtual server, VMware, database, API, frontend, backend,
    Python, JavaScript, React, Node.js, Docker, Kubernetes bor.
    O'zbekiston, Toshkent, dasturchi, muhandis, texnologiya."""

    def __init__(self, model_size="large-v3", device="auto", compute_type="auto"):
        """
        Initialize Jarvis STT Engine

        Args:
            model_size: "large-v3" recommended for best accuracy
            device: "auto", "cuda", or "cpu"
            compute_type: "auto", "float16", "int8", or "float32"
        """
        print(f"[JARVIS STT] Loading Faster-Whisper {model_size}...")
        start = time.time()

        # Auto-detect best settings (GPU preferred)
        if device == "auto":
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if compute_type == "auto":
            compute_type = "float16" if device == "cuda" else "int8"

        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            download_root="/mnt/c/Users/Admin/Desktop/voice_dataset/ISSAI_USC/uzbek_stt_project/whisper_models"
        )

        load_time = time.time() - start
        print(f"[JARVIS STT] Model loaded in {load_time:.1f}s on {device.upper()}")

    def transcribe(self, audio_path: str, language: str = "uz",
                   use_context: bool = True, beam_size: int = 5) -> dict:
        """
        Transcribe audio to text

        Args:
            audio_path: Path to audio file (WAV, MP3, etc.)
            language: Language code ("uz" for Uzbek, "en" for English, None for auto)
            use_context: Use context prompting for better accuracy
            beam_size: Beam size for decoding (higher = more accurate but slower)

        Returns:
            dict with text, segments, confidence, and timing info
        """
        start = time.time()

        # Prepare prompting
        initial_prompt = self.CONTEXT_PROMPT if use_context else None

        # Transcribe with Faster-Whisper
        segments, info = self.model.transcribe(
            audio_path,
            language=language,
            beam_size=beam_size,
            initial_prompt=initial_prompt,
            vad_filter=True,  # Filter out silence
            vad_parameters=dict(
                min_silence_duration_ms=500,  # 500ms silence = end of speech
                speech_pad_ms=200,
            ),
        )

        # Collect results
        segments_list = []
        full_text = []

        for segment in segments:
            segments_list.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip(),
                "confidence": segment.avg_logprob,
            })
            full_text.append(segment.text.strip())

        transcription_time = time.time() - start
        text = " ".join(full_text)

        # Calculate average confidence
        avg_confidence = 0.0
        if segments_list:
            avg_confidence = sum(s["confidence"] for s in segments_list) / len(segments_list)
            # Convert log prob to 0-1 scale (approximate)
            avg_confidence = min(1.0, max(0.0, 1.0 + avg_confidence / 5))

        return {
            "text": text,
            "language": info.language,
            "language_probability": info.language_probability,
            "confidence": round(avg_confidence, 3),
            "duration": info.duration,
            "processing_time": round(transcription_time, 3),
            "segments": segments_list,
            "realtime_factor": round(transcription_time / max(info.duration, 0.1), 3),
        }

    def transcribe_simple(self, audio_path: str) -> tuple:
        """
        Simple transcription - returns (text, confidence)
        Compatible with old STTEngine interface
        """
        result = self.transcribe(audio_path)
        return result["text"], result["confidence"]


# Singleton instance for API
_jarvis_instance = None

def get_jarvis_stt():
    """Get or create Jarvis STT instance"""
    global _jarvis_instance
    if _jarvis_instance is None:
        _jarvis_instance = JarvisSTT()
    return _jarvis_instance


# Test
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python stt_engine_jarvis.py <audio_file>")
        print("\nExample:")
        print("  python stt_engine_jarvis.py test_audio.wav")
        sys.exit(1)

    audio_path = sys.argv[1]

    print("\n" + "=" * 60)
    print("JARVIS STT TEST")
    print("=" * 60)

    jarvis = JarvisSTT()
    result = jarvis.transcribe(audio_path)

    print(f"\n📝 Text: {result['text']}")
    print(f"🌐 Language: {result['language']} ({result['language_probability']:.1%})")
    print(f"📊 Confidence: {result['confidence']:.1%}")
    print(f"⏱️ Audio Duration: {result['duration']:.2f}s")
    print(f"⚡ Processing Time: {result['processing_time']:.3f}s")
    print(f"🚀 Realtime Factor: {result['realtime_factor']:.2f}x")

    if result['segments']:
        print(f"\n📋 Segments ({len(result['segments'])}):")
        for i, seg in enumerate(result['segments'][:5], 1):
            print(f"   {i}. [{seg['start']:.1f}s-{seg['end']:.1f}s] {seg['text']}")
