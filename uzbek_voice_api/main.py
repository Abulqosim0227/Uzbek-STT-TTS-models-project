#!/usr/bin/env python3
"""
UZBEK VOICE API - TTS + STT
Professional FastAPI backend for Uzbek language voice processing

Endpoints:
    POST /api/tts - Text to Speech
    POST /api/stt - Speech to Text
    GET /api/health - Health check
    GET /api/models - List available models
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import os
import uuid
import time

# Import engines
from tts_engine import TTSEngine
from stt_engine import STTEngine

# === APP SETUP ===
app = FastAPI(
    title="Uzbek Voice API",
    description="""
# O'zbek Ovoz API

Professional Text-to-Speech and Speech-to-Text API for Uzbek language.

## Features
- **TTS (Text-to-Speech)**: Convert Uzbek text to natural speech
- **STT (Speech-to-Text)**: Transcribe Uzbek audio to text
- **High Quality**: Trained on native Uzbek speech data
- **Fast**: Optimized for production use

## Models
- TTS: VITS model trained on ISSAI USC dataset (540k steps)
- STT: Whisper + LoRA fine-tuned for Uzbek

## Usage
See the interactive documentation below for API details.
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS for dashboard and ngrok
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,  # Must be False when allow_origins=["*"]
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,
)

# === OUTPUT FOLDERS ===
AUDIO_OUTPUT = "/mnt/c/Users/Admin/Desktop/voice_dataset/ISSAI_USC/uzbek_voice_api/audio_output"
os.makedirs(AUDIO_OUTPUT, exist_ok=True)

# === GLOBAL ENGINES (lazy load) ===
tts_engines = {}  # Store both V1 and V3
stt_engine: Optional[STTEngine] = None

def get_tts(model_version="v1"):
    global tts_engines
    if model_version not in tts_engines:
        tts_engines[model_version] = TTSEngine(model_version=model_version)
    return tts_engines[model_version]

def get_stt():
    global stt_engine
    if stt_engine is None:
        stt_engine = STTEngine()
    return stt_engine

# === REQUEST/RESPONSE MODELS ===
class TTSRequest(BaseModel):
    text: str = Field(..., description="Uzbek text to convert to speech", example="Salom! Bugun ob-havo juda yaxshi.")
    speed: float = Field(default=1.0, ge=0.5, le=2.0, description="Speech speed (0.5-2.0)")
    model: str = Field(default="v1", description="Model version: v1 (fast) or v3 (IPA, accurate)")

class TTSResponse(BaseModel):
    success: bool
    audio_url: str
    duration: float
    text: str
    processed_text: str
    model: str
    processing_time: float

class STTResponse(BaseModel):
    success: bool
    text: str
    confidence: float
    processing_time: float

class HealthResponse(BaseModel):
    status: str
    tts_loaded: bool
    stt_loaded: bool
    version: str

class ModelsResponse(BaseModel):
    tts: dict
    stt: dict

# === ENDPOINTS ===

@app.get("/", response_class=HTMLResponse)
async def root():
    """Redirect to dashboard"""
    return """
    <html>
        <head>
            <meta http-equiv="refresh" content="0; url=/dashboard" />
        </head>
        <body>
            <p>Redirecting to <a href="/dashboard">Dashboard</a>...</p>
        </body>
    </html>
    """

@app.get("/api/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check API health and model status"""
    return {
        "status": "healthy",
        "tts_loaded": len(tts_engines) > 0,
        "stt_loaded": stt_engine is not None,
        "version": "1.0.0"
    }

@app.get("/api/models", response_model=ModelsResponse, tags=["System"])
async def list_models():
    """List available models and their details"""
    return {
        "tts": {
            "name": "VITS Uzbek",
            "version": "V1 (540k steps)",
            "description": "Single-speaker Uzbek TTS trained on ISSAI USC dataset",
            "sample_rate": 22050,
            "features": ["Natural speech", "Punctuation-aware pauses", "Noise reduction"]
        },
        "stt": {
            "name": "Whisper Uzbek",
            "version": "V2 (LoRA fine-tuned)",
            "description": "Whisper model fine-tuned for Uzbek speech recognition",
            "features": ["High accuracy", "Fast inference", "Noise robust"]
        }
    }

@app.post("/api/tts", response_model=TTSResponse, tags=["TTS"])
async def text_to_speech(request: TTSRequest):
    """
    Convert Uzbek text to speech

    - **text**: The Uzbek text to convert (required)
    - **speed**: Speech speed multiplier (default: 1.0)
    - **model**: Model version - "v1" (fast, 540k steps) or "v3" (IPA, accurate pronunciation)

    **Features:**
    - Numbers converted to Uzbek words (1970 → "ming to'qqiz yuz yetmish")
    - Tech words pronounced correctly (SQL → "es kyu el", JS → "jey es")

    Returns audio file URL and metadata.
    """
    start_time = time.time()

    try:
        # Get engine for requested model
        model_version = request.model.lower()
        if model_version not in ["v1", "v3"]:
            model_version = "v1"

        engine = get_tts(model_version)

        # Generate unique filename
        filename = f"tts_{model_version}_{uuid.uuid4().hex[:8]}.wav"
        filepath = os.path.join(AUDIO_OUTPUT, filename)

        # Get processed text (after number/tech word conversion)
        processed_text = engine.clean_text(request.text)
        if model_version == "v3":
            processed_text = engine.to_ipa(processed_text)

        # Generate speech
        duration = engine.synthesize(request.text, filepath, speed=request.speed)

        processing_time = time.time() - start_time

        return {
            "success": True,
            "audio_url": f"/audio/{filename}",
            "duration": duration,
            "text": request.text,
            "processed_text": processed_text,
            "model": model_version.upper(),
            "processing_time": round(processing_time, 3)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/stt", response_model=STTResponse, tags=["STT"])
async def speech_to_text(audio: UploadFile = File(..., description="Audio file (WAV, MP3, etc.)")):
    """
    Transcribe Uzbek speech to text

    - **audio**: Audio file to transcribe (required)

    Supported formats: WAV, MP3, OGG, FLAC
    """
    start_time = time.time()

    try:
        engine = get_stt()

        # Save uploaded file temporarily
        temp_path = os.path.join(AUDIO_OUTPUT, f"temp_{uuid.uuid4().hex[:8]}")
        with open(temp_path, "wb") as f:
            content = await audio.read()
            f.write(content)

        # Transcribe
        text, confidence = engine.transcribe(temp_path)

        # Cleanup
        os.remove(temp_path)

        processing_time = time.time() - start_time

        return {
            "success": True,
            "text": text,
            "confidence": confidence,
            "processing_time": round(processing_time, 3)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# === SERVE AUDIO FILES ===
@app.get("/audio/{filename}")
async def get_audio(filename: str):
    """Serve generated audio files"""
    filepath = os.path.join(AUDIO_OUTPUT, filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(filepath, media_type="audio/wav")

# === SERVE DASHBOARD ===
@app.get("/dashboard", response_class=HTMLResponse, tags=["Dashboard"])
async def dashboard():
    """Interactive demo dashboard"""
    with open("/mnt/c/Users/Admin/Desktop/voice_dataset/ISSAI_USC/uzbek_voice_api/dashboard.html", "r", encoding="utf-8") as f:
        return f.read()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
