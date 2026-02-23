# O'zbek Ovoz API

Professional Text-to-Speech and Speech-to-Text API for Uzbek Language.

## Features

- **TTS (Text-to-Speech)**: Convert Uzbek text to natural speech using VITS model
- **STT (Speech-to-Text)**: Transcribe Uzbek audio using fine-tuned Whisper
- **Interactive Dashboard**: Beautiful web UI for testing
- **Auto-generated API Docs**: Swagger + ReDoc documentation
- **Production Ready**: Nginx config + systemd service included

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the server
chmod +x run.sh
./run.sh
```

Server starts at: http://localhost:8000

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/dashboard` | GET | Interactive demo UI |
| `/docs` | GET | Swagger API documentation |
| `/redoc` | GET | ReDoc documentation |
| `/api/tts` | POST | Text-to-Speech |
| `/api/stt` | POST | Speech-to-Text |
| `/api/health` | GET | Health check |
| `/api/models` | GET | Model information |

## API Usage

### TTS (Text-to-Speech)

```bash
curl -X POST "http://localhost:8000/api/tts" \
  -H "Content-Type: application/json" \
  -d '{"text": "Salom! Bugun ob-havo juda yaxshi.", "speed": 1.0}'
```

Response:
```json
{
  "success": true,
  "audio_url": "/audio/tts_abc123.wav",
  "duration": 2.5,
  "text": "Salom! Bugun ob-havo juda yaxshi.",
  "processing_time": 1.234
}
```

### STT (Speech-to-Text)

```bash
curl -X POST "http://localhost:8000/api/stt" \
  -F "audio=@recording.wav"
```

Response:
```json
{
  "success": true,
  "text": "Salom bugun ob-havo juda yaxshi",
  "confidence": 0.95,
  "processing_time": 2.456
}
```

## Models

### TTS Model
- **Architecture**: VITS (Variational Inference Text-to-Speech)
- **Training**: 540,000 steps on ISSAI USC dataset
- **Speaker**: Single speaker (natural Uzbek female voice)
- **Sample Rate**: 22,050 Hz
- **Features**: Noise reduction, volume normalization

### STT Model
- **Base**: OpenAI Whisper Large V3
- **Fine-tuning**: LoRA adapters trained on Uzbek speech
- **Language**: Uzbek (uz)
- **Features**: High accuracy, noise robust

## Production Deployment

### 1. Setup Nginx

```bash
# Copy nginx config
sudo cp nginx.conf /etc/nginx/sites-available/uzbek-voice-api
sudo ln -s /etc/nginx/sites-available/uzbek-voice-api /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
```

### 2. Setup Systemd Service

```bash
# Copy service file
sudo cp uzbek-voice-api.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable uzbek-voice-api
sudo systemctl start uzbek-voice-api
```

### 3. Check Status

```bash
sudo systemctl status uzbek-voice-api
curl http://localhost:8000/api/health
```

## Project Structure

```
uzbek_voice_api/
├── main.py              # FastAPI application
├── tts_engine.py        # TTS model wrapper
├── stt_engine.py        # STT model wrapper
├── dashboard.html       # Web UI
├── requirements.txt     # Python dependencies
├── nginx.conf           # Nginx configuration
├── uzbek-voice-api.service  # Systemd service
├── run.sh               # Development run script
└── audio_output/        # Generated audio files
```

## Author

Abulqosim Rafiqov - December 2025

## License

MIT License
