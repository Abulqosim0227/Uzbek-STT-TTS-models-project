# Uzbek Voice AI System (STT + TTS)

End-to-end Uzbek language Speech-to-Text and Text-to-Speech system with a production-ready FastAPI server, Telegram bot integration, and a web dashboard.

Fine-tuned **Whisper Large V3** for Uzbek speech recognition achieving **26.7% WER** using LoRA adapters (91 MB). Trained **VITS TTS** models with custom Uzbek-to-IPA phoneme conversion for natural Uzbek speech synthesis. Deployed as REST API and Telegram bot.

---

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                  UZBEK VOICE API                     │
│                 (FastAPI Server)                      │
│          /api/tts    /api/stt    /dashboard           │
└──────────┬───────────────────────────────┬───────────┘
           │                               │
   ┌───────▼───────────┐        ┌──────────▼──────────┐
   │    TTS ENGINE      │        │     STT ENGINE      │
   │  (Text → Speech)   │        │   (Speech → Text)   │
   └───┬────────────┬───┘        └──────────┬──────────┘
       │            │                       │
 ┌─────▼─────┐ ┌───▼──────┐     ┌──────────▼──────────┐
 │ V1 Model  │ │ V3 Model │     │  Whisper Large V3   │
 │ Plain Text│ │ IPA Input│     │  + LoRA Adapters     │
 │ 540k steps│ │ 60k steps│     │  (91 MB fine-tune)   │
 └───────────┘ └──────────┘     └──────────────────────┘
```

## Features

### Speech-to-Text (STT)
- Fine-tuned **OpenAI Whisper Large V3** on 108K+ Uzbek audio samples
- **LoRA adapters** (91 MB) — no need to store the full 3 GB model weights
- **26.7% WER** on clean Uzbek speech
- IT vocabulary prompt injection for technical term recognition (Kubernetes, Docker, API, etc.)
- Supports real-time and batch transcription

### Text-to-Speech (TTS)
- **V1 Model**: VITS trained for 540K steps on plain Uzbek text
- **V3 Model**: VITS trained with IPA (International Phonetic Alphabet) input for better pronunciation of Uzbek-specific sounds (o', g', x, q)
- Custom **Uzbek-to-IPA converter** handling digraphs (sh, ch, ng) and context-dependent rules
- Text preprocessing pipeline: number-to-words, English tech term transliteration, percentage conversion
- Audio post-processing: noise reduction, volume normalization, fade-out

### API & Integration
- **FastAPI** server with TTS/STT endpoints and web dashboard
- **Telegram bot** — send voice to get text, send text to get voice
- Nginx reverse proxy configuration included
- Health check and model info endpoints

---

## Dataset

**ISSAI Uzbek Speech Corpus (USC)** from the Institute of Smart Systems and Artificial Intelligence (ISSAI), Kazakhstan.

| Property | Value |
|---|---|
| Total Samples | 108,387 audio + text pairs |
| Duration | ~100+ hours of speech |
| Speakers | 932 native Uzbek speakers |
| Audio Format | WAV, 16 kHz |
| Train Split | 100,767 samples |
| Dev Split | 3,783 samples |
| Test Split | 3,837 samples |

---

## Project Structure

```
├── uzbek_stt_project/              # Speech-to-Text
│   ├── prepare_dataset.py          # Data preprocessing & normalization
│   ├── finetune_whisper_synthetic.py  # LoRA fine-tuning script
│   ├── inference.py                # STT inference (standard & Faster-Whisper)
│   ├── evaluate_models.py          # WER evaluation
│   ├── test_finetuned_whisper.py   # Model testing
│   └── train_stt_improved.py       # Alternative training script
│
├── uzbek_tts_project/              # Text-to-Speech
│   ├── uzbek_ipa.py                # Uzbek → IPA phoneme converter
│   ├── text_preprocessor.py        # Number/tech word preprocessing
│   ├── prepare_ipa_dataset.py      # Dataset conversion to IPA format
│   ├── train_ipa_tts.py            # V3 IPA model training
│   ├── continue_training.py        # Resume training from checkpoint
│   ├── use_trained_model.py        # V1 inference
│   ├── speak_v3_enhanced.py        # V3 enhanced inference
│   ├── speak_perfect.py            # Refined V3 inference
│   ├── speak_final.py              # Final V3 inference
│   ├── speak_fast.py               # Optimized fast inference
│   ├── config_single_speaker.json  # V1 model config
│   └── config_ipa_v3.json          # V3 model config
│
├── uzbek_voice_api/                # Production API
│   ├── main.py                     # FastAPI server & endpoints
│   ├── stt_engine.py               # STT wrapper with prompt injection
│   ├── tts_engine.py               # TTS wrapper (V1 & V3 support)
│   ├── telegram_bot.py             # Telegram bot integration
│   ├── jarvis_pro.py               # Voice assistant
│   ├── stt_engine_jarvis.py        # Jarvis STT variant
│   ├── requirements.txt            # Python dependencies
│   └── nginx.conf                  # Nginx reverse proxy config
│
├── PROJECT_DOCUMENTATION.txt       # Complete project journey
├── UZBEK_STT_GUIDE.md             # Bilingual STT technical guide
├── UZBEK_STT_GUIDE.txt            # Practical STT implementation guide
└── UZBEK_TEXT_DIALECT_GUIDE.txt   # Uzbek dialect handling solutions
```

> **Note**: Model weights, dataset files, and audio outputs are excluded from this repository due to size. See [Installation](#installation) for download instructions.

---

## Results

### STT Performance (Word Error Rate — lower is better)

| Condition | WER | Accuracy |
|---|---|---|
| Clean speech | **26.7%** | 73.3% |
| TTS-generated audio | 49.7% | 50.3% |
| Noisy audio | 64.6% | 35.4% |

### TTS Models

| Model | Training Steps | Input | Strengths |
|---|---|---|---|
| V1 | 540,000 | Plain text | Fast, stable |
| V3 | 60,000+ | IPA phonemes | Better pronunciation of o', g', x, q |

---

## Tech Stack

| Component | Technology |
|---|---|
| STT Base Model | OpenAI Whisper Large V3 (1.5B params) |
| STT Fine-tuning | LoRA / PEFT (rank=32, alpha=64) |
| TTS Model | VITS (Coqui TTS) |
| Phoneme Conversion | Custom Uzbek → IPA converter |
| API Framework | FastAPI + Uvicorn |
| Task Queue | Celery + Redis |
| Database | PostgreSQL |
| Bot | python-telegram-bot |
| Deployment | Docker, Nginx, Ngrok |
| GPU | NVIDIA RTX 5070 Ti (16 GB VRAM) |

---

## Installation

### Prerequisites

- Python 3.9+
- NVIDIA GPU with CUDA 12.x (recommended)
- ~16 GB VRAM for full STT + TTS inference

### Setup

```bash
# Clone the repository
git clone https://github.com/Abulqosim0227/Uzbek-STT-TTS-models-project.git
cd Uzbek-STT-TTS-models-project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r uzbek_voice_api/requirements.txt

# For RTX 50-series GPUs (Blackwell architecture), install PyTorch nightly:
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

### Download Models

The trained model weights are not included in this repository. To use the system:

**STT (LoRA Adapters)**:
- Base model: `openai/whisper-large-v3` (auto-downloaded from HuggingFace)
- LoRA weights: Place the `final_model/` directory in `uzbek_stt_project/output/`

**TTS (VITS Checkpoints)**:
- V1: Place `checkpoint_540000.pth` in `uzbek_tts_project/training_output/`
- V3: Place checkpoint in `uzbek_tts_project/training_output_ipa_v3/`

---

## Usage

### Start the API Server

```bash
cd uzbek_voice_api
uvicorn main:app --host 0.0.0.0 --port 8000
```

### API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/tts` | Convert text to speech |
| POST | `/api/stt` | Convert speech to text |
| GET | `/api/health` | Health check |
| GET | `/api/models` | List available models |
| GET | `/dashboard` | Web UI |

### TTS Request Example

```bash
curl -X POST http://localhost:8000/api/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "Salom! Bugun ob-havo juda yaxshi.", "model": "v1", "speed": 1.0}'
```

### STT Request Example

```bash
curl -X POST http://localhost:8000/api/stt \
  -F "file=@audio.wav"
```

### Run the Telegram Bot

```bash
cd uzbek_voice_api
python telegram_bot.py
```

Send a **voice message** to get transcription, or send **text** to get Uzbek speech.

---

## Training

### Fine-tune Whisper STT

```bash
cd uzbek_stt_project

# Step 1: Prepare the dataset
python prepare_dataset.py

# Step 2: Train with LoRA
python finetune_whisper_synthetic.py

# Step 3: Evaluate
python evaluate_models.py
```

**Training config**: batch size 16 (effective), learning rate 1e-4, 5000 steps, fp16 mixed precision.

### Train VITS TTS

```bash
cd uzbek_tts_project

# Step 1: Prepare IPA dataset
python prepare_ipa_dataset.py

# Step 2: Train
python train_ipa_tts.py

# Step 3: Resume training (optional)
python continue_training.py
```

---

## Data Preprocessing

Key preprocessing steps for Uzbek language:

- **Apostrophe normalization**: Uzbek uses o' and g' — multiple Unicode variants (`'`, `'`, `` ` ``, `ʻ`, `ʼ`) are standardized to a single form
- **Cyrillic to Latin conversion**: Mixed-script text is converted to Latin Uzbek
- **Audio filtering**: Clips shorter than 0.5s or longer than 30s are excluded
- **Resampling**: 16 kHz for STT (Whisper), 22,050 Hz for TTS (VITS)

---

## Dialect Handling

Uzbekistan has significant regional dialect variation. This project includes a three-method approach documented in `UZBEK_TEXT_DIALECT_GUIDE.txt`:

1. **Dictionary lookup** — fast, exact matching for known dialect words
2. **Semantic similarity** — LaBSE embeddings for unknown dialect words
3. **Combined approach** — dictionary first, semantic fallback, with auto-learning

---

## Author

**Abulqosim Rafiqov**
- Email: rafiqovbulqosim@gmail.com
- GitHub: [@Abulqosim0227](https://github.com/Abulqosim0227)

---

## Acknowledgments

- [ISSAI](https://issai.nu.edu.kz/) for the Uzbek Speech Corpus dataset
- [OpenAI](https://openai.com/) for the Whisper model
- [Coqui TTS](https://github.com/coqui-ai/TTS) for the VITS implementation
- [HuggingFace](https://huggingface.co/) for Transformers and PEFT libraries

---

## License

This project is for educational and research purposes.
