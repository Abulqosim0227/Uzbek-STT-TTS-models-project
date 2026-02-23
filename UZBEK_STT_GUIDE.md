# O'zbek Speech-to-Text (STT) Tizimi — Texnik Qo'llanma
### Uzbek Speech-to-Text System — Technical Guide

**Muallif:** Abulqosim Rafiqov
**Sana:** Fevral 2026
**Loyiha:** Nihol AI — O'zbek Ovozli Yordamchi

---

## 1. Tizim Haqida Qisqacha / System Overview

Bu tizim **o'zbek tilidagi nutqni matnga aylantiradi** (Speech-to-Text). Foydalanuvchi ovozli xabar yuborsa, tizim uni avtomatik ravishda matn ko'rinishiga o'tkazadi.

This system converts **Uzbek speech into text**. When a user sends a voice message, the system automatically transcribes it.

**Asosiy ko'rsatkichlar / Key Metrics:**

| Ko'rsatkich | Qiymat |
|-------------|--------|
| Model | OpenAI Whisper Large V3 + LoRA |
| WER (Word Error Rate) | 26.7% (toza audio) |
| So'zlarni to'g'ri aniqlash | ~73% |
| IT terminlar | Prompt Injection orqali qo'llab-quvvatlanadi |
| Audio format | 16kHz, WAV/OGG |

---

## 2. Qanday Ishlaydi / How It Works

```
Foydalanuvchi ovozli xabar yuboradi
         |
         v
Audio 16kHz ga resample qilinadi
         |
         v
Whisper Large V3 modeli audio'ni qayta ishlaydi
         |
         v
LoRA adapterlari o'zbek tilini aniqroq tushunadi
         |
         v
IT Prompt Injection texnik so'zlarni to'g'ri yozadi
         |
         v
Matn natija qaytariladi
```

---

## 3. Model Qanday O'rgatilgan / How the Model Was Trained

### 3.1 Boshlang'ich Model (Base Model)

Biz **OpenAI Whisper Large V3** modelidan foydalandik. Bu model 100+ tilda nutqni taniy oladi, jumladan o'zbek tilida ham ba'zi darajada ishlaydi.

- **URL:** https://huggingface.co/openai/whisper-large-v3
- **Parametrlar soni:** 1.5 milliard
- **O'lcham:** ~3 GB

### 3.2 Ma'lumotlar To'plami (Dataset)

**ISSAI Uzbek Speech Corpus (USC)** — Qozog'iston ISSAI instituti tomonidan yaratilgan.

- **URL:** https://issai.nu.edu.kz/uzbek-speech-corpus/
- **HuggingFace:** https://huggingface.co/datasets/ISSAI/uzbek_speech_corpus

| Bo'lim | Audio + Matn juftliklari | Hajmi |
|--------|--------------------------|-------|
| Train | 100,767 | 14 GB |
| Dev | 3,783 | 543 MB |
| Test | 3,837 | 590 MB |
| **Jami** | **108,387** | **~15 GB** |

- 932 ta turli spikerlar (so'zlovchilar)
- Har bir audio 0.5 — 30 soniya
- Toza studio yozuvlari
- Lotin alifbosida transkripsiya

### 3.3 Fine-Tuning Usuli: LoRA

Biz butun modelni emas, faqat **kichik adapterlarni** o'rgatdik. Bu usul **LoRA (Low-Rank Adaptation)** deb ataladi.

- **URL:** https://huggingface.co/docs/peft/conceptual_guides/adapter
- **Afzalliklari:**
  - To'liq model: 3 GB o'rgatish kerak → LoRA: faqat **~50-90 MB**
  - Kamroq GPU xotira talab qiladi
  - Tezroq o'rgatiladi
  - Bazaviy modelning bilimlarini saqlaydi

**LoRA konfiguratsiyasi:**

```python
from peft import LoraConfig

lora_config = LoraConfig(
    r=32,                    # Rang (rank) — qanchalik ko'p = shunchalik kuchli
    lora_alpha=64,           # Masshtablash koeffitsiyenti
    target_modules=[         # Qaysi qatlamlarni moslashtirish
        "q_proj",            # Query proektsiyasi
        "v_proj",            # Value proektsiyasi
        "k_proj",            # Key proektsiyasi
        "o_proj"             # Output proektsiyasi
    ],
    lora_dropout=0.05,
    bias="none"
)
```

### 3.4 O'rgatish Jarayoni (Training Process)

```python
training_args = Seq2SeqTrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,   # Samarali batch = 16
    learning_rate=1e-4,
    max_steps=5000,
    fp16=True,                        # Aralash aniqlik (tezroq)
    eval_steps=500,
    save_steps=1000
)
```

**O'rgatish natijalari:**

| Step | WER (Word Error Rate) |
|------|----------------------|
| 500 | 73.0% |
| 1000 | 59.0% |
| 1500 | 49.8% |
| **2000** | **36.7% (eng yaxshi checkpoint)** |
| 5000 | 41.9% |

### 3.5 IT Terminlarni Aniqlash (Prompt Injection)

Qayta o'rgatmasdan IT so'zlarni yaxshiroq aniqlash uchun **Prompt Injection** texnikasidan foydalandik:

```python
IT_PROMPT = "Server, Kubernetes, Oracle, API, Cloud, Linux, Python, \
             DevOps, Docker, Deploy, Backend, Frontend, Database, \
             Cyber Security, GitHub, React, Node.js"

prompt_ids = processor.get_prompt_ids(IT_PROMPT, return_tensors="pt")

outputs = model.generate(
    input_features,
    prompt_ids=prompt_ids,    # IT so'zlarni "ko'rsatib" beradi
    language="uz",
    task="transcribe"
)
```

Bu usul modelga IT terminlarni "eslatadi" — qayta o'rgatishsiz!

---

## 4. O'rgatishni Boshlash Uchun Qo'llanma / Step-by-Step Training Guide

### 4.1 Kerakli kutubxonalar

```bash
pip install torch transformers peft datasets evaluate librosa soundfile accelerate
```

### 4.2 Ma'lumotlarni tayyorlash

```python
# 1. Audio + matn juftliklarini yuklash
# 2. Matnni normalizatsiya qilish:
#    - Apostroflarni standartlashtirish (o', g')
#    - Kirill → Lotin konvertatsiya
#    - Kichik harflarga o'tkazish
# 3. Audiolarni filtrlash (0.5s — 30s)
# 4. 16kHz ga resample qilish
# 5. HuggingFace Dataset formatida saqlash
```

### 4.3 O'rgatishni ishga tushirish

```python
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from peft import get_peft_model, LoraConfig

# 1. Bazaviy modelni yuklash
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3")
processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")

# 2. LoRA adapterlarini qo'shish
lora_config = LoraConfig(r=32, lora_alpha=64, target_modules=["q_proj","v_proj","k_proj","o_proj"])
model = get_peft_model(model, lora_config)

# 3. O'rgatishni boshlash (Seq2SeqTrainer bilan)
trainer = Seq2SeqTrainer(model=model, args=training_args, ...)
trainer.train()
```

### 4.4 Modelni ishlatish (Inference)

```python
import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel

# Modelni yuklash
base_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3")
model = PeftModel.from_pretrained(base_model, "output/final_model")
processor = WhisperProcessor.from_pretrained("output/final_model")

# Audio faylni yuklash
audio, sr = librosa.load("audio.wav", sr=16000)

# Transkripsiya qilish
input_features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features
outputs = model.generate(input_features, language="uz", task="transcribe")
text = processor.batch_decode(outputs, skip_special_tokens=True)[0]

print(text)  # "bugun ob-havo juda yaxshi"
```

---

## 5. Mintaqaviy Lahjalar Muammosi va Yechimi / Regional Dialects Solution

### 5.1 Muammo

O'zbekistonda bir xil narsa turli mintaqalarda turlicha aytiladi:

| Ma'no | Toshkent | Farg'ona | Xorazm | Samarqand |
|-------|----------|----------|--------|-----------|
| Chelak | chelak | satil | paqir | bedra |
| Bola | bola | bala | bola | bala |
| Keling | keling | keling | keli | kelang |
| Yaxshi | yaxshi | yaxshi | yaxshi | yahshi |

### 5.2 Yechim: 3 Bosqichli Strategiya

#### Bosqich 1: Mintaqaviy ma'lumotlar to'plash (Data Collection)

Eng muhim qadam — **har bir mintaqadan ovozli ma'lumot to'plash**:

```
Toshkent shahri    → 10,000+ audio
Farg'ona vodiysi   → 10,000+ audio
Xorazm viloyati    → 10,000+ audio
Samarqand/Buxoro   → 10,000+ audio
Qashqadaryo/Surxon → 10,000+ audio
Qoraqalpog'iston   → 10,000+ audio
```

**Qanday to'plash:**
- Davlat idoralari, universitet talabalari, maktab o'quvchilari orqali
- Telegram bot orqali (foydalanuvchilar ovozli xabar yuboradi)
- Mahalliy radio/TV arxivlaridan

**Foydali resurslar:**
- Mozilla Common Voice (crowdsourcing platformasi): https://commonvoice.mozilla.org/
- HuggingFace Datasets Hub: https://huggingface.co/datasets

#### Bosqich 2: Multi-dialect model o'rgatish

Ikki usul mavjud:

**Usul A: Bitta universal model (Tavsiya etiladi)**
```
Barcha mintaqalardan ma'lumotlarni aralashtirish
         |
         v
Whisper + LoRA ni barcha lahjalar ustida o'rgatish
         |
         v
Model barcha lahja variantlarini tushuna oladi
"chelak" ham, "satil" ham, "paqir" ham → to'g'ri transkripsiya
```

Bu usul eng yaxshi ishlaydi chunki:
- Whisper allaqachon multi-lingual — turli talaffuzlarni tushunadi
- LoRA adapteri barcha variantlarni o'rganadi
- Foydalanuvchi qaysi mintaqadan bo'lishidan qat'i nazar ishlaydi

**Usul B: Mintaqaviy adapterlar**
```
Bazaviy o'zbek modeli (umumiy)
    |
    +-- LoRA adapter: Toshkent lahjasi
    +-- LoRA adapter: Farg'ona lahjasi
    +-- LoRA adapter: Xorazm lahjasi
```

Har bir mintaqa uchun alohida LoRA adapter o'rgatiladi (~50 MB). Foydalanuvchi o'z mintaqasini tanlaydi.

#### Bosqich 3: Normalizatsiya qatlami (Post-processing)

STT dan keyin **lahja so'zlarini adabiy tilga aylantirish**:

```python
DIALECT_MAP = {
    # Mintaqaviy so'z → Adabiy/Standart so'z
    "satil": "chelak",
    "paqir": "chelak",
    "bedra": "chelak",
    "bala": "bola",
    "yahshi": "yaxshi",
    "kelang": "keling",
    "borasan": "borasiz",
    # ... yuzlab so'zlar
}

def normalize_dialect(text):
    words = text.split()
    normalized = []
    for word in words:
        normalized.append(DIALECT_MAP.get(word, word))
    return " ".join(normalized)

# Misol:
# STT natija: "menga bir satil suv bering"
# Normalizatsiya: "menga bir chelak suv bering"
```

Bu chatbot uchun juda foydali — foydalanuvchi qanday gaplashishidan qat'i nazar, chatbot tushunadi.

---

## 6. Foydali Havolalar / Useful URLs

### Modellar va Dataset
| Resurs | URL |
|--------|-----|
| OpenAI Whisper | https://github.com/openai/whisper |
| Whisper Large V3 (HuggingFace) | https://huggingface.co/openai/whisper-large-v3 |
| ISSAI Uzbek Speech Corpus | https://issai.nu.edu.kz/uzbek-speech-corpus/ |
| ISSAI USC (HuggingFace) | https://huggingface.co/datasets/ISSAI/uzbek_speech_corpus |
| Coqui TTS (VITS model) | https://github.com/coqui-ai/TTS |

### O'rgatish texnologiyalari
| Texnologiya | URL |
|-------------|-----|
| HuggingFace Transformers | https://huggingface.co/docs/transformers |
| PEFT (LoRA) | https://huggingface.co/docs/peft |
| HuggingFace Datasets | https://huggingface.co/docs/datasets |
| Fine-tune Whisper (rasmiy qo'llanma) | https://huggingface.co/blog/fine-tune-whisper |

### Lahja va til resurslari
| Resurs | URL |
|--------|-----|
| Mozilla Common Voice (ma'lumot to'plash) | https://commonvoice.mozilla.org/ |
| Whisper fine-tuning tutorial | https://huggingface.co/blog/fine-tune-whisper |
| LoRA paper (arxiv) | https://arxiv.org/abs/2106.09685 |

### Infratuzilma
| Texnologiya | URL |
|-------------|-----|
| FastAPI | https://fastapi.tiangolo.com/ |
| Faster-Whisper (tez inference) | https://github.com/SYSTRAN/faster-whisper |
| Python Telegram Bot | https://python-telegram-bot.readthedocs.io/ |

---

## 7. Texnik Talablar / System Requirements

### O'rgatish uchun (Training)
- **GPU:** NVIDIA GPU, 16+ GB VRAM (masalan: RTX 3090, A100)
- **RAM:** 32 GB+
- **Disk:** 50 GB+ bo'sh joy
- **OS:** Linux (Ubuntu 20.04+) yoki WSL2
- **Python:** 3.9+
- **CUDA:** 11.8+

### Ishlatish uchun (Inference)
- **GPU:** NVIDIA GPU, 8+ GB VRAM (yoki CPU — sekinroq)
- **RAM:** 16 GB+
- **Python:** 3.9+

---

## 8. Loyihaning Kelajagi / Roadmap

| Bosqich | Vazifa | Vaqt |
|---------|--------|------|
| 1 | Mintaqaviy ovozli ma'lumot to'plash | 2-3 oy |
| 2 | Multi-dialect LoRA o'rgatish | 2-4 hafta |
| 3 | Lahja normalizatsiya lug'ati | 1-2 hafta |
| 4 | API va Telegram bot integratsiya | 1 hafta |
| 5 | Chatbot bilan to'liq integratsiya | 1-2 hafta |

---

## 9. Xulosa / Conclusion

Bu tizim **OpenAI Whisper** ning kuchini **LoRA fine-tuning** orqali o'zbek tiliga moslashtirilgan. Mintaqaviy lahjalar muammosi **ko'proq ma'lumot to'plash + multi-dialect o'rgatish + normalizatsiya** orqali hal qilinadi.

Eng muhimi — bu tizim **kengaytiriladigan (scalable)**. Yangi lahja yoki so'z qo'shish uchun:
- Yangi audio ma'lumotlar to'plash
- LoRA adapterni qayta o'rgatish (1-2 hafta)
- Lahja lug'atini yangilash (1 kun)

Butun modelni qaytadan o'rgatish shart emas!

---

*Tayyorlagan: Abulqosim Rafiqov | Nihol AI*
