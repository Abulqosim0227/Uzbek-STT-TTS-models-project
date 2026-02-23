# Audio Samples

This folder contains audio samples demonstrating the Uzbek Voice AI system.

## TTS V1 Samples (Plain Text Model)

VITS model trained for 540,000 steps on plain Uzbek text input.

| File | Description |
|---|---|
| `tts_v1/tts_v1_70db2ef3.wav` | Generated speech from API |
| `tts_v1/tts_v1_7adfa781.wav` | Generated speech from API |
| `tts_v1/challenge_v1_01.wav` | Challenge test sentence |
| `tts_v1/challenge_v1_02.wav` | Challenge test sentence |
| `tts_v1/challenge_v1_03.wav` | Challenge test sentence |

## TTS V3 Samples (IPA Model — Better Pronunciation)

VITS model trained with IPA (International Phonetic Alphabet) input for improved pronunciation of Uzbek-specific sounds (o', g', x, q).

| File | Description |
|---|---|
| `tts_v3/tts_v3_c6f1d666.wav` | Generated speech from API (IPA) |
| `tts_v3/tts_v3_eddaf11b.wav` | Generated speech from API (IPA) |
| `tts_v3/challenge_v3_01.wav` | Challenge test sentence (IPA) |
| `tts_v3/challenge_v3_02.wav` | Challenge test sentence (IPA) |
| `tts_v3/challenge_v3_03.wav` | Challenge test sentence (IPA) |

## Original Dataset Samples (ISSAI USC)

Real human speech recordings from the ISSAI Uzbek Speech Corpus used for STT training. Each `.wav` file has a matching `.txt` transcription.

| Audio | Transcription |
|---|---|
| `1126542829_1_33727_1.wav` | fojia oqibatida halok bo'lganlarga to'lanishi kerak bo'lgan to'lov pullarini hisoblash jarayonlari davom etmoqda |
| `1131474547_2_29207_1.wav` | shu o'rinda odam emassiz hayvonsiz desam qo'polligim uchun ranjiysiz |
| `1131474547_2_29208_1.wav` | ammo boshqa bir sabab bilan hayvon deya olmayman |
| `1131474547_2_29209_1.wav` | buni aytsam o'sha begunoh jonivorlarni haqoratlagan balchiqqa bulg'agan bo'laman |
| `1131474547_2_29210_1.wav` | chekinishni bas qilib yana depara ichki ishlar bo'limiga qaytaylik |

These samples represent a small subset of the 108,387 audio recordings in the full ISSAI USC dataset (100+ hours of Uzbek speech from 932 speakers).
