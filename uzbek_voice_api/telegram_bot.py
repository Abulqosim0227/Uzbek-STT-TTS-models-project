#!/usr/bin/env python3
"""
UZBEK VOICE TELEGRAM BOT
TTS + STT via Telegram

Features:
- Send text → Get voice message (TTS)
- Send voice → Get text transcription (STT)
- /v1 - Switch to V1 model (plain text, 540k steps)
- /v3 - Switch to V3 model (IPA, training)

Usage:
    python telegram_bot.py
"""

import os
import asyncio
import logging
from pathlib import Path
from telegram import Update, Bot
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# === CONFIG ===
BOT_TOKEN = "8347822379:AAGwgicXcBu7FUzsaB3mnmUk1JT43Ol7GpA"
AUDIO_DIR = Path("/mnt/c/Users/Admin/Desktop/voice_dataset/ISSAI_USC/uzbek_voice_api/telegram_audio")
AUDIO_DIR.mkdir(exist_ok=True)

# === ENGINES (lazy load) ===
tts_engine = None
stt_engine = None
current_tts_version = "v1"  # Default to V1

def get_tts(version=None):
    global tts_engine, current_tts_version
    if version:
        current_tts_version = version
    if tts_engine is None or (version and tts_engine.get_version() != version):
        logger.info(f"Loading TTS engine ({current_tts_version})...")
        from tts_engine import TTSEngine
        tts_engine = TTSEngine(model_version=current_tts_version)
        logger.info(f"TTS engine loaded: {current_tts_version.upper()}")
    return tts_engine

def get_stt():
    global stt_engine
    if stt_engine is None:
        logger.info("Loading STT engine...")
        from stt_engine import STTEngine
        stt_engine = STTEngine()
        logger.info("STT engine loaded!")
    return stt_engine

# === COMMANDS ===

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send welcome message"""
    welcome = """🎙 **O'zbek Ovoz Bot**

Salom! Men O'zbek tilida ovoz bilan ishlayman.

**Imkoniyatlar:**
📝➡️🔊 Matn yuboring → Ovozli xabar olasiz
🎤➡️📝 Ovozli xabar yuboring → Matn olasiz

**Buyruqlar:**
/start - Boshlash
/help - Yordam
/about - Bot haqida
/v1 - V1 model (540k, tez)
/v3 - V3 model (IPA, aniq talaffuz)
/status - Joriy model

Matn yozing yoki ovozli xabar yuboring!"""

    await update.message.reply_text(welcome, parse_mode='Markdown')

async def switch_v1(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Switch to V1 model"""
    await update.message.chat.send_action("typing")
    get_tts("v1")
    await update.message.reply_text(
        "✅ **V1 model tanlandi**\n\n"
        "• 540,000 qadamda o'qitilgan\n"
        "• Tez ishlaydi\n"
        "• Ba'zi so'zlar noto'g'ri talaffuz qilinishi mumkin",
        parse_mode='Markdown'
    )

async def switch_v3(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Switch to V3 model"""
    await update.message.chat.send_action("typing")
    get_tts("v3")
    await update.message.reply_text(
        "✅ **V3 model tanlandi (IPA)**\n\n"
        "• IPA fonetik alifbosi bilan o'qitilgan\n"
        "• Aniq talaffuz: G', X, O', Q\n"
        "• Hali o'qitilmoqda - sifat yaxshilanadi",
        parse_mode='Markdown'
    )

async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show current model status"""
    global current_tts_version
    version = current_tts_version.upper()
    await update.message.reply_text(
        f"📊 **Joriy holat:**\n\n"
        f"TTS model: **{version}**\n"
        f"STT model: Whisper + LoRA (IT Prompt Injection)",
        parse_mode='Markdown'
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send help message"""
    help_text = """📖 **Yordam**

**TTS (Matn → Ovoz):**
Oddiy matn yuboring, bot uni o'qib beradi.

Misol: `Salom! Bugun ob-havo juda yaxshi.`

**STT (Ovoz → Matn):**
Ovozli xabar yuboring, bot uni matnga aylantiradi.

**Maslahatlar:**
• Tinish belgilarini ishlating (. , ! ?)
• Aniq gapiring
• Shovqinsiz joyda yozing"""

    await update.message.reply_text(help_text, parse_mode='Markdown')

async def about(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send about message"""
    about_text = """🤖 **O'zbek Ovoz Bot**

**Texnologiyalar:**
• TTS: VITS model (540k steps)
• STT: Whisper + LoRA (26.7% WER)
• IT Vocabulary: Prompt Injection

**Muallif:** Abulqosim Rafiqov
**Yil:** 2025

Bu bot O'zbek tilida sun'iy ovoz sintezi va nutqni aniqlash uchun yaratilgan."""

    await update.message.reply_text(about_text, parse_mode='Markdown')

# === TTS: Text → Voice ===

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Convert text to speech"""
    text = update.message.text.strip()

    if not text:
        return

    # Skip if too long
    if len(text) > 500:
        await update.message.reply_text("⚠️ Matn juda uzun! 500 belgidan kam bo'lsin.")
        return

    # Send "recording" action
    await update.message.chat.send_action("record_voice")

    try:
        # Generate speech
        engine = get_tts()
        output_path = AUDIO_DIR / f"tts_{update.message.message_id}.wav"
        duration = engine.synthesize(text, str(output_path))

        # Send voice message
        with open(output_path, 'rb') as audio:
            await update.message.reply_voice(
                voice=audio,
                caption=f"🔊 {text[:100]}{'...' if len(text) > 100 else ''}",
                duration=int(duration)
            )

        # Cleanup
        output_path.unlink(missing_ok=True)

    except Exception as e:
        logger.error(f"TTS error: {e}")
        await update.message.reply_text(f"❌ Xatolik: {str(e)[:100]}")

# === STT: Voice → Text ===

async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Convert voice to text"""
    voice = update.message.voice or update.message.audio

    if not voice:
        return

    # Send "typing" action
    await update.message.chat.send_action("typing")

    try:
        # Download voice file
        file = await context.bot.get_file(voice.file_id)
        input_path = AUDIO_DIR / f"stt_{update.message.message_id}.ogg"
        await file.download_to_drive(str(input_path))

        # Transcribe
        engine = get_stt()
        text, confidence = engine.transcribe(str(input_path))

        # Send result
        if text:
            await update.message.reply_text(
                f"📝 **Aniqlangan matn:**\n\n{text}\n\n"
                f"✅ Ishonchlilik: {int(confidence * 100)}%",
                parse_mode='Markdown'
            )
        else:
            await update.message.reply_text("⚠️ Ovoz aniqlanmadi. Qaytadan urinib ko'ring.")

        # Cleanup
        input_path.unlink(missing_ok=True)

    except Exception as e:
        logger.error(f"STT error: {e}")
        await update.message.reply_text(f"❌ Xatolik: {str(e)[:100]}")

# === MAIN ===

def main():
    """Start the bot"""
    print("=" * 50)
    print("🎙 O'ZBEK OVOZ TELEGRAM BOT")
    print("=" * 50)
    print(f"Bot: @butelegrambotteststttss_bot")
    print("Starting...")
    print("=" * 50)

    # Create application
    app = Application.builder().token(BOT_TOKEN).build()

    # Add handlers
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("about", about))
    app.add_handler(CommandHandler("v1", switch_v1))
    app.add_handler(CommandHandler("v3", switch_v3))
    app.add_handler(CommandHandler("status", status))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, handle_voice))

    # Start polling - drop pending updates to take over from any other instance
    print("\n✅ Bot is running! Press Ctrl+C to stop.\n")
    app.run_polling(
        allowed_updates=Update.ALL_TYPES,
        drop_pending_updates=True
    )

if __name__ == "__main__":
    main()
