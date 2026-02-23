#!/usr/bin/env python3
"""
Use your trained Uzbek TTS model
Run after training completes
"""

from TTS.api import TTS
from pathlib import Path
import sys

# Project directory
project_dir = Path(__file__).parent
checkpoint_dir = project_dir / 'training_output' / 'checkpoints'

# Find best model
print("🔍 Looking for trained model...")

# Try to find best_model.pth (saved at end of training)
best_model = checkpoint_dir / 'best_model.pth'

if not best_model.exists():
    # Find latest checkpoint
    checkpoints = sorted(checkpoint_dir.glob('checkpoint_*.pth'),
                        key=lambda p: p.stat().st_mtime)

    if not checkpoints:
        print("❌ No trained model found!")
        print(f"   Expected location: {checkpoint_dir}")
        print("\n   Train the model first: python3 train_production.py")
        sys.exit(1)

    best_model = checkpoints[-1]
    print(f"✓ Found checkpoint: {best_model.name}")
else:
    print(f"✓ Found best model: {best_model.name}")

# Find config
config_file = project_dir / 'training_output' / 'config.json'
if not config_file.exists():
    config_file = project_dir / 'config_production.json'

print(f"✓ Config: {config_file.name}")
print()

# Load model
print("📥 Loading model (this may take a moment)...")
try:
    tts = TTS(model_path=str(best_model),
              config_path=str(config_file),
              gpu=True)
    print("✓ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    sys.exit(1)

print()
print("="*60)
print("UZBEK TTS - READY!")
print("="*60)
print()

# Interactive mode
while True:
    print("Enter Uzbek text (or 'quit' to exit):")
    text = input("> ").strip()

    if text.lower() in ['quit', 'exit', 'q']:
        print("Goodbye!")
        break

    if not text:
        continue

    # Generate filename
    import time
    output_file = f"output_{int(time.time())}.wav"

    print(f"\n🎤 Generating speech...")

    try:
        tts.tts_to_file(text=text, file_path=output_file)
        print(f"✓ Audio saved: {output_file}")
        print(f"   Play with: aplay {output_file}")
        print()

    except Exception as e:
        print(f"❌ Error: {e}")
        print()
