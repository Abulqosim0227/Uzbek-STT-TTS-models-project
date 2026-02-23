#!/bin/bash
# RESUME V3 FERRARI - From checkpoint 145k to 450k
# Target: Smooth like butter, sharp natural

cd /mnt/c/Users/Admin/Desktop/voice_dataset/ISSAI_USC/uzbek_tts_project

TRAINING_DIR="training_output_ipa_v3/uzbek_tts_ipa_v3-December-12-2025_09+25AM-0000000"

echo "=============================================="
echo "V3 FERRARI - RESUME TRAINING"
echo "=============================================="
echo "Resuming from: checkpoint_145000.pth"
echo "Target: 3000 epochs (~450k steps)"
echo "Goal: Smooth like butter, sharp natural"
echo ""
echo "Special letters to perfect:"
echo "  X (χ) - Xiva, Xorazm"
echo "  G' (ʁ) - G'ozal, bog'"
echo "  O' (ø) - O'zbek, ko'z"
echo "  SH (ʃ) - Shahar, oshxona"
echo "  CH (tʃ) - Chiroyli, chaman"
echo "=============================================="
echo ""

CUDA_VISIBLE_DEVICES=0 python3 train_tts_runner.py \
    --continue_path "$TRAINING_DIR" \
    2>&1 | tee -a training_v3_resume.log

echo ""
echo "Training finished or interrupted."
