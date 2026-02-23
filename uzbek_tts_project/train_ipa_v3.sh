#!/bin/bash
# V3 Ferrari Training - Single Speaker IPA

cd /mnt/c/Users/Admin/Desktop/voice_dataset/ISSAI_USC/uzbek_tts_project

echo "=============================================="
echo "V3 FERRARI - Single Speaker IPA Training"
echo "=============================================="
echo "Train samples: 4652"
echo "Val samples: 245"
echo "Speaker: 1459541555"
echo ""

mkdir -p training_output_ipa_v3

CUDA_VISIBLE_DEVICES=0 nohup python3 train_tts_runner.py \
    --config_path config_ipa_v3.json \
    > training_v3_live.log 2>&1 &

echo "Training started with PID: $!"
echo ""
echo "Monitor with: tail -f training_v3_live.log"
