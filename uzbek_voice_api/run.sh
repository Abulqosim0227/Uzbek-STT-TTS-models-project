#!/bin/bash
# Run Uzbek Voice API

cd /mnt/c/Users/Admin/Desktop/voice_dataset/ISSAI_USC/uzbek_voice_api

echo "=============================================="
echo "   O'ZBEK OVOZ API"
echo "   Text-to-Speech + Speech-to-Text"
echo "=============================================="
echo ""
echo "Starting server on http://0.0.0.0:8000"
echo ""
echo "Endpoints:"
echo "  Dashboard:  http://localhost:8000/dashboard"
echo "  API Docs:   http://localhost:8000/docs"
echo "  ReDoc:      http://localhost:8000/redoc"
echo "  Health:     http://localhost:8000/api/health"
echo ""
echo "=============================================="
echo ""

# Run with uvicorn
CUDA_VISIBLE_DEVICES=0 python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
