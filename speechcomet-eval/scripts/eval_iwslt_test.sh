#!/bin/bash
# SpeechCOMET evaluation on the IWSLT 2026 shared task test set.

MODEL=orkney-sum-from-text-ckpt-BIG
MODALITY=audiotext
SPLIT=test
DATASET=maikezu/iwslt2026-metrics-shared-test
CHECKPOINT_FOLDER=trained_models
AUDIO_BASE_DIR=iwslt2026data

OUTPUT_DIR="$CHECKPOINT_FOLDER/$MODEL"

echo "=== $MODEL ==="

python iwslt2026/eval_speechcomet.py \
    --model-folder "$OUTPUT_DIR" \
    --modality "$MODALITY" \
    --split "$SPLIT" \
    --dataset "$DATASET" \
    --audio-base-dir "$AUDIO_BASE_DIR"
