#!/bin/bash
# Evaluate SpeechCOMET models on ContraProST pairwise accuracy.

# format: model_name:modality
MODELS=(
    audio-sonar:audio
    audio-whisper:audio
    text:text
    audiotext-sonar:audiotext
)

CHECKPOINT_FOLDER=trained_models
DATA_DIR="data/contraProST"

for ENTRY in "${MODELS[@]}"; do
    IFS=: read -r model modality <<< "$ENTRY"

    echo "=== ContraProST eval for $model ==="

    python speechcomet-eval/contraprost_eval.py \
        --data-dir "$DATA_DIR" \
        --modality "$modality" \
        --batch-size 32 \
        --model-folder "$CHECKPOINT_FOLDER/$model"
done
