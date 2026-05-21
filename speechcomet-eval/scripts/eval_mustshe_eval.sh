#!/bin/bash
# Evaluate SpeechCOMET models on MuST-SHE pairwise accuracy.

# format: model_name:modality
MODELS=(
    audio-sonar:audio
    audio-whisper:audio
    text:text
    audiotext-sonar:audiotext
)

CHECKPOINT_FOLDER=trained_models
MUSTSHE_DIR="data/MuST-SHE_v1.2/MuST-SHE-v1.2-data/tsv"

for ENTRY in "${MODELS[@]}"; do
    IFS=: read -r model modality <<< "$ENTRY"

    echo "=== MuST-SHE eval for $model ==="

    python evaluation/mustshe_eval.py \
        --mustshe-dir "$MUSTSHE_DIR" \
        --modality "$modality" \
        --batch-size 32 \
        --model-folder "$CHECKPOINT_FOLDER/$model"
done
