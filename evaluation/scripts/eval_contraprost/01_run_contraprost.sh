#!/bin/bash
# Run ContraProST pairwise accuracy evaluation for a set of models.

MODEL_NAMES=(
    harris-20ep
    # shetland-20ep
)
MODALITY=audio  # audio, text, textaudio, audiotext
HF=false

CHECKPOINT_FOLDER=trained_models
HF_USER=maikezu
DATA_DIR="data/contraProST"

# --------------
for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    if [ "$HF" = false ]; then
        MODEL_ARG="--model-folder $CHECKPOINT_FOLDER/$MODEL_NAME"
    else
        MODEL_ARG="--hf-model $HF_USER/$MODEL_NAME"
    fi

    echo "=== ContraProST eval for $MODEL_NAME ==="

    python evaluation/contraprost_eval.py \
        --data-dir "$DATA_DIR" \
        --modality "$MODALITY" \
        --batch-size 32 \
        $MODEL_ARG
done
