#!/bin/bash
# Run ContraProST pairwise accuracy evaluation — textaudio modality.

MODEL_NAMES=(
    orkney-avg-20ep
    orkney-sum-20ep
    orkney-concat-20ep
    orkney-sum-from-text-ckpt-20ep
)
MODALITY=textaudio
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

    echo "=== ContraProST ($MODALITY) for $MODEL_NAME ==="

    python evaluation/contraprost_eval.py \
        --data-dir "$DATA_DIR" \
        --modality "$MODALITY" \
        --batch-size 32 \
        $MODEL_ARG
done
