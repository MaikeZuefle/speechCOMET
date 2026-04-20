#!/bin/bash
# Run ContraProST pairwise accuracy evaluation — audio modality.

MODEL_NAMES=(
    harris-20ep
    shetland-20ep
    mull-avg-20ep
    mull-avg-lora-10ep
    mull-attn-10ep
    mull-attn-lora-10ep
)




MODALITY=audio
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
