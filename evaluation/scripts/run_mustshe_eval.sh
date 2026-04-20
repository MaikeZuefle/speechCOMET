#!/bin/bash
# Evaluate SpeechCOMET models on MuST-SHE pairwise accuracy.

MODEL_NAMES=(
    # shetland
    # harris-20ep
    shetland-20ep
    # orkney-avg-20ep
    # orkney-sum-20ep
    # orkney-concat-20ep
    # orkney-sum-from-text-ckpt-20ep
    # mull-avg-20ep
    # mull-avg-lora-10ep
    # mull-attn-10-ep
    # mull-attn-lora-10ep

)
MODALITY=audio  # audio, text, textaudio
HF=false        # true if all models are from HF

# define paths
CHECKPOINT_FOLDER=trained_models  # ignored if HF=true
HF_USER=maikezu                   # ignored if HF=false
MUSTSHE_DIR="data/MuST-SHE_v1.2/MuST-SHE-v1.2-data/tsv"

# --------------
for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    if [ "$HF" = false ]; then
        MODEL_ARG="--model-folder $CHECKPOINT_FOLDER/$MODEL_NAME"
    else
        MODEL_ARG="--hf-model $HF_USER/$MODEL_NAME"
    fi

    echo "=== MuST-SHE eval for $MODEL_NAME ==="

    python evaluation/mustshe_eval.py \
        --mustshe-dir "$MUSTSHE_DIR" \
        --modality "$MODALITY" \
        --batch-size 32 \
        $MODEL_ARG
done
