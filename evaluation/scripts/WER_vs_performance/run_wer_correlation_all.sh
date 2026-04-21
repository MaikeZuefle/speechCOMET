#!/bin/bash

CHECKPOINT_FOLDER=trained_models
SPLIT=dev_asr

MODELS=(
    harris-20ep
    lewis-10ep
    mull-attn-10ep
    mull-attn-lora-10ep
    mull-avg-20ep
    mull-avg-lora-10ep
    orkney-avg-20ep
    orkney-concat-20ep
    orkney-sum-20ep
    orkney-sum-from-text-ckpt-20ep
    shetland-20ep
    skye-20ep
)

THRESHOLDS=(80 90)

for model_name in "${MODELS[@]}"; do
    model_dir="$CHECKPOINT_FOLDER/$model_name"
    if ls "${model_dir}"/output_scores_${SPLIT}_*.jsonl 2>/dev/null | grep -q .; then
        echo "=== $model_name ==="
        for thresh in "${THRESHOLDS[@]}"; do
            python evaluation/wer_correlation_analysis.py \
                --model-dir "$model_dir" \
                --split $SPLIT \
                --challenge-score-threshold $thresh
        done
    else
        echo "=== $model_name — no ${SPLIT} outputs found, skipping ==="
    fi
done

# --- SpeechLLM baselines (Qwen) ---
QWEN_DIR="speechllm-baselines/Qwen_Qwen2.5-Omni-7B"
for modality in text audio audiotext; do
    if ls "${QWEN_DIR}"/output_scores_${SPLIT}_*_${modality}.jsonl 2>/dev/null | grep -q .; then
        echo "=== Qwen2.5-Omni (${modality}) ==="
        for thresh in "${THRESHOLDS[@]}"; do
            python evaluation/wer_correlation_analysis.py \
                --model-dir "$QWEN_DIR" \
                --split $SPLIT \
                --score-suffix "$modality" \
                --challenge-score-threshold $thresh
        done
    else
        echo "=== Qwen2.5-Omni (${modality}) — no ${SPLIT} outputs found, skipping ==="
    fi
done
