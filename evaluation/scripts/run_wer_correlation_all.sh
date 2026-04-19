#!/bin/bash

CHECKPOINT_FOLDER=trained_models
SPLIT=dev_asr

# declare -A MODELS=(
#     ["harris"]="harris-20ep"
#     ["lewis"]="lewis-10ep"
#     ["mull-attn"]="mull-attn-10ep"
#     ["mull-attn-lora"]="mull-attn-lora-10ep"
#     ["mull-avg"]="mull-avg-20ep"
#     ["mull-avg-lora"]="mull-avg-lora-10ep"
#     ["orkney-avg"]="orkney-avg-20ep"
#     ["orkney-concat"]="orkney-concat-20ep"
#     ["orkney-sum"]="orkney-sum-20ep"
#     ["orkney-sum-from-text-ckpt"]="orkney-sum-from-text-ckpt-20ep"
#     ["shetland"]="shetland-20ep"
#     ["skye"]="skye-20ep"
# )

# for hf_name in "${!MODELS[@]}"; do
#     local_name="${MODELS[$hf_name]}"
#     model_dir="$CHECKPOINT_FOLDER/$local_name"

#     if ls "${model_dir}"/output_scores_${SPLIT}_*.jsonl 2>/dev/null | grep -q .; then
#         echo "=== $local_name ==="
#         python evaluation/wer_correlation_analysis.py \
#             --model-dir "$model_dir" \
#             --split $SPLIT
#     else
#         echo "=== $local_name — no ${SPLIT} outputs found, skipping ==="
#     fi
# done

# --- SpeechLLM baselines (Qwen) ---
QWEN_DIR="speechllm-baselines/Qwen_Qwen2.5-Omni-7B"
for modality in text audio audiotext; do
    if ls "${QWEN_DIR}"/output_scores_${SPLIT}_*_${modality}.jsonl 2>/dev/null | grep -q .; then
        echo "=== Qwen2.5-Omni (${modality}) ==="
        python evaluation/wer_correlation_analysis.py \
            --model-dir "$QWEN_DIR" \
            --split $SPLIT \
            --score-suffix "$modality"
    else
        echo "=== Qwen2.5-Omni (${modality}) — no ${SPLIT} outputs found, skipping ==="
    fi
done
