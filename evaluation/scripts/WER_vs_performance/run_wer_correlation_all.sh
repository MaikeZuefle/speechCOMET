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

WER_CSV="data/wer_analysis/wer_dev_asr.csv"
THRESHOLDS=(80) # 90

# for model_name in "${MODELS[@]}"; do
#     model_dir="$CHECKPOINT_FOLDER/$model_name"
#     if ls "${model_dir}"/output_scores_${SPLIT}_*.jsonl 2>/dev/null | grep -q .; then
#         echo "=== $model_name ==="
#         for thresh in "${THRESHOLDS[@]}"; do
#             python evaluation/wer_correlation_analysis.py \
#                 --model-dir "$model_dir" \
#                 --split $SPLIT \
#                 --wer-csv "$WER_CSV" \
#                 --challenge-score-threshold $thresh
#         done
#     else
#         echo "=== $model_name — no ${SPLIT} outputs found, skipping ==="
#     fi
# done

# --- QE baselines ---
for qe_method in qe-comet qe-comet-partial qe-blaser qe-speechqe; do
    qe_dir="QE-baselines/results/$qe_method"
    if ls "${qe_dir}"/output_scores_${SPLIT}_*.jsonl 2>/dev/null | grep -q .; then
        echo "=== $qe_method ==="
        for thresh in "${THRESHOLDS[@]}"; do
            python evaluation/wer_correlation_analysis.py \
                --model-dir "$qe_dir" \
                --split $SPLIT \
                --wer-csv "$WER_CSV" \
                --challenge-score-threshold $thresh
        done
    else
        echo "=== $qe_method — no ${SPLIT} outputs found, skipping ==="
    fi
done

# # --- SpeechLLM models (base + FT, format: output_name:modality) ---
# SPEECHLLM_RESULTS="speechllm-baselines/results"
# SPEECHLLM_MODELS=(
#     "Qwen_Qwen2.5-Omni-7B:text"
#     "Qwen_Qwen2.5-Omni-7B:audio"
#     "Qwen_Qwen2.5-Omni-7B:audiotext"
#     "Qwen_Qwen2.5-Omni-7B-iwslt26-text:text"
#     "Qwen_Qwen2.5-Omni-7B-iwslt26-audio:audio"
#     "Qwen_Qwen2.5-Omni-7B-iwslt26-textaudio:audiotext"
# )
# for entry in "${SPEECHLLM_MODELS[@]}"; do
#     model_name="${entry%%:*}"
#     modality="${entry##*:}"
#     model_dir="$SPEECHLLM_RESULTS/$model_name"
#     if ls "${model_dir}"/output_scores_${SPLIT}_*_${modality}.jsonl 2>/dev/null | grep -q .; then
#         echo "=== $model_name (${modality}) ==="
#         for thresh in "${THRESHOLDS[@]}"; do
#             python evaluation/wer_correlation_analysis.py \
#                 --model-dir "$model_dir" \
#                 --split $SPLIT \
#                 --wer-csv "$WER_CSV" \
#                 --score-suffix "$modality" \
#                 --challenge-score-threshold $thresh
#         done
#     else
#         echo "=== $model_name (${modality}) — no ${SPLIT} outputs found, skipping ==="
#     fi
# done
