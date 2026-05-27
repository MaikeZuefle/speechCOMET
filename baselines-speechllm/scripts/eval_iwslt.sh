#!/bin/bash
cd "$(dirname "$0")/../src" || exit 1

DATASET="maikezu/iwslt2026-metrics-shared-train-dev"
RESULTS_BASE="baselines-speechllm/results"

BASE="Qwen/Qwen2.5-Omni-7B"
FT_TEXT="../../saves/qwen2.5-omni-7b/merged/iwslt26_text"
FT_AUDIO="../../saves/qwen2.5-omni-7b/merged/iwslt26_audio"
FT_TEXTAUDIO="../../saves/qwen2.5-omni-7b/merged/iwslt26_textaudio"

# format: model_path:output_name:modality
MODELS=(
    "$BASE:Qwen_Qwen2.5-Omni-7B:all"
    "$FT_TEXT:Qwen_Qwen2.5-Omni-7B-iwslt26-text:text"
    "$FT_AUDIO:Qwen_Qwen2.5-Omni-7B-iwslt26-audio:audio"
    "$FT_TEXTAUDIO:Qwen_Qwen2.5-Omni-7B-iwslt26-textaudio:audiotext"
)

for ENTRY in "${MODELS[@]}"; do
    MODEL_PATH="${ENTRY%%:*}"
    REST="${ENTRY#*:}"
    OUTPUT_NAME="${REST%%:*}"
    MODALITY="${REST##*:}"
    OUTPUT_DIR="$RESULTS_BASE/$OUTPUT_NAME"

    echo "=== $OUTPUT_NAME ==="

    python generate_qwen_omni.py \
        --model-name "$MODEL_PATH" \
        --dataset "$DATASET" \
        --split dev \
        --output-name "$OUTPUT_DIR" \
        --modality "$MODALITY"

    cd ../../speechcomet-eval/iwslt26-metrics/
    for scores_file in ../../$OUTPUT_DIR/output_scores_dev_*_${MODALITY}.jsonl; do
        lang_pair=$(basename "$scores_file" .jsonl | sed "s/output_scores_dev_//;s/_${MODALITY}$//")
        input_file="../../$OUTPUT_DIR/input_data_dev_${lang_pair}.jsonl"
        echo "Evaluating $lang_pair ..."
        python evaluation/__main__.py -i "$input_file" -m "$scores_file"
    done
    cd ../../baselines-speechllm/src
done
