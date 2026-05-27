#!/bin/bash
# IWSLT correlation evaluation.

# format: model_name:modality
MODELS=(
    audio-sonar:audio
    audio-whisper:audio
    text:text
    audiotext-sonar:audiotext
)

SPLIT=dev
DATASET=maikezu/iwslt2026-metrics-shared-train-dev
CHECKPOINT_FOLDER=trained_models

for ENTRY in "${MODELS[@]}"; do
    IFS=: read -r model modality <<< "$ENTRY"
    OUTPUT_DIR="$CHECKPOINT_FOLDER/$model"

    echo ""
    echo "=== $model ==="

    python speechcomet-eval/iwslt_eval.py \
        --model-folder "$OUTPUT_DIR" \
        --modality "$modality" \
        --split "$SPLIT" \
        --dataset "$DATASET"

    cd speechcomet-eval/iwslt26-metrics/
    for scores_file in ../../$OUTPUT_DIR/output_scores_${SPLIT}_*.jsonl; do
        lang_pair=$(basename "$scores_file" .jsonl | sed "s/output_scores_${SPLIT}_//")
        input_file="../../$OUTPUT_DIR/input_data_${SPLIT}_${lang_pair}.jsonl"
        corr_file="../../$OUTPUT_DIR/correlation_${SPLIT}_${lang_pair}.txt"
        echo "Evaluating $lang_pair ..."
        python evaluation/__main__.py -i "$input_file" -m "$scores_file" | tee "$corr_file"
    done
    cd ../..
done
