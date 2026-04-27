#!/bin/bash
# Evaluate SpeechCOMET models with shuffled source inputs.
# Run from repo root: bash evaluation/scripts/shuffled_src/run_shuffled_src.sh

CHECKPOINT_FOLDER=trained_models
SPLIT=dev_asr

# format: model_name:modality
MODELS=(
    # harris-FT-sonar:audio
    # harris-20ep:audio
    # lewis-10ep:text
    # skye-20ep:text
    # shetland-20ep:audio
    # mull-avg-20ep:audio
    # mull-attn-10ep:audio
    # mull-attn-lora-10ep:audio
    # mull-avg-lora-10ep:audio
    #shetland-FT-sonar:audio
    # orkney-sum-from-text-ckpt-20ep:audiotext
    bute-pretrain:audio
    # orkney-avg-20ep:audiotext
    # orkney-sum-20ep:audiotext
    # orkney-concat-20ep:audiotext
)

for ENTRY in "${MODELS[@]}"; do
    MODEL_NAME="${ENTRY%%:*}"
    MODALITY="${ENTRY##*:}"
    OUTPUT_DIR="$CHECKPOINT_FOLDER/$MODEL_NAME"

    echo "=== $MODEL_NAME ($MODALITY) ==="

    python evaluation/run_inf_shuffled_src.py \
        --model-folder "$OUTPUT_DIR" \
        --modality "$MODALITY" \
        --split "$SPLIT"

    # correlation evaluation
    cd evaluation/iwslt26-metrics/
    for scores_file in "../../$OUTPUT_DIR/shuffled_src/output_scores_${SPLIT}_"*.jsonl; do
        lang_pair=$(basename "$scores_file" .jsonl | sed "s/output_scores_${SPLIT}_//")
        input_file="../../$OUTPUT_DIR/shuffled_src/input_data_${SPLIT}_${lang_pair}.jsonl"
        echo "Evaluating $lang_pair (shuffled src) ..."
        python evaluation/__main__.py -i "$input_file" -m "$scores_file"
    done
    cd ../..
done
