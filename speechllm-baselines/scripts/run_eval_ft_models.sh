#!/bin/bash
# Evaluate all three Qwen models (base + FT text/audio/textaudio) on
# dev, dev_asr, MuST-SHE, and ContraProST.
# Run from speechllm-baselines/: bash scripts/run_eval_all_models.sh

cd "$(dirname "$0")/.." || exit 1

SCRIPTS_DIR="$(dirname "$0")"
SRC_DIR="$(dirname "$0")/../src"

FT_TEXT="../saves/qwen2.5-omni-7b/merged/iwslt26_text"
FT_AUDIO="../saves/qwen2.5-omni-7b/merged/iwslt26_audio"
FT_TEXTAUDIO="../saves/qwen2.5-omni-7b/merged/iwslt26_textaudio"

# format: path:output_name:modality
MODELS=(
    "$FT_TEXT:Qwen_Qwen2.5-Omni-7B-iwslt26-text:text"
    "$FT_AUDIO:Qwen_Qwen2.5-Omni-7B-iwslt26-audio:audio"
    "$FT_TEXTAUDIO:Qwen_Qwen2.5-Omni-7B-iwslt26-textaudio:audiotext"
)

DATASET="maikezu/scottish-metrics"
MUSTSHE_DIR="data/MuST-SHE_v1.2/MuST-SHE-v1.2-data/tsv"
CONTRAPROST_DIR="data/contraProST"
RESULTS_BASE="speechllm-baselines/results"

cd src || exit 1

for ENTRY in "${MODELS[@]}"; do
    MODEL_PATH="${ENTRY%%:*}"
    REST="${ENTRY#*:}"
    OUTPUT_NAME="${REST%%:*}"
    MODALITY="${REST##*:}"
    OUTPUT_DIR="$RESULTS_BASE/$OUTPUT_NAME"

    echo ""
    echo "======================================================"
    echo "Model: $OUTPUT_NAME  ($MODEL_PATH)"
    echo "======================================================"

    echo "--- dev ---"
    python generate_qwen_omni.py \
        --model-name "$MODEL_PATH" \
        --output-name "$OUTPUT_DIR" \
        --dataset "$DATASET" \
        --split dev \
        --modality "$MODALITY"

    cd ../../evaluation/iwslt26-metrics/
    for scores_file in ../../$OUTPUT_DIR/output_scores_dev_*_${MODALITY}.jsonl; do
        lang_pair=$(basename "$scores_file" .jsonl | sed "s/output_scores_dev_//;s/_${MODALITY}$//")
        input_file="../../$OUTPUT_DIR/input_data_dev_${lang_pair}.jsonl"
        echo "Evaluating dev $lang_pair ..."
        python evaluation/__main__.py -i "$input_file" -m "$scores_file"
    done
    cd ../../src

    echo "--- dev_asr ---"
    python generate_qwen_omni.py \
        --model-name "$MODEL_PATH" \
        --output-name "$OUTPUT_DIR" \
        --dataset "$DATASET" \
        --split dev_asr \
        --modality "$MODALITY"

    cd ../../evaluation/iwslt26-metrics/
    for scores_file in ../../$OUTPUT_DIR/output_scores_dev_asr_*_${MODALITY}.jsonl; do
        lang_pair=$(basename "$scores_file" .jsonl | sed "s/output_scores_dev_asr_//;s/_${MODALITY}$//")
        input_file="../../$OUTPUT_DIR/input_data_dev_asr_${lang_pair}.jsonl"
        echo "Evaluating dev_asr $lang_pair ..."
        python evaluation/__main__.py -i "$input_file" -m "$scores_file"
    done
    cd ../../src

    echo "--- MuST-SHE ---"
    python generate_qwen_omni.py \
        --model-name "$MODEL_PATH" \
        --output-name "$OUTPUT_DIR" \
        --mustshe-dir "../../$MUSTSHE_DIR" \
        --modality "$MODALITY"

    echo "--- ContraProST ---"
    python generate_qwen_omni.py \
        --model-name "$MODEL_PATH" \
        --output-name "$OUTPUT_DIR" \
        --contraprost-dir "../../$CONTRAPROST_DIR" \
        --modality "$MODALITY"
done

echo ""
echo "All evaluations done."
