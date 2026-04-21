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

MODELS=(
    "$FT_TEXT:Qwen_Qwen2.5-Omni-7B-iwslt26-text"
    "$FT_AUDIO:Qwen_Qwen2.5-Omni-7B-iwslt26-audio"
    "$FT_TEXTAUDIO:Qwen_Qwen2.5-Omni-7B-iwslt26-textaudio"
)

DATASET="maikezu/scottish-metrics"
MUSTSHE_DIR="data/MuST-SHE_v1.2/MuST-SHE-v1.2-data/tsv"
CONTRAPROST_DIR="data/contraProST"

cd src || exit 1

for ENTRY in "${MODELS[@]}"; do
    MODEL_PATH="${ENTRY%%:*}"
    OUTPUT_NAME="${ENTRY##*:}"

    echo ""
    echo "======================================================"
    echo "Model: $OUTPUT_NAME  ($MODEL_PATH)"
    echo "======================================================"

    echo "--- dev ---"
    python generate_qwen_omni.py \
        --model-name "$MODEL_PATH" \
        --output-name "$OUTPUT_NAME" \
        --dataset "$DATASET" \
        --split dev

    echo "--- dev_asr ---"
    python generate_qwen_omni.py \
        --model-name "$MODEL_PATH" \
        --output-name "$OUTPUT_NAME" \
        --dataset "$DATASET" \
        --split dev_asr

    echo "--- MuST-SHE ---"
    python generate_qwen_omni.py \
        --model-name "$MODEL_PATH" \
        --output-name "$OUTPUT_NAME" \
        --mustshe-dir "../../$MUSTSHE_DIR"

    echo "--- ContraProST ---"
    python generate_qwen_omni.py \
        --model-name "$MODEL_PATH" \
        --output-name "$OUTPUT_NAME" \
        --contraprost-dir "../../$CONTRAPROST_DIR"
done

echo ""
echo "All evaluations done."
