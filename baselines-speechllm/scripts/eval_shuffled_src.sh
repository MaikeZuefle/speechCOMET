#!/bin/bash
# Evaluate Qwen2.5-Omni models with shuffled source inputs.
# Each model is run with the modality it was trained on.
# Run from baselines-speechllm/: bash scripts/eval_shuffled_src.sh

cd "$(dirname "$0")/.." || exit 1

FT_TEXT="../saves/qwen2.5-omni-7b/merged/iwslt26_text"
FT_AUDIO="../saves/qwen2.5-omni-7b/merged/iwslt26_audio"
FT_TEXTAUDIO="../saves/qwen2.5-omni-7b/merged/iwslt26_textaudio"

DATASET="maikezu/iwslt2026-metrics-shared-train-dev"
SPLIT=dev
RESULTS_BASE="baselines-speechllm/results"

# format: model_path:output_name:modality
MODELS=(
    # zero-shot base model — one entry per modality
    "Qwen/Qwen2.5-Omni-7B:Qwen_Qwen2.5-Omni-7B:text"
    "Qwen/Qwen2.5-Omni-7B:Qwen_Qwen2.5-Omni-7B:audio"
    "Qwen/Qwen2.5-Omni-7B:Qwen_Qwen2.5-Omni-7B:audiotext"
    # fine-tuned models — trained modality only
    "$FT_TEXT:Qwen_Qwen2.5-Omni-7B-iwslt26-text:text"
    "$FT_AUDIO:Qwen_Qwen2.5-Omni-7B-iwslt26-audio:audio"
    "$FT_TEXTAUDIO:Qwen_Qwen2.5-Omni-7B-iwslt26-textaudio:audiotext"
)

cd src || exit 1

for ENTRY in "${MODELS[@]}"; do
    MODEL_PATH="${ENTRY%%:*}"
    REST="${ENTRY#*:}"
    OUTPUT_NAME="${REST%%:*}"
    MODALITY="${REST##*:}"
    OUTPUT_DIR="$RESULTS_BASE/$OUTPUT_NAME"

    echo ""
    echo "======================================================"
    echo "Model: $OUTPUT_NAME  modality: $MODALITY"
    echo "======================================================"

    python generate_qwen_omni_shuffled_src.py \
        --model-name "$MODEL_PATH" \
        --output-name "$OUTPUT_DIR" \
        --dataset "$DATASET" \
        --split "$SPLIT" \
        --modality "$MODALITY"
done

echo ""
echo "All SpeechLLM shuffled-source evaluations done."
