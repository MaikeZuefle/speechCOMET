#!/bin/bash
cd "$(dirname "$0")/../src" || exit 1

MUSTSHE_DIR="../../data/MuST-SHE_v1.2/MuST-SHE-v1.2-data/tsv"
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

    echo "=== $OUTPUT_NAME ==="

    # Standard prompt
    python generate_qwen_omni.py \
        --model-name "$MODEL_PATH" \
        --mustshe-dir "$MUSTSHE_DIR" \
        --output-name "$RESULTS_BASE/$OUTPUT_NAME" \
        --modality "$MODALITY"

    # Speech-aware prompt (instructs model to consider paralinguistic cues)
    python generate_qwen_omni.py \
        --model-name "$MODEL_PATH" \
        --mustshe-dir "$MUSTSHE_DIR" \
        --output-name "$RESULTS_BASE/$OUTPUT_NAME" \
        --modality "$MODALITY" \
        --prompt mustshe_gender
done
