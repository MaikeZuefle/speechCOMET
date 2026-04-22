#!/bin/bash
# Compute segment- and system-level correlations for FT Qwen models.
# Each model is evaluated on its own modality (text/audio/audiotext).
# Run from repo root: bash speechllm-baselines/scripts/run_correlation_eval.sh

EVAL_DIR="evaluation/iwslt26-metrics"
BASE="speechllm-baselines/results"

# format: output_name:modality
MODELS=(
    "Qwen_Qwen2.5-Omni-7B-iwslt26-text:text"
    "Qwen_Qwen2.5-Omni-7B-iwslt26-audio:audio"
    "Qwen_Qwen2.5-Omni-7B-iwslt26-textaudio:audiotext"
)

for ENTRY in "${MODELS[@]}"; do
    MODEL_NAME="${ENTRY%%:*}"
    MODALITY="${ENTRY##*:}"
    MODEL_DIR="$BASE/$MODEL_NAME"

    echo ""
    echo "======================================================"
    echo "Model: $MODEL_NAME  (modality: $MODALITY)"
    echo "======================================================"

    for SPLIT in dev dev_asr; do
        INPUT_COMBINED=$(mktemp)
        SCORES_COMBINED=$(mktemp)

        for LANG_PAIR in en-de en-zh; do
            INPUT_FILE="$MODEL_DIR/input_data_${SPLIT}_${LANG_PAIR}.jsonl"
            SCORES_FILE="$MODEL_DIR/output_scores_${SPLIT}_${LANG_PAIR}_${MODALITY}.jsonl"

            if [[ ! -f "$INPUT_FILE" || ! -f "$SCORES_FILE" ]]; then
                echo "  WARNING: missing files for $SPLIT $LANG_PAIR, skipping"
                continue
            fi
            cat "$INPUT_FILE"  >> "$INPUT_COMBINED"
            cat "$SCORES_FILE" >> "$SCORES_COMBINED"
        done

        if [[ ! -s "$INPUT_COMBINED" ]]; then
            echo "  No data for split=$SPLIT, skipping"
            rm -f "$INPUT_COMBINED" "$SCORES_COMBINED"
            continue
        fi

        echo "--- $SPLIT ---"
        CORR_OUTPUT=$(python evaluation/__main__.py \
            -i "$(realpath "$INPUT_COMBINED")" \
            -m "$(realpath "$SCORES_COMBINED")" 2>&1)
        echo "$CORR_OUTPUT"

        CORR_FILE="$MODEL_DIR/correlation_${SPLIT}_${MODALITY}.txt"
        echo "$CORR_OUTPUT" > "$CORR_FILE"
        echo "  Saved to $CORR_FILE"

        rm -f "$INPUT_COMBINED" "$SCORES_COMBINED"
    done
done

echo ""
echo "Done."
