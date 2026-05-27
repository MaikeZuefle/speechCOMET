#!/bin/bash
# Evaluate fine-tuned SpeechLLM (audiotext) on IWSLT test set.
cd "$(dirname "$0")/../.." || exit 1

DATASET="maikezu/iwslt2026-metrics-shared-test"
RESULTS_BASE="baselines-speechllm/results"
AUDIO_BASE_DIR=iwslt2026data

FT_TEXTAUDIO="saves/qwen2.5-omni-7b/merged/iwslt26_textaudio"
OUTPUT_NAME="Qwen_Qwen2.5-Omni-7B-iwslt26-textaudio"
OUTPUT_DIR="$RESULTS_BASE/$OUTPUT_NAME"

echo "=== $OUTPUT_NAME ==="

python iwslt2026/eval_speechllm.py \
    --model-name "baselines-speechllm/$FT_TEXTAUDIO" \
    --dataset "$DATASET" \
    --split test \
    --output-name "$OUTPUT_DIR" \
    --audio-base-dir "$AUDIO_BASE_DIR"
